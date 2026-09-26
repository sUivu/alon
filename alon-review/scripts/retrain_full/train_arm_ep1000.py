#!/usr/bin/env python3
"""day1_1000ep: 1000-epoch retrain of dual_overnight arms (T1) for the
PROTEINS / WS_100_200 / RB200 head-to-head against OptGNN ep1000.

Verbatim copy of scripts/dual_overnight/train_arm.py (NOT modified, so the
validated harness / epoch-0 loss match is preserved) with ONLY the output
namespace patched and two convenience args added:
  ckpts/retrain_full/<run_name>/
  results/retrain_full/eval_<run_name>.json

run_name defaults to '<dataset>_<arm>_ep<epochs>', i.e.
retrain_full/PROTEINS_T1_ep1000 (prefix requested in the task).

Extra args (do NOT alter the training loop):
  --lambda_freq N   override the arm's dual-update interval (T2b: 2)
  --run-name NAME   override the checkpoint/results tag
  --result-path P   override the eval JSON path (T2b -> dual_tuning/sweep)

Usage:
  CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 \
  python scripts/retrain_full/train_arm_ep1000.py \
      --arm T1 --dataset PROTEINS --seed 0 --epochs 1000
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import random_split

HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parent.parent
DUAL_OVERNIGHT = PROJECT_ROOT / 'scripts' / 'dual_overnight'
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(DUAL_OVERNIGHT))
sys.path.insert(0, str(PROJECT_ROOT / 'scripts' / 'dual_tuning'))
sys.path.insert(0, str(HERE))

from overnight_configs import get_arm_cfg  # noqa: E402
from overnight_model import build_overnight_network  # noqa: E402

from data.loader import construct_dataset, prepare_alon_dataset  # noqa: E402
from model.alon_a_fair import (  # noqa: E402
    ALONA_Fair, make_batch_graphs, edge_residual, node_violation,
    round_repair_batch, eval_config, summarize, _PHI, _PHI2,
)
from model.alon_a import build_split_graphs  # noqa: E402
from train_alon_upgrade import (  # noqa: E402
    DATASET_SPECS, build_ns, seed_all, make_batches, load_reference,
)
from train_sweep import c_violation, tuned_dual_step  # noqa: E402
from problem.problems import get_problem  # noqa: E402


def al_loss_full(x_out, batch, mu_batch, problem, cfg, penalty):
    phi = edge_residual(x_out, batch, cfg['residual'])
    pen = 0.5 * (phi ** 2).sum() if phi.numel() > 0 else x_out.sum() * 0.0
    loss = problem.objective(x_out, batch) + \
        penalty * 4.0 / _PHI2[cfg['residual']] * pen
    if mu_batch is not None:
        c = c_violation(x_out, batch, cfg)
        loss = loss + (mu_batch * c).sum() / _PHI[cfg['residual']]
    return loss


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--arm', required=True)
    ap.add_argument('--dataset', default='PROTEINS',
                    choices=list(DATASET_SPECS))
    ap.add_argument('--epochs', type=int, default=200)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--lr', type=float, default=1e-3)
    ap.add_argument('--grad_clip', type=float, default=5.0)
    # Penalty schedule: linearly annealed +0.5 -> +2.5 (paper requires rho > 0);
    # explicit --penalty_s/--penalty_e values override the default.
    ap.add_argument('--penalty_s', type=float, default=0.5)
    ap.add_argument('--penalty_e', type=float, default=2.5)
    ap.add_argument('--num_layers', type=int, default=16)
    ap.add_argument('--lift_ratio', type=float, default=0.1)
    ap.add_argument('--skip_train', action='store_true')
    ap.add_argument('--lambda_freq', type=int, default=None,
                    help='override arm dual-update interval (T2b: 2)')
    ap.add_argument('--run-name', default=None)
    ap.add_argument('--result-path', default=None)
    args = ap.parse_args()

    cfg = get_arm_cfg(args.arm)
    if args.lambda_freq is not None:
        cfg['lambda_freq'] = args.lambda_freq
    run_name = args.run_name or f'{args.dataset}_{args.arm}_ep{args.epochs}'
    save_dir = os.path.join('ckpts', 'retrain_full', run_name)
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs('results/retrain_full', exist_ok=True)
    out_path = args.result_path or \
        f'results/retrain_full/eval_{run_name}.json'
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    log_path = os.path.join(save_dir, 'train.log')
    _fh = open(log_path, 'a')

    def log(msg):
        line = f'{time.strftime("%H:%M:%S")} {msg}'
        print(line, flush=True)
        _fh.write(line + '\n')
        _fh.flush()

    seed_all(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ns = build_ns(args)
    log(f'[cmd] {" ".join(sys.argv)}')
    log(f'[cfg] {cfg}')

    dataset = construct_dataset(ns)
    dataset, total_nodes = prepare_alon_dataset(dataset)
    n = len(dataset)
    tr = int(ns.train_fraction * n)
    va = (n - tr) // 2
    gen = torch.Generator().manual_seed(ns.split_seed)
    train_sub, val_sub, test_sub = random_split(
        dataset, [tr, va, n - tr - va], generator=gen)
    train_data = [dataset[int(i)] for i in train_sub.indices]
    log(f'[data] n={n} train={tr} val={va} test={n - tr - va}')

    problem = get_problem(ns)

    mu_param = torch.zeros(total_nodes, device=device)
    mu_prev = torch.zeros_like(mu_param)
    tinfo = {}
    if args.skip_train:
        ck = torch.load(os.path.join(save_dir, 'alon_upgrade.pt'),
                        map_location='cpu', weights_only=False)
        net = build_overnight_network(cfg, problem, rank=ns.rank,
                                      num_layers=ns.num_layers,
                                      lift_ratio=ns.lift_ratio,
                                      create_graph=False)
        net.load_state_dict(ck['net'])
        net = net.to(device)
        tinfo = ck['train_info']
        log(f'[skip_train] loaded {save_dir}/alon_upgrade.pt')
    else:
        net = build_overnight_network(cfg, problem, rank=ns.rank,
                                      num_layers=ns.num_layers,
                                      lift_ratio=ns.lift_ratio,
                                      create_graph=True).to(device)
        n_params = sum(p.numel() for p in net.parameters())
        log(f'[model] arm={args.arm} residual={cfg["residual"]} '
            f'norm={cfg["norm"]} c_signal={cfg["c_signal"]} params={n_params}')
        opt = torch.optim.Adam(net.parameters(), lr=args.lr)
        net.train()
        add_mu = torch.zeros_like(mu_param)
        rho_t, c_mean_prev = cfg['rho_init'], None
        loss_hist, mu_norm_hist, viol_hist, rho_hist = [], [], [], []
        t0 = time.time()
        for ep in range(args.epochs):
            penalty = args.penalty_s + (args.penalty_e - args.penalty_s) * \
                ep / max(args.epochs - 1, 1)
            batches = make_batches(train_data, DATASET_SPECS[
                args.dataset]['batch_size'], args.seed * 100003 + ep)
            add_mu.zero_()
            nb = len(batches)
            tot, cnt, ep_viol = 0.0, 0, 0.0
            for bi, blist in enumerate(batches):
                from torch_geometric.data import Batch
                batch = Batch.from_data_list(blist).to(device)
                batch.penalty = float(penalty)
                torch.manual_seed(args.seed * 1000003 + ep * 10007 + bi)
                x_in = F.normalize(
                    torch.randn(batch.num_nodes, ns.rank, device=device),
                    dim=1)
                mu_batch = mu_param[batch.node_global_idx]
                x_out = net(x_in, batch, mu_batch)
                loss = al_loss_full(x_out, batch, mu_batch, problem, cfg,
                                    penalty=float(penalty))
                opt.zero_grad()
                loss.backward()
                if args.grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(net.parameters(),
                                                   args.grad_clip)
                opt.step()
                with torch.no_grad():
                    c = c_violation(x_out.detach(), batch, cfg)
                    add_mu[batch.node_global_idx] += c.detach()
                    ep_viol += float(c.clamp(min=0).mean())
                tot += float(loss) * batch.num_graphs
                cnt += batch.num_graphs
            if cfg['lambda_freq'] and \
                    (ep + 1) % cfg['lambda_freq'] == 0 and \
                    ep >= cfg['warmup']:
                mu_param, rho_t, c_mean_prev = tuned_dual_step(
                    mu_param, add_mu, nb, cfg, rho_t, c_mean_prev, mu_prev)
            loss_hist.append(tot / max(cnt, 1))
            mu_norm_hist.append(float(mu_param.norm()))
            viol_hist.append(ep_viol / max(nb, 1))
            rho_hist.append(rho_t)
            if ep % max(args.epochs // 10, 1) == 0 or ep == args.epochs - 1:
                log(f'  ep={ep:4d} pen={penalty:6.3f} '
                    f'loss={loss_hist[-1]:9.4f} |mu|={mu_norm_hist[-1]:8.3f} '
                    f'c+={viol_hist[-1]:7.4f} rho={rho_t:6.3f} '
                    f't={time.time() - t0:6.1f}s')
        train_time = time.time() - t0
        log(f'[train] done {args.epochs} epochs in {train_time:.1f}s')
        tinfo = {'loss_history': loss_hist,
                 'mu_norm_history': mu_norm_hist,
                 'viol_history': viol_hist,
                 'rho_history': rho_hist,
                 'train_seconds': train_time, 'total_nodes': total_nodes,
                 'rho_final': rho_t, 'c_mean_final': c_mean_prev,
                 'mu_norm_final': mu_norm_hist[-1],
                 'mu_max_final': float(mu_param.max()),
                 'mu_mean_final': float(mu_param.mean())}
        torch.save({'arm': args.arm, 'cfg': cfg, 'dataset': args.dataset,
                    'seed': args.seed, 'command': sys.argv,
                    'net': net.state_dict(), 'train_info': tinfo},
                   os.path.join(save_dir, 'alon_upgrade.pt'))
        log(f'[saved] {save_dir}/alon_upgrade.pt')
        net.eval()

    # ---------------------------------------------------------------- eval
    ref_per = load_reference(args.dataset)
    graphs, _ = build_split_graphs(ns, split='test', limit=0)
    x_in, big, offsets = make_batch_graphs(graphs, ns.rank, device, seed=0)
    alon = ALONA_Fair(net, problem, rank=ns.rank).to(device)
    alon.eval()
    split_out = {'n_test_graphs': len(graphs), 'runs': []}

    def run(name, K, rho, statusquo=False):
        rows, mh, rh = eval_config(
            alon, x_in, big, graphs, offsets, K, rho, use_mu_phi=False,
            adaptive=True, statusquo=statusquo)
        r = summarize(name, rows, K, rho, mu_norm_traj=mh, resid_traj=rh)
        if ref_per:
            vals = [ref_per.get(int(i)) for _, _, i in graphs
                    if ref_per.get(int(i)) is not None]
            if vals:
                rv = float(np.mean(vals))
                r['gap_vs_ref'] = (r['mean_repaired'] - rv) / rv
        split_out['runs'].append(r)
        log(f'[eval] {name:<16} rep_mean={r["mean_repaired"]:8.3f} '
            f'infeas={r["frac_infeasible_raw"]:.3f} '
            f'gap={r.get("gap_vs_ref")}')

    run('statusquo', None, 0.0, statusquo=True)
    run('asc_K5_r0.02', 5, 0.02)

    out = {
        'arm': args.arm, 'cfg': cfg, 'dataset': args.dataset,
        'seed': args.seed, 'checkpoint':
            os.path.join(save_dir, 'alon_upgrade.pt'),
        'train_info': {k: v for k, v in tinfo.items()
                       if not k.endswith('history')},
        'splits': {'test': split_out},
    }
    json.dump(out, open(out_path, 'w'), indent=1)
    sq = split_out['runs'][0]
    log(f'[done] statusquo rep={sq["mean_repaired"]:.3f} '
        f'infeas={sq["frac_infeasible_raw"]:.3f} -> {out_path}')
    _fh.close()


if __name__ == '__main__':
    main()
