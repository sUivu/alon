#!/usr/bin/env python3
"""ALON upgrade campaign -- train one (arm, dataset, seed) FROM SCRATCH and
evaluate under the corrected protocol in the same process.

Arms D0-D4 see model/alon_upgrade.py.  Everything except the named mechanism
changes is identical across arms:
  * paper split (train_fraction=0.8, split_seed=0, test = last 10%)
  * 200 epochs, lr 1e-3, grad_clip 5, penalty annealing -0.5 -> 2.5 linear
    (the released ALONGNN schedule), batch size per dataset (PROTEINS 9,
    WS 16, RB200 32), rank 32 (RB200 64), 16 layers, lift_ratio 0.1
  * seed controls data order / init / feature draws
  * node-wise dual lambda_ij = mu_i + mu_j (never materialized per edge)

Eval (same process, test split): randomized-hyperplane rounding seed 0 with
1000 hyperplanes -> greedy repair (violation == 0 asserted).
  statusquo      : mu = 0 (paper protocol)          <- PRIMARY metric
  asc_K*_r*      : free dual ascent at inference    <- secondary (H4 lever)

Isolated namespace: writes only ckpts/alon_upgrade_* and results/alon_upgrade/.

Usage
-----
  CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=2 \
  python train_alon_upgrade.py \
      --arm D3 --dataset WS --epochs 200 --seed 0
"""
import argparse
import json
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import random_split

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from data.loader import construct_dataset, prepare_alon_dataset  # noqa: E402
from problem.problems import get_problem  # noqa: E402
from model.alon_a_fair import (  # noqa: E402
    FairGradLayer, make_batch_graphs, eval_config, summarize, al_loss,
    node_violation, ALONA_Fair,
)
from model.alon_upgrade import (  # noqa: E402
    UPGRADE_ARMS, ARM_ORDER, arm_spec, build_network, dual_step,
)
from model.alon_a import build_split_graphs  # noqa: E402

# dataset -> generator kwargs + per-dataset campaign-matching config
DATASET_SPECS = {
    'PROTEINS': dict(dataset='PROTEINS', num_graphs=2000,
                     gen_n=None, gen_m=None, gen_k=None, gen_p=None,
                     batch_size=9, rank=32),
    'WS': dict(dataset='WattsStrogatz', num_graphs=2000,
               gen_n=[100, 200], gen_m=None, gen_k=[4], gen_p=[0.25],
               batch_size=16, rank=32),
    'RB200': dict(dataset='ForcedRB', num_graphs=1000,
                  gen_n=[6, 15], gen_m=None, gen_k=[12, 21], gen_p=None,
                  batch_size=32, rank=64),
    'RB500': dict(dataset='ForcedRB', num_graphs=1000,
                  gen_n=[20, 34], gen_m=None, gen_k=[10, 29], gen_p=None,
                  batch_size=16, rank=64),
    'COLLAB': dict(dataset='COLLAB', num_graphs=2000,
                   gen_n=None, gen_m=None, gen_k=None, gen_p=None,
                   batch_size=40, rank=32),
}
REF_NAME = {'PROTEINS': 'PROTEINS', 'WS': 'WS_100_200', 'RB200': 'RB200',
            'RB500': 'RB500', 'COLLAB': 'COLLAB'}

# dual-schedule hyperparameters for the 'alg1' arms (identical across arms)
ALG1_KW = dict(rho_init=0.01, delta=0.5, tau=2.0, rho_max=0.25)


def build_ns(cli):
    spec = DATASET_SPECS[cli.dataset]
    return argparse.Namespace(
        problem_type='vertex_cover', dataset=spec['dataset'],
        num_graphs=spec['num_graphs'], gen_n=spec['gen_n'],
        gen_m=spec['gen_m'], gen_k=spec['gen_k'], gen_p=spec['gen_p'],
        data_seed=0, parallel=0, infinite=False, positional_encoding=None,
        pe_dimension=8, train_fraction=0.8, split_seed=0, rank=spec['rank'],
        num_layers=cli.num_layers, lift_ratio=cli.lift_ratio,
        hidden_channels=spec['rank'])


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def make_batches(data_list, batch_size, seed):
    from torch_geometric.data import Batch
    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(len(data_list), generator=g).tolist()
    return [data_list[i:i + batch_size]
            for i in range(0, len(perm), batch_size)]
    _ = Batch  # imported lazily in loop


def load_reference(dataset):
    p = PROJECT_ROOT / 'results' / 'classical_baselines' / \
        f'{REF_NAME[dataset]}.json'
    if not p.exists():
        return None
    d = json.load(open(p))
    per = {}
    for g in d.get('per_graph', []):
        v = g.get('akiba_iwata')
        if isinstance(v, dict):
            v = v.get('size')
        if v is not None:
            per[int(g['idx'])] = v
    return per


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--arm', required=True, choices=ARM_ORDER)
    ap.add_argument('--dataset', required=True, choices=list(DATASET_SPECS))
    ap.add_argument('--epochs', type=int, default=200)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--lr', type=float, default=1e-3)
    ap.add_argument('--grad_clip', type=float, default=5.0)
    ap.add_argument('--penalty_s', type=float, default=-0.125)
    ap.add_argument('--penalty_e', type=float, default=0.625)
    ap.add_argument('--lambda_freq', type=int, default=None,
                    help='dual update interval; default per arm '
                         '(D0: 250, others: 25)')
    ap.add_argument('--lambda_ratio', type=float, default=0.01)
    ap.add_argument('--num_layers', type=int, default=16)
    ap.add_argument('--lift_ratio', type=float, default=0.1)
    ap.add_argument('--save_dir', default=None)
    ap.add_argument('--eval_only', default=None,
                    help='checkpoint path: skip training, eval only')
    ap.add_argument('--skip_eval', action='store_true')
    ap.add_argument('--eval_splits', nargs='+', default=['test'],
                    choices=['val', 'test'])
    ap.add_argument('--log_file', default=None)
    cli = ap.parse_args()

    spec = arm_spec(cli.arm)
    if cli.lambda_freq is None:
        cli.lambda_freq = 250 if spec['dual'] == 'released' else 25
    tag = f'alon_upgrade_{cli.arm}_{cli.dataset}_s{cli.seed}'
    save_dir = cli.save_dir or os.path.join('ckpts', tag)
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs('results/alon_upgrade', exist_ok=True)
    log_path = cli.log_file or os.path.join(save_dir, 'train.log')
    _fh = open(log_path, 'a')

    def log(msg):
        line = f'{time.strftime("%H:%M:%S")} {msg}'
        print(line, flush=True)
        _fh.write(line + '\n')
        _fh.flush()

    seed_all(cli.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ns = build_ns(cli)
    log(f'[cmd] {" ".join(sys.argv)}')
    log(f'[arm] {cli.arm} {spec} lambda_freq={cli.lambda_freq} '
        f'lambda_ratio={cli.lambda_ratio} alg1={ALG1_KW}')

    # ---------------------------------------------------------------- data
    dataset = construct_dataset(ns)
    dataset, total_nodes = prepare_alon_dataset(dataset)
    n = len(dataset)
    tr = int(ns.train_fraction * n)
    va = (n - tr) // 2
    gen = torch.Generator().manual_seed(ns.split_seed)
    train_sub, val_sub, test_sub = random_split(
        dataset, [tr, va, n - tr - va], generator=gen)
    train_data = [dataset[int(i)] for i in train_sub.indices]
    log(f'[data] n={n} train={tr} val={va} test={n - tr - va} '
        f'total_nodes={total_nodes}')

    problem = get_problem(ns)

    # -------------------------------------------------------------- train
    mu_param = torch.zeros(total_nodes, device=device)
    tinfo = {}
    if cli.eval_only:
        ck = torch.load(cli.eval_only, map_location='cpu', weights_only=False)
        net = build_network(cli.arm, problem, rank=ns.rank,
                            num_layers=ns.num_layers,
                            lift_ratio=ns.lift_ratio, create_graph=False)
        net.load_state_dict(ck['net'])
        net = net.to(device)
        log(f'[eval_only] loaded {cli.eval_only}')
    else:
        net = build_network(cli.arm, problem, rank=ns.rank,
                            num_layers=ns.num_layers,
                            lift_ratio=ns.lift_ratio, create_graph=True).to(
                                device)
        n_params = sum(p.numel() for p in net.parameters())
        log(f'[model] arm={cli.arm} residual={spec["residual"]} '
            f'norm={spec["norm"]} dual={spec["dual"]} params={n_params}')
        opt = torch.optim.Adam(net.parameters(), lr=cli.lr)
        net.train()
        add_mu = torch.zeros_like(mu_param)
        rho_t, c_mean_prev = ALG1_KW['rho_init'], None
        loss_hist, mu_norm_hist = [], []
        t0 = time.time()
        for ep in range(cli.epochs):
            penalty = cli.penalty_s + (cli.penalty_e - cli.penalty_s) * \
                ep / max(cli.epochs - 1, 1)
            batches = make_batches(train_data, DATASET_SPECS[
                cli.dataset]['batch_size'], cli.seed * 100003 + ep)
            add_mu.zero_()
            nb = len(batches)
            tot, cnt = 0.0, 0
            for bi, blist in enumerate(batches):
                from torch_geometric.data import Batch
                batch = Batch.from_data_list(blist).to(device)
                batch.penalty = float(penalty)
                torch.manual_seed(cli.seed * 1000003 + ep * 10007 + bi)
                x_in = F.normalize(
                    torch.randn(batch.num_nodes, ns.rank, device=device),
                    dim=1)
                mu_batch = mu_param[batch.node_global_idx]
                x_out = net(x_in, batch, mu_batch)
                loss = al_loss(x_out, batch, mu_batch, problem,
                               spec['residual'], penalty=float(penalty))
                opt.zero_grad()
                loss.backward()
                if cli.grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(net.parameters(),
                                                   cli.grad_clip)
                opt.step()
                with torch.no_grad():
                    c = node_violation(x_out.detach(), batch, spec['residual'])
                    add_mu[batch.node_global_idx] += c.detach()
                tot += float(loss) * batch.num_graphs
                cnt += batch.num_graphs
            if cli.lambda_freq and (ep + 1) % cli.lambda_freq == 0:
                mu_param, rho_t, c_mean_prev = dual_step(
                    mu_param, add_mu, nb, spec['dual'], rho_t, c_mean_prev,
                    lambda_ratio=cli.lambda_ratio, residual=spec['residual'],
                    **ALG1_KW)
            loss_hist.append(tot / max(cnt, 1))
            mu_norm_hist.append(float(mu_param.norm()))
            if ep % max(cli.epochs // 10, 1) == 0 or ep == cli.epochs - 1:
                log(f'  ep={ep:4d} pen={penalty:6.3f} loss={loss_hist[-1]:9.4f} '
                    f'|mu|={mu_norm_hist[-1]:8.3f} t={time.time() - t0:6.1f}s')
        train_time = time.time() - t0
        log(f'[train] done {cli.epochs} epochs in {train_time:.1f}s')
        tinfo = {'loss_history': loss_hist,
                 'mu_norm_history': mu_norm_hist,
                 'train_seconds': train_time, 'total_nodes': total_nodes,
                 'rho_final': rho_t, 'c_mean_final': c_mean_prev}
        torch.save({'arm': cli.arm, 'spec': spec,
                    'config': vars(cli) | {'ns_rank': ns.rank},
                    'net': net.state_dict(),
                    'mu_param_norm': float(mu_param.norm()),
                    'train_info': tinfo,
                    'command': sys.argv},
                   os.path.join(save_dir, 'alon_upgrade.pt'))
        log(f'[saved] {save_dir}/alon_upgrade.pt')
        net.eval()

    # ---------------------------------------------------------------- eval
    if cli.skip_eval:
        _fh.close()
        return
    ref_per = load_reference(cli.dataset)
    eval_out = {
        'arm': cli.arm, 'dataset': cli.dataset, 'seed': cli.seed,
        'spec': spec, 'config': vars(cli), 'train_info': tinfo,
        'checkpoint': cli.eval_only or
        os.path.join(save_dir, 'alon_upgrade.pt'),
        'lambda_freq': cli.lambda_freq,
        'splits': {},
    }

    for split in cli.eval_splits:
        graphs, _ = build_split_graphs(ns, split=split, limit=0)
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
            log(f'[eval:{split}] {name:<16} rep_mean={r["mean_repaired"]:8.3f} '
                f'med={r["median_repaired"]:7.1f} raw={r["mean_raw"]:8.3f} '
                f'infeas={r["frac_infeasible_raw"]:.3f} '
                f'gap={r.get("gap_vs_ref")}')

        run('statusquo', None, 0.0, statusquo=True)
        for K in (1, 5):
            for rho in (0.02, 0.2):
                run(f'asc_K{K}_r{rho}', K, rho)
        eval_out['splits'][split] = split_out

    suffix = '_valonly' if cli.eval_only else ''
    out_path = f'results/alon_upgrade/eval_{cli.arm}_{cli.dataset}' \
               f'_s{cli.seed}{suffix}.json'
    json.dump(eval_out, open(out_path, 'w'), indent=1)
    log(f'[done] {out_path}')
    _fh.close()


if __name__ == '__main__':
    main()
