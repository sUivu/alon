#!/usr/bin/env python3
"""Workstream A: train ONE dual-hyperparameter sweep config FROM SCRATCH.

Extends model/alon_upgrade.py + train_alon_upgrade.py READ-ONLY (imports
only; no file outside scripts/dual_tuning/, results/dual_tuning/ and
ckpts/dual_tuning_* is touched).  The training loop is a verbatim copy of the
upgrade campaign loop with these extensions:
  * lambda_freq / rho_init / tau / rho_max configurable per config
  * mu stabilization: elementwise cap (mu <- min(mu, cap)) or proximal
    damping (mu += rho_t*c - eta_d*(mu - mu_prev))
  * constraint signal c: signed phi (default) | relu(phi) | product g_ij
    (product switches the primal residual too, = D4 representation)
  * dual warm-up: no dual updates before epoch `warmup`
  * norm: pernode (D1-style, magnitude-blind) | graph (D2-style)

Eval (identical to campaign primary metric): test split, mu=0 statusquo,
rounding seed 0 / 1000 hyperplanes -> greedy repair (feasibility asserted);
report mean repaired size + infeasible-before-repair rate.

Usage:
  CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=3 \
  python scripts/dual_tuning/train_sweep.py \
      --config T3 --dataset PROTEINS --seed 0
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
from torch_scatter import scatter_add

HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(HERE))

from configs import get_config  # noqa: E402

from data.loader import construct_dataset, prepare_alon_dataset  # noqa: E402
from model.alon_a_fair import (  # noqa: E402
    FairGradLayer, make_batch_graphs, edge_residual, node_violation,
    round_repair_batch, EPS, _PHI, _PHI2,
)
from model.alon_upgrade import UpgradeALONGNN  # noqa: E402
from model.alon_a import build_split_graphs  # noqa: E402
from train_alon_upgrade import (  # noqa: E402
    DATASET_SPECS, build_ns, seed_all, make_batches, load_reference,
)

from problem.problems import get_problem  # noqa: E402


# ----------------------------------------------------------------------------
# Tunable grad layer: FairGradLayer with a configurable constraint signal
# ----------------------------------------------------------------------------
class TunableGradLayer(FairGradLayer):
    """g_obj from the primal residual (as fair harness); g_dual / c from
    c_signal in {'signed','relu','product'}."""

    def __init__(self, problem, residual='signed', c_signal='signed',
                 create_graph=True):
        super().__init__(problem, residual=residual, create_graph=create_graph)
        self.c_signal = c_signal

    def c_of(self, x, batch):
        if self.c_signal == 'signed':
            return node_violation(x, batch, self.residual)
        if self.c_signal == 'relu':
            phi = edge_residual(x, batch, self.residual)
            ei = batch.edge_index
            if phi.numel() == 0:
                return x.new_zeros(x.size(0))
            return scatter_add(torch.relu(phi), ei[0], dim=0,
                               dim_size=x.size(0))
        if self.c_signal == 'product':
            assert self.residual == 'product', \
                'c_signal=product requires primal residual=product'
            return node_violation(x, batch, 'product')
        raise ValueError(self.c_signal)

    def forward(self, x, batch, mu=None):
        with torch.enable_grad():
            if not x.requires_grad:
                x.requires_grad_(True)
            phi = edge_residual(x, batch, self.residual)
            pen = 0.5 * (phi ** 2).sum() if phi.numel() > 0 else x.sum() * 0.0
            # self.rho stored un-halved; product residuals unchanged -> 4/PHI2.
            loss_obj = self.problem.objective(x, batch) + \
                (self.rho * 4.0 / _PHI2[self.residual] / 2.0) * pen
            g_obj = torch.autograd.grad(loss_obj, x,
                                        create_graph=self.create_graph,
                                        retain_graph=True)[0]
            c = self.c_of(x, batch)
            if mu is None:
                g_dual = torch.zeros_like(x)
            else:
                loss_dual = (mu * c).sum() / _PHI[self.residual]
                g_dual = torch.autograd.grad(loss_dual, x,
                                             create_graph=self.create_graph)[0]
            return g_obj, g_dual, c


def build_tuned_network(cfg, problem, rank, num_layers, lift_ratio,
                        create_graph=True):
    grad = TunableGradLayer(problem, residual=cfg['residual'],
                            c_signal=cfg['c_signal'],
                            create_graph=create_graph)
    return UpgradeALONGNN(grad, in_channels=rank, num_layers=num_layers,
                         lift_ratio=lift_ratio, norm=cfg['norm'])


def c_violation(x, batch, cfg):
    """Standalone c_signal violation (no grad layer needed)."""
    if cfg['c_signal'] == 'signed':
        return node_violation(x, batch, cfg['residual'])
    if cfg['c_signal'] == 'relu':
        phi = edge_residual(x, batch, cfg['residual'])
        if phi.numel() == 0:
            return x.new_zeros(x.size(0))
        return scatter_add(torch.relu(phi), batch.edge_index[0], dim=0,
                           dim_size=x.size(0))
    if cfg['c_signal'] == 'product':
        return node_violation(x, batch, 'product')
    raise ValueError(cfg['c_signal'])


def al_loss_full(x_out, batch, mu_batch, problem, cfg, penalty):
    phi = edge_residual(x_out, batch, cfg['residual'])
    pen = 0.5 * (phi ** 2).sum() if phi.numel() > 0 else x_out.sum() * 0.0
    # penalty stored un-halved; product residuals unchanged -> 4/PHI2.
    loss = problem.objective(x_out, batch) + \
        penalty * 4.0 / _PHI2[cfg['residual']] * pen
    if mu_batch is not None:
        c = c_violation(x_out, batch, cfg)
        loss = loss + (mu_batch * c).sum() / _PHI[cfg['residual']]
    return loss


# ----------------------------------------------------------------------------
# Tunable dual step (extension of model.alon_upgrade.dual_step alg1 branch)
# ----------------------------------------------------------------------------
def tuned_dual_step(mu_param, add_mu, n_batches, cfg, rho_t, c_mean_prev,
                    mu_prev):
    # add_mu is un-halved; /PHI restores the historical c_mean value, and
    # 2/PHI restores the historical multiplier-step scale for product residuals.
    c_mean = float(add_mu.abs().sum()) / max(n_batches, 1) / _PHI[cfg['residual']]
    if c_mean_prev is not None and c_mean > cfg['delta'] * c_mean_prev:
        rho_t = min(cfg['tau'] * rho_t, cfg['rho_max'])
    with torch.no_grad():
        step = rho_t * (2.0 / _PHI[cfg['residual']]) * add_mu / max(n_batches, 1)
        if cfg['mu_mode'] == 'prox':
            new = mu_param + step - cfg['eta_d'] * (mu_param - mu_prev)
        else:
            new = mu_param + step
        new.clamp_(min=0.0)
        if cfg['mu_mode'] == 'cap':
            new.clamp_(max=cfg['mu_cap'])
        mu_prev.copy_(mu_param)
        mu_param.copy_(new)
    return mu_param, rho_t, c_mean


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', required=True)
    ap.add_argument('--dataset', default='PROTEINS',
                    choices=list(DATASET_SPECS))
    ap.add_argument('--epochs', type=int, default=200)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--lr', type=float, default=1e-3)
    ap.add_argument('--grad_clip', type=float, default=5.0)
    ap.add_argument('--penalty_s', type=float, default=-0.125)
    ap.add_argument('--penalty_e', type=float, default=0.625)
    ap.add_argument('--num_layers', type=int, default=16)
    ap.add_argument('--lift_ratio', type=float, default=0.1)
    ap.add_argument('--skip_train', action='store_true',
                    help='eval only from existing checkpoint')
    args = ap.parse_args()

    cfg = get_config(args.config)
    tag = f'dual_tuning_{args.config}_{args.dataset}_s{args.seed}'
    save_dir = os.path.join('ckpts', 'dual_tuning_sweep', tag)
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs('results/dual_tuning/sweep', exist_ok=True)
    out_path = (f'results/dual_tuning/sweep/'
                f'{args.config}_{args.dataset}_s{args.seed}.json')
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
    log(f'[data] n={n} train={tr} val={va} test={n - tr - va}')

    problem = get_problem(ns)

    # -------------------------------------------------------------- train
    mu_param = torch.zeros(total_nodes, device=device)
    mu_prev = torch.zeros_like(mu_param)
    tinfo = {}
    if args.skip_train:
        ck = torch.load(os.path.join(save_dir, 'alon_upgrade.pt'),
                        map_location='cpu', weights_only=False)
        net = build_tuned_network(cfg, problem, rank=ns.rank,
                                  num_layers=ns.num_layers,
                                  lift_ratio=ns.lift_ratio,
                                  create_graph=False)
        net.load_state_dict(ck['net'])
        net = net.to(device)
        tinfo = ck['train_info']
        log(f'[skip_train] loaded {save_dir}/alon_upgrade.pt')
    else:
        net = build_tuned_network(cfg, problem, rank=ns.rank,
                                  num_layers=ns.num_layers,
                                  lift_ratio=ns.lift_ratio,
                                  create_graph=True).to(device)
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
                    (ep + 1) % cfg['lambda_freq'] == 0 and ep >= cfg['warmup']:
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
        torch.save({'config': args.config, 'cfg': cfg,
                    'dataset': args.dataset, 'seed': args.seed,
                    'command': sys.argv,
                    'net': net.state_dict(),
                    'train_info': tinfo},
                   os.path.join(save_dir, 'alon_upgrade.pt'))
        log(f'[saved] {save_dir}/alon_upgrade.pt')
        net.eval()

    # ---------------------------------------------------------------- eval
    # Campaign primary metric: statusquo (mu = 0) rounding + greedy repair.
    ref_per = load_reference(args.dataset)
    graphs, _ = build_split_graphs(ns, split='test', limit=0)
    x_in, big, offsets = make_batch_graphs(graphs, ns.rank, device, seed=0)
    with torch.no_grad():
        zero = torch.zeros(big.num_nodes, device=device)
        x_out = net(x_in, big, zero)
    rows = round_repair_batch(x_out, graphs, offsets, seed=0,
                              n_hyperplanes=1000)
    sizes = np.array([r['repaired_size'] for r in rows], dtype=float)
    raw = np.array([r['raw_size'] for r in rows], dtype=float)
    out = {
        'config': args.config, 'cfg': cfg, 'dataset': args.dataset,
        'seed': args.seed, 'n_test_graphs': len(graphs),
        'checkpoint': os.path.join(save_dir, 'alon_upgrade.pt'),
        'train_info': {k: v for k, v in tinfo.items()
                       if not k.endswith('history')},
        'mean_repaired': float(sizes.mean()),
        'median_repaired': float(np.median(sizes)),
        'std_repaired': float(sizes.std()),
        'mean_raw': float(raw.mean()),
        'frac_infeasible_raw': float(np.mean(
            [r['infeasible_raw'] for r in rows])),
        'mean_raw_uncovered': float(np.mean(
            [r['raw_uncovered'] for r in rows])),
        'per_graph': rows,
    }
    if ref_per:
        vals = [ref_per.get(int(i)) for _, _, i in graphs
                if ref_per.get(int(i)) is not None]
        if vals:
            rv = float(np.mean(vals))
            out['gap_vs_ref'] = (out['mean_repaired'] - rv) / rv
    json.dump(out, open(out_path, 'w'), indent=1)
    log(f'[done] rep={out["mean_repaired"]:.3f} infeas='
        f'{out["frac_infeasible_raw"]:.3f} -> {out_path}')
    _fh.close()


if __name__ == '__main__':
    main()
