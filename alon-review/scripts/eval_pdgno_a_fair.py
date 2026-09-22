#!/usr/bin/env python3
"""
PDGNO-A FAIR bake-off -- evaluation.  Identical protocol for every candidate.

Per (split, config) it runs the trained representation and reports the REPAIRED
cover from randomized-hyperplane rounding (seed 0) + greedy feasibility repair,
asserts feasibility of every reported cover (violation == 0), and records the
infeasible-before-repair rate.  Reference optimum = the ``akiba_iwata`` field in
results/classical_baselines/<name>.json.

Configs
-------
  statusquo     : trained theta, mu = 0
  asc_K{r}      : mu = 0 then K ascent steps, fixed/adaptive rho
  amort_K{r}    : amortized mu_phi init then K ascent steps  (only if ckpt has mu_phi)

The rho sweep is evaluated on the validation split; aggregation selects the best
(K, rho) on val and reports the test number -> no test leakage.

Usage
-----
  CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=3 \
  /app/pdgno-env/bin/python scripts/eval_pdgno_a_fair.py \
      --checkpoint ckpts/pdgno_a_fair_C1_PROTEINS_seed0/pdgno_a_fair.pt \
      --dataset PROTEINS --splits val test \
      --ks 0 1 5 20 --rho_sweep 0.05 0.2 1.0 --adaptive
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

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from problem.problems import get_problem  # noqa: E402
from model.pdgno_a import build_split_graphs  # noqa: E402
from model.pdgno_a_fair import (  # noqa: E402
    spec_of, build_network, PDGNOA_Fair, make_batch_graphs, eval_config,
    summarize, CANDIDATE_ORDER,
)

REF_NAME = {'ErdosRenyi': 'ER_100_200', 'BarabasiAlbert': 'BA_100_200',
            'PROTEINS': 'PROTEINS', 'ENZYMES': 'ENZYMES',
            'PowerlawCluster': 'HK_100_200', 'WattsStrogatz': 'WS_100_200'}
DATASET_GEN = {
    'PROTEINS': dict(num_graphs=2000, gen_n=None, gen_m=None, gen_k=None, gen_p=None),
    'ENZYMES': dict(num_graphs=2000, gen_n=None, gen_m=None, gen_k=None, gen_p=None),
    'ErdosRenyi': dict(num_graphs=2000, gen_n=[100, 200], gen_m=None, gen_k=None, gen_p=[0.15]),
    'BarabasiAlbert': dict(num_graphs=2000, gen_n=[100, 200], gen_m=[4], gen_k=None, gen_p=None),
}


def make_ns(dataset, rank, num_layers, lift_ratio, hidden_channels=32):
    spec = DATASET_GEN[dataset]
    return argparse.Namespace(
        problem_type='vertex_cover', dataset=dataset,
        num_graphs=spec['num_graphs'], gen_n=spec['gen_n'],
        gen_m=spec['gen_m'], gen_k=spec['gen_k'], gen_p=spec['gen_p'],
        data_seed=0, parallel=0, infinite=False, positional_encoding=None,
        pe_dimension=8, train_fraction=0.8, split_seed=0, rank=rank,
        num_layers=num_layers, lift_ratio=lift_ratio,
        hidden_channels=hidden_channels)


def load_reference(dataset, graphs):
    ref_name = REF_NAME.get(dataset, dataset)
    p = PROJECT_ROOT / 'results' / 'classical_baselines' / f'{ref_name}.json'
    if not p.exists():
        return None, {}
    d = json.load(open(p))
    ai = d.get('summary', {}).get('akiba_iwata', {})
    per = {}
    for g in d.get('per_graph', []):
        v = g.get('akiba_iwata')
        if isinstance(v, dict):
            v = v.get('size')
        if v is not None:
            per[int(g['idx'])] = v
    vals = [per[i] for _, _, i in graphs if per.get(int(i)) is not None]
    ref_eval = float(np.mean(vals)) if vals else None
    meta = {
        'ref_name': ref_name, 'avg_size_full': ai.get('avg_size'),
        'avg_size_common': ai.get('avg_size_common'),
        'coverage': ai.get('coverage'), 'n_eval_graphs': len(graphs),
        'n_exact_on_eval': len(vals),
        'coverage_on_eval': f'{len(vals)}/{len(graphs)}',
        'ref_eval_mean_exact': ref_eval,
    }
    return meta, per


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--checkpoint', required=True)
    ap.add_argument('--dataset', required=True,
                    help='training dataset defining the split')
    ap.add_argument('--eval_dataset', default=None,
                    help='override for OOD (e.g. BarabasiAlbert)')
    ap.add_argument('--splits', nargs='+', default=['test'],
                    choices=['val', 'test'])
    ap.add_argument('--ks', nargs='+', type=int, default=[0, 1, 5, 20])
    ap.add_argument('--rho_sweep', nargs='+', type=float,
                    default=[0.005, 0.02, 0.1, 0.5])
    ap.add_argument('--rho', type=float, default=0.2,
                    help='rho used when --rho_sweep is empty')
    ap.add_argument('--adaptive', action='store_true', default=None)
    ap.add_argument('--no_adaptive', dest='adaptive', action='store_false')
    ap.add_argument('--delta', type=float, default=0.25)
    ap.add_argument('--tau', type=float, default=2.0)
    ap.add_argument('--rho_max', type=float, default=50.0)
    ap.add_argument('--n_hyperplanes', type=int, default=1000)
    ap.add_argument('--limit', type=int, default=0)
    ap.add_argument('--out', required=True)
    cli = ap.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ck = torch.load(cli.checkpoint, map_location='cpu', weights_only=False)
    cfg = ck['config']
    cand = cfg['candidate']
    spec = spec_of(cand)
    adaptive = cfg.get('adaptive', True) if cli.adaptive is None else cli.adaptive
    ev_ds = cli.eval_dataset or cli.dataset
    ns = make_ns(ev_ds, cfg['rank'], cfg['num_layers'], cfg['lift_ratio'])
    problem = get_problem(ns)

    net = build_network(cand, problem, rank=cfg['rank'],
                        num_layers=cfg['num_layers'],
                        lift_ratio=cfg['lift_ratio'], create_graph=False)
    net.load_state_dict(ck['net'])
    net = net.to(device)
    net.eval()
    pdgno = PDGNOA_Fair(net, problem, rank=cfg['rank'],
                        hidden=cfg.get('mu_hidden', 32),
                        num_dual_layers=cfg.get('mu_layers', 2)).to(device)
    has_mu_phi = 'mu_phi' in ck
    if has_mu_phi:
        pdgno.mu_phi.load_state_dict(ck['mu_phi'])
    pdgno.eval()

    out = {
        'checkpoint': cli.checkpoint, 'candidate': cand, 'spec': spec,
        'config': cfg, 'eval_dataset': ev_ds, 'ran_adaptive': bool(adaptive),
        'has_mu_phi': has_mu_phi,
        'n_hyperplanes': cli.n_hyperplanes,
        'env': {'torch': torch.__version__, 'python': sys.version.split()[0]},
        'command': sys.argv,
        'splits': {},
    }
    rhos = cli.rho_sweep if cli.rho_sweep else [cli.rho]

    for split in cli.splits:
        graphs, _ = build_split_graphs(ns, split=split, limit=cli.limit)
        x_in, big, offsets = make_batch_graphs(graphs, cfg['rank'], device,
                                               seed=0)
        ref_meta, _ = load_reference(ev_ds, graphs)
        split_out = {
            'n_graphs': len(graphs),
            'avg_n': float(np.mean([g[0] for g in graphs])) if graphs else 0.0,
            'reference': ref_meta, 'runs': [],
        }

        def run(name, K, rho, use_mu_phi, statusquo=False):
            if device.type == 'cuda':
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            rows, mh, rh = eval_config(
                pdgno, x_in, big, graphs, offsets, K, rho,
                use_mu_phi=use_mu_phi, adaptive=adaptive, delta=cli.delta,
                tau=cli.tau, rho_max=cli.rho_max,
                n_hyperplanes=cli.n_hyperplanes, statusquo=statusquo)
            if device.type == 'cuda':
                torch.cuda.synchronize()
            dt = time.perf_counter() - t0
            r = summarize(name, rows, K, rho, mu_norm_traj=mh, resid_traj=rh,
                          extra={'use_mu_phi': bool(use_mu_phi),
                                 'mean_time_s_per_graph': dt / max(len(graphs), 1),
                                 'wall_s': dt})
            if ref_meta and ref_meta['ref_eval_mean_exact'] is not None:
                rv = ref_meta['ref_eval_mean_exact']
                r['gap_vs_ref'] = (r['mean_repaired'] - rv) / rv
                r['delta_vs_ref'] = r['mean_repaired'] - rv
            print(f'  [{split}] {name:<22} rep_mean={r["mean_repaired"]:8.3f} '
                  f'med={r["median_repaired"]:7.1f} infeas={r["frac_infeasible_raw"]:.3f} '
                  f'|mu|={r["mu_norm_traj"][-1]:9.3f} t/g={r["mean_time_s_per_graph"]:.4f}s',
                  flush=True)
            return r

        split_out['runs'].append(run('statusquo', None, 0.0, False, statusquo=True))
        for K in cli.ks:
            for rho in rhos:
                split_out['runs'].append(run(f'asc_K{K}_r{rho}', K, rho, False))
        if has_mu_phi:
            for K in cli.ks:
                for rho in rhos:
                    split_out['runs'].append(
                        run(f'amort_K{K}_r{rho}', K, rho, True))
        out['splits'][split] = split_out

    Path(cli.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(cli.out, 'w'), indent=1)
    print(f'saved -> {cli.out}')


if __name__ == '__main__':
    main()
