#!/usr/bin/env python3
"""Evaluate completed retraining-campaign checkpoints under the feasibility-
corrected protocol and write per-cell results to
results/retrain_campaign/repaired_cells.json (merge-only).

REUSES scripts/reeval_main_feasible.py (imported, NOT modified):
  - paper split train_fraction=0.8, split_seed=0, test = last 10%
  - randomized-hyperplane rounding, n_hyperplanes=1000, seed 0
  - greedy repair; violation==0 asserted per graph
  - exact optima from results/classical_baselines/<ds>.json (akiba_iwata)
  - ALM/ALON cells evaluated with mu = 0 (paper protocol)
OptGNN (model_type LiftMP, trained by train_baseline.py) uses the same protocol
with the fused LiftMP baseline in model/baselines.py (mirrors
run_supplementary_ood in reeval_main_feasible.py).

Only writes under results/retrain_campaign/ - results/reeval_main/ untouched.
Run: CUDA_VISIBLE_DEVICES=2 python \
       scripts/retrain_campaign/eval_repaired_cells.py [--job JOB_TAG ...]
"""
import argparse
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

# import reeval_main_feasible without making scripts/ a package
_spec = importlib.util.spec_from_file_location(
    'reeval_main_feasible', ROOT / 'scripts' / 'reeval_main_feasible.py')
R = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(R)

OUT_PATH = ROOT / 'results' / 'retrain_campaign' / 'repaired_cells.json'

# row tag -> classical_baselines stem (row tags use ER_[100_200] style)
ROW_STEM = {}
for short in ['ER', 'BA', 'WS', 'HK']:
    for scale in ['50_100', '100_200', '400_500']:
        ROW_STEM[f'{short}_[{scale}]'] = f'{short}_{scale}'
ROW_STEM.update({d: d for d in
                 ['MUTAG', 'ENZYMES', 'PROTEINS', 'IMDB-BINARY', 'COLLAB',
                  'RB200', 'RB500']})

from torch_geometric.loader import DataLoader  # noqa: E402
from data.loader import construct_dataset, prepare_alon_dataset  # noqa: E402
from model.training import featurize_batch  # noqa: E402
from model.baselines import build_liftmp_baseline  # noqa: E402
from problem.baselines import random_hyperplane_projector  # noqa: E402
from problem.problems import get_problem  # noqa: E402


def eval_optgnn_checkpoint(ckpt_path, params, device):
    """LiftMP/OptGNN eval under the identical protocol (paper split, rounding
    seed 0 / 1000 hyperplanes, greedy repair), using the fused LiftMP baseline
    in model/baselines.py."""
    args = R.make_args(params, device)
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)

    dataset = construct_dataset(args)
    dataset, _ = prepare_alon_dataset(dataset)
    n = len(dataset)
    train_size = int(args.train_fraction * n)
    val_size = (n - train_size) // 2
    test_size = n - train_size - val_size
    gen = torch.Generator().manual_seed(args.split_seed)
    _, _, test_subset = torch.utils.data.random_split(
        dataset, [train_size, val_size, test_size], generator=gen)
    test_indices = sorted(test_subset.indices)

    test_loader = DataLoader(test_subset, batch_size=args.batch_size, shuffle=False)
    problem = get_problem(args)  # score / constraint (identical metric values)
    model = build_liftmp_baseline(args)
    sd = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    if isinstance(sd, dict) and 'state_dict' in sd:
        sd = sd['state_dict']
    model.load_state_dict(sd, strict=True)
    model.to(device).eval()

    per_graph = []
    pos = 0
    with torch.no_grad():
        for batch in test_loader:
            batch.penalty = args.penalty
            x_in, batch = featurize_batch(args, batch)
            x_out = model(x_in, batch)
            x_proj = random_hyperplane_projector(args, x_out, batch, problem.score,
                                                 n_hyperplanes=1000)
            x_proj = torch.where(x_proj == 0, torch.ones_like(x_proj), x_proj)
            ptr = batch.ptr.cpu().numpy()
            data_list = batch.to_data_list()
            xp = x_proj.cpu().numpy()
            for gi, dg in enumerate(data_list):
                xv = xp[ptr[gi]:ptr[gi + 1]]
                cover = xv > 0
                eu, ev = R.undirected_edges(dg.edge_index)
                raw_size = int(cover.sum())
                n_unc = int((~(cover[eu] | cover[ev])).sum())
                repaired = R.greedy_repair(cover, eu, ev)
                rep_size = int(repaired.sum())
                assert int((~(repaired[eu] | repaired[ev])).sum()) == 0, 'repair failed'
                per_graph.append({'dataset_idx': int(test_subset.indices[pos]),
                                  'raw_size': raw_size, 'uncovered_edges': n_unc,
                                  'feasible_before_repair': n_unc == 0,
                                  'repaired_size': rep_size})
                pos += 1

    agg = {
        'n_test_graphs': len(per_graph),
        'raw_size_mean': float(np.mean([g['raw_size'] for g in per_graph])),
        'infeasible_pct': 100.0 * float(np.mean([not g['feasible_before_repair'] for g in per_graph])),
        'mean_uncovered_edges': float(np.mean([g['uncovered_edges'] for g in per_graph])),
        'repaired_size_mean': float(np.mean([g['repaired_size'] for g in per_graph])),
        'violation_after_repair': 0,
        'per_graph': per_graph,
    }
    return agg, test_indices


def evaluate_job(job, device):
    run_base = ROOT / 'training_runs' / 'retrain' / job
    dirs = sorted(run_base.glob('paramhash*'))  # ':' (train.py) or '_' (train_dp/baseline)
    if not dirs:
        return None
    rd = dirs[0]
    if not (rd / 'final_model.pt').exists():
        return None
    params = json.load(open(rd / 'params.txt'))
    COLS = ['gin_raw', 'gat_raw', 'gcnn_raw', 'gated_raw', 'gated_pd',
            'alon', 'optgnn']
    col = next(c for c in COLS if job.endswith('_' + c))
    row = job[: -(len(col) + 1)]
    stem = ROW_STEM.get(row)
    if stem is None:
        raise ValueError(f'unknown row for job {job}')
    classical = R.load_classical(stem)

    if params['model_type'] == 'LiftMP':
        agg, test_idx = eval_optgnn_checkpoint(str(rd / 'final_model.pt'), params, device)
    else:
        agg, test_idx = R.eval_checkpoint(str(rd / 'final_model.pt'), params, device)

    split_match = classical is not None and sorted(classical['test_indices']) == test_idx
    opts = [classical['opt_by_idx'].get(i) for i in test_idx] if classical else [None] * len(test_idx)
    n_opt = sum(o is not None for o in opts)
    gap = (float(np.mean([g['repaired_size'] - o for g, o in
                          zip(agg['per_graph'], opts) if o is not None]))
           if n_opt else None)
    opt_mean = (float(np.mean([o for o in opts if o is not None])) if n_opt else None)

    rec = {
        'status': 'OK',
        'checkpoint': str((rd / 'final_model.pt').relative_to(ROOT)),
        'model_type': params['model_type'],
        'retrained': '2026-09 retraining campaign',
        'mu_at_eval': 0 if params['model_type'] == 'ALONGNN' else 'n/a (no dual variable)',
        'split_matches_classical': split_match,
        'n_with_exact_optimum': n_opt,
        'exact_optimum_mean': opt_mean,
        'gap_repaired_vs_exact': gap,
        **{k: v for k, v in agg.items() if k != 'per_graph'},
    }
    return (row, col), rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--job', nargs='*', default=None,
                    help='job tags to (re)evaluate; default = all completed')
    ap.add_argument('--device', default='cuda')
    cli = ap.parse_args()
    device = torch.device(cli.device)

    results = {}
    if OUT_PATH.exists():
        results = json.load(open(OUT_PATH))

    base = ROOT / 'training_runs' / 'retrain'
    jobs = cli.job if cli.job else sorted(
        p.parent.parent.relative_to(base).as_posix()
        for p in base.glob('*/paramhash*/done.txt')) if base.exists() else []

    for job in jobs:
        try:
            out = evaluate_job(job, device)
        except Exception as e:  # record failure honestly, keep going
            print(f'[eval] {job} FAILED: {type(e).__name__}: {e}')
            continue
        if out is None:
            print(f'[eval] {job}: no final_model.pt yet, skipped')
            continue
        key, rec = out
        results['|'.join(key)] = rec
        json.dump(results, open(OUT_PATH, 'w'), indent=2)
        print(f"[eval] {job}: repaired={rec['repaired_size_mean']:.2f} "
              f"raw={rec['raw_size_mean']:.2f} infeas={rec['infeasible_pct']:.1f}% "
              f"gap={rec['gap_repaired_vs_exact']} split_ok={rec['split_matches_classical']}")
    print(f'wrote {OUT_PATH} ({len(results)} cells)')


if __name__ == '__main__':
    main()
