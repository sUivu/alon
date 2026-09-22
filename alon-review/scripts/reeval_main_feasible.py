#!/usr/bin/env python3
"""
Feasibility-corrected re-evaluation of the paper's MAIN TABLE (tab:main_results).

Background: the shared evaluation path scores -(cover_size + penalty) and takes
min|score| with NO feasibility repair (problem/losses.py:156-162,
problem/baselines.py:139-200, model/training.py:77-88).  Reported OBJ values can
be infeasible and even below the true optimum.  This script re-evaluates every
checkpoint that could be located for the main table under a single, uniform
protocol:

  1. Same test split as the paper (train_fraction=0.8, split_seed=0,
     test = last 10%; reproduced exactly via data.loader's random_split and
     cross-checked against results/classical_baselines/<ds>.json test_indices).
  2. Randomized-hyperplane rounding (seed 0, n_hyperplanes=1000, identical to
     the original eval) -> GREEDY REPAIR of uncovered edges -> REPAIRED
     feasible cover size.
  3. Per cell we record: paper's reported value, raw (un-repaired) size,
     infeasible-before-repair fraction, mean uncovered edges, paper-protocol
     OBJ (size + constraint, exactly the quantity the paper's eval minimized),
     repaired size, violation-after-repair (must be 0), and gap to the exact
     Akiba-Iwata optimum from results/classical_baselines/<ds>.json.
  4. ALM (+PD) cells are evaluated with mu = 0 (a zero dual vector), matching
     the paper's inference protocol for these models.

Honesty: cells whose checkpoint cannot be found are listed as MISSING and are
NOT substituted with log scores.

Outputs (all programmatic):
  results/reeval_main/main_table_repaired.json
  results/reeval_main/main_table_repaired.md
  results/reeval_main/ood_tu_repaired.json   (supplementary, --with-ood)

Run:  bash scripts/reeval_main_feasible.sh
"""

import argparse
import json
import os
import re
import sys
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from torch_geometric.loader import DataLoader

from data.loader import construct_dataset, prepare_alon_dataset
from model.models import construct_model
from model.training import featurize_batch
from problem.baselines import random_hyperplane_projector
from problem.problems import get_problem

TEX = PROJECT_ROOT / 'neurips_2026 (1).tex'
CLASSICAL_DIR = PROJECT_ROOT / 'results' / 'classical_baselines'
OUT_DIR = PROJECT_ROOT / 'results' / 'reeval_main'

# ---------------------------------------------------------------------------
# Cell -> checkpoint map (main table).  Only cells whose checkpoint EXISTS on
# this machine are listed here.  Provenance: training_runs/<ts>/final_model.pt,
# identity verified via params.txt (model_type + dataset + scale).
# ---------------------------------------------------------------------------
ROW_PARAMS = {  # tex row -> (classical_baselines json stem)
    'ER [50,100]': 'ER_50_100', 'BA [50,100]': 'BA_50_100',
    'WS [50,100]': 'WS_50_100', 'HK [50,100]': 'HK_50_100',
    'ER [100,200]': 'ER_100_200', 'BA [100,200]': 'BA_100_200',
    'WS [100,200]': 'WS_100_200', 'HK [100,200]': 'HK_100_200',
    'ER [400,500]': 'ER_400_500', 'BA [400,500]': 'BA_400_500',
    'WS [400,500]': 'WS_400_500', 'HK [400,500]': 'HK_400_500',
    'MUTAG': 'MUTAG', 'ENZYMES': 'ENZYMES', 'PROTEINS': 'PROTEINS',
    'IMDB-BIN': 'IMDB-BINARY', 'COLLAB': 'COLLAB',
    'RB200': 'RB200', 'RB500': 'RB500',
}

CELL_CKPTS = {
    # (tex_row, method_column): training_runs timestamp dir with final_model.pt
    ('ER [100,200]', 'gcnn_pd'): '2026-05-05_17:17:38',   # ALONGCNN ErdosRenyi 100_200
    ('BA [100,200]', 'gcnn_pd'): '2026-05-05_17:18:43',   # ALONGCNN BarabasiAlbert 100_200
    ('WS [100,200]', 'gcnn_pd'): '2026-05-05_18:27:36',   # ALONGCNN WattsStrogatz 100_200
    ('HK [100,200]', 'gcnn_pd'): '2026-05-05_18:54:40',   # ALONGCNN PowerlawCluster 100_200
    ('ER [400,500]', 'gcnn_pd'): '2026-05-05_19:02:21',   # ALONGCNN ErdosRenyi 400_500
    ('BA [400,500]', 'gcnn_pd'): '2026-05-05_19:03:50',   # ALONGCNN BarabasiAlbert 400_500
    ('WS [400,500]', 'gcnn_pd'): '2026-05-05_19:55:20',   # ALONGCNN WattsStrogatz 400_500
    ('HK [400,500]', 'gcnn_pd'): '2026-05-05_20:13:10',   # ALONGCNN PowerlawCluster 400_500
    ('ER [100,200]', 'gated_pd'): '2026-05-05_20:31:06',  # ALONGatedGCNN ErdosRenyi 100_200
    ('BA [100,200]', 'gated_pd'): '2026-05-05_21:17:47',  # ALONGatedGCNN BarabasiAlbert 100_200
    ('WS [100,200]', 'gated_pd'): '2026-05-05_21:31:52',  # ALONGatedGCNN WattsStrogatz 100_200
    ('HK [100,200]', 'gated_pd'): '2026-05-05_21:37:26',  # ALONGatedGCNN PowerlawCluster 100_200
    ('ER [400,500]', 'gated_pd'): '2026-05-05_21:59:54',  # ALONGatedGCNN ErdosRenyi 400_500
    ('BA [400,500]', 'gated_pd'): '2026-05-05_22:50:21',  # ALONGatedGCNN BarabasiAlbert 400_500
    ('WS [400,500]', 'gated_pd'): '2026-05-05_23:02:17',  # ALONGatedGCNN WattsStrogatz 400_500
    ('HK [400,500]', 'gated_pd'): '2026-05-05_23:09:22',  # ALONGatedGCNN PowerlawCluster 400_500
    ('HK [400,500]', 'gat_pd'): '2026-05-05_17:05:41',    # ALONGAT PowerlawCluster 400_500
}

METHOD_COLS = ['gin_raw', 'gin_pd', 'gat_raw', 'gat_pd', 'gcnn_raw', 'gcnn_pd',
               'gated_raw', 'gated_pd', 'optgnn', 'alon']

OOD_CKPTS = {  # supplementary: OOD-table checkpoints (trained on synthetic [100,200])
    'ALONGNN': {
        'ErdosRenyi': 'training_runs_ood/ALONGNN_ErdosRenyi/final_model.pt',
        'BarabasiAlbert': 'training_runs_ood/ALONGNN_BarabasiAlbert/final_model.pt',
        'WattsStrogatz': 'training_runs_ood/ALONGNN_WattsStrogatz/final_model.pt',
        'PowerlawCluster': 'training_runs_ood/ALONGNN_PowerlawCluster/final_model.pt',
    },
    'OptGNN': {
        'ErdosRenyi': 'training_runs_ood/OptGNN_ErdosRenyi/final_model.pt',
        'BarabasiAlbert': 'training_runs_ood/OptGNN_BarabasiAlbert/final_model.pt',
        'WattsStrogatz': 'training_runs_ood/OptGNN_WattsStrogatz/final_model.pt',
        'PowerlawCluster': 'training_runs_ood/OptGNN_PowerlawCluster/final_model.pt',
    },
}
TU_DATASETS = ['MUTAG', 'ENZYMES', 'PROTEINS', 'IMDB-BINARY', 'COLLAB']


# ---------------------------------------------------------------------------
# Paper table parsing (values come programmatically from the .tex)
# ---------------------------------------------------------------------------
def parse_paper_table():
    if not TEX.exists():
        # The paper .tex is not part of this package; skip the optional
        # cross-check against the printed table.
        return {}
    text = TEX.read_text()
    # isolate tab:main_results tabular body
    start = text.index(r'\label{tab:main_results}')
    end = text.index(r'\end{tabular}', start)
    body = text[start:end]
    rows = {}
    for line in body.splitlines():
        line = line.strip()
        m = re.match(r'^(ER|BA|WS|HK|MUTAG|ENZYMES|PROTEINS|IMDB-BIN|COLLAB|RB200|RB500)'
                     r'((?:\s*\[[\d,\s]+\])?)\s*&(.*)$', line)
        if not m:
            continue
        key = m.group(1) + ((' ' + m.group(2).replace(' ', '')) if m.group(2) else '')
        cells = []
        for cell in m.group(3).split('&'):
            cell = cell.replace(r'\textbf', '').replace('{', '').replace('}', '')
            cell = cell.replace(' ', '').rstrip('\\').strip()
            try:
                cells.append(float(cell))
            except ValueError:
                cells.append(None)
        if len(cells) < 11:
            continue
        rows[key] = {
            'optimal_gurobi10s': cells[0],
            'gin_raw': cells[1], 'gin_pd': cells[2],
            'gat_raw': cells[3], 'gat_pd': cells[4],
            'gcnn_raw': cells[5], 'gcnn_pd': cells[6],
            'gated_raw': cells[7], 'gated_pd': cells[8],
            'optgnn': cells[9], 'alon': cells[10],
        }
    return rows


def load_classical(stem):
    path = CLASSICAL_DIR / f'{stem}.json'
    if not path.exists():
        return None
    d = json.load(open(path))
    opt_by_idx = {e['idx']: e['akiba_iwata']['size'] for e in d['per_graph']}
    return {'test_indices': d['test_indices'], 'opt_by_idx': opt_by_idx}


# ---------------------------------------------------------------------------
# Greedy repair: add vertices (greedy max uncovered-degree, lowest-index
# tie-break) until every edge is covered.  Deterministic.
# ---------------------------------------------------------------------------
def greedy_repair(cover, eu, ev):
    """cover: bool array (True = in cover). eu/ev: undirected edge arrays."""
    cover = cover.copy()
    unc = ~(cover[eu] | cover[ev])
    while unc.any():
        cnt = np.zeros(cover.shape[0], dtype=np.int64)
        np.add.at(cnt, eu[unc], 1)
        np.add.at(cnt, ev[unc], 1)
        v = int(np.argmax(cnt))
        cover[v] = True
        unc = ~(cover[eu] | cover[ev])
    return cover


# ---------------------------------------------------------------------------
# Evaluation of one checkpoint on one dataset split
# ---------------------------------------------------------------------------
def make_args(params, device):
    ns = argparse.Namespace(**params)
    ns.device = device
    if getattr(ns, 'linear_annealing', False):
        ns.penalty = params.get('penalty_e', 1.0)
    else:
        ns.penalty = params.get('penalty', 1.0)
    return ns


def undirected_edges(edge_index):
    ei = edge_index.cpu().numpy()
    lo = np.minimum(ei[0], ei[1])
    hi = np.maximum(ei[0], ei[1])
    pairs = np.unique(np.stack([lo, hi], axis=1), axis=0)
    return pairs[:, 0].astype(np.int64), pairs[:, 1].astype(np.int64)


def eval_checkpoint(ckpt_path, params, device, batch_cap=None):
    """Returns per-cell aggregate metrics + per-graph records."""
    args = make_args(params, device)
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)

    dataset = construct_dataset(args)
    dataset, total_nodes = prepare_alon_dataset(dataset)
    mu_param = torch.zeros(total_nodes, device=device)  # mu = 0 at eval

    n = len(dataset)
    train_size = int(args.train_fraction * n)
    val_size = (n - train_size) // 2
    test_size = n - train_size - val_size
    gen = torch.Generator().manual_seed(args.split_seed)
    _, _, test_subset = torch.utils.data.random_split(
        dataset, [train_size, val_size, test_size], generator=gen)
    test_indices = sorted(test_subset.indices)

    bs = args.batch_size if batch_cap is None else min(args.batch_size, batch_cap)
    test_loader = DataLoader(test_subset, batch_size=bs, shuffle=False)

    problem = get_problem(args)
    model, _ = construct_model(args, mu_param)
    sd = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    if isinstance(sd, dict) and 'state_dict' in sd:
        sd = sd['state_dict']
    missing, unexpected = model.load_state_dict(sd, strict=True)
    model.to(device)
    model.eval()

    per_graph = []
    pos = 0  # position within test subset (batches are sequential, shuffle=False)
    with torch.no_grad():
        for batch in test_loader:
            batch.penalty = args.penalty
            x_in, batch = featurize_batch(args, batch)
            if args.model_type in ['GIN', 'GAT', 'GCNN', 'GatedGCNN']:
                x_out = model(x_in, batch)
            else:
                x_out = model(x_in, batch, mu_param)
            x_proj = random_hyperplane_projector(args, x_out, batch, problem.score, n_hyperplanes=1000)
            x_proj = torch.where(x_proj == 0, torch.ones_like(x_proj), x_proj)
            ptr = batch.ptr.cpu().numpy()
            data_list = batch.to_data_list()
            xp = x_proj.cpu().numpy()
            for gi, dg in enumerate(data_list):
                xv = xp[ptr[gi]:ptr[gi + 1]]
                xv_t = torch.from_numpy(xv).float().to(x_out.device).unsqueeze(1) if xv.ndim == 1 \
                    else torch.from_numpy(xv).float().to(x_out.device)
                cover = (xv > 0)  # +1 = in cover (vertex_cover_obj counts (1+x)/2)
                eu, ev = undirected_edges(dg.edge_index)
                raw_size = int(cover.sum())
                unc_mask = ~(cover[eu] | cover[ev])
                n_unc = int(unc_mask.sum())
                repaired = greedy_repair(cover, eu, ev)
                rep_size = int(repaired.sum())
                assert int((~(repaired[eu] | repaired[ev])).sum()) == 0, 'repair failed'
                # exact paper-protocol OBJ for this graph: size + constraint
                cons_g = float(problem.constraint(xv_t, dg))
                per_graph.append({
                    'dataset_idx': int(test_subset.indices[pos]),
                    'n': int(dg.num_nodes),
                    'raw_size': raw_size,
                    'uncovered_edges': n_unc,
                    'feasible_before_repair': n_unc == 0,
                    'repaired_size': rep_size,
                    'paper_obj_raw': raw_size + cons_g,
                })
                pos += 1

    agg = {
        'n_test_graphs': len(per_graph),
        'raw_size_mean': float(np.mean([g['raw_size'] for g in per_graph])),
        'paper_obj_raw_mean': float(np.mean([g['paper_obj_raw'] for g in per_graph])),
        'infeasible_pct': 100.0 * float(np.mean([not g['feasible_before_repair'] for g in per_graph])),
        'mean_uncovered_edges': float(np.mean([g['uncovered_edges'] for g in per_graph])),
        'repaired_size_mean': float(np.mean([g['repaired_size'] for g in per_graph])),
        'violation_after_repair': 0,  # asserted per graph in the loop
        'per_graph': per_graph,
    }
    return agg, test_indices


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--with-ood', action='store_true',
                    help='also run supplementary OOD-checkpoint eval on TU datasets')
    ap.add_argument('--device', type=str, default='cuda')
    cli = ap.parse_args()
    device = torch.device(cli.device)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    paper = parse_paper_table()
    print(f'parsed {len(paper)} rows from {TEX.name} tab:main_results')

    # ------- main table re-evaluation -------
    results = {}   # (row, col) -> record
    for row in ROW_PARAMS:
        for col in METHOD_COLS:
            results[(row, col)] = {'status': 'MISSING_CHECKPOINT', 'checkpoint': None}

    for (row, col), ts in CELL_CKPTS.items():
        run_dir = PROJECT_ROOT / 'training_runs' / ts
        ckpt = run_dir / 'final_model.pt'
        params = json.load(open(run_dir / 'params.txt'))
        classical = load_classical(ROW_PARAMS[row])
        print(f'=== {row} {col} <- {ts} ({params["model_type"]})')
        agg, test_idx = eval_checkpoint(str(ckpt), params, device)
        split_match = classical is not None and sorted(classical['test_indices']) == test_idx
        opts = [classical['opt_by_idx'].get(i) for i in test_idx] if classical else [None] * len(test_idx)
        n_opt = sum(o is not None for o in opts)
        gap_mean = float(np.mean([g['repaired_size'] - o for g, o in zip(agg['per_graph'], opts) if o is not None])) if n_opt else None
        opt_mean = float(np.mean([o for o in opts if o is not None])) if n_opt else None
        rec = {
            'status': 'OK',
            'checkpoint': str(ckpt.relative_to(PROJECT_ROOT)),
            'model_type': params['model_type'],
            'paper_reported': paper.get(row, {}).get(col),
            'split_matches_classical': split_match,
            'n_with_exact_optimum': n_opt,
            'exact_optimum_mean': opt_mean,
            'paper_optimal_gurobi10s': paper.get(row, {}).get('optimal_gurobi10s'),
            'gap_repaired_vs_exact': gap_mean,
            **{k: v for k, v in agg.items() if k != 'per_graph'},
        }
        results[(row, col)] = rec
        print(json.dumps({k: v for k, v in rec.items()}, indent=2, default=str))

    # ------- assemble JSON / MD -------
    table = {}
    for (row, col), rec in results.items():
        table.setdefault(row, {})[col] = rec
    json.dump({'protocol': {
        'split': 'train_fraction=0.8, split_seed=0, test=last 10% (data.loader random_split)',
        'rounding': 'randomized hyperplanes, n_hyperplanes=1000, seed 0 (same as original eval)',
        'repair': 'greedy: iteratively add vertex covering most uncovered edges (lowest-index tie-break)',
        'mu_at_eval': 0,
        'note': 'paper_reported values parsed programmatically from neurips_2026 (1).tex tab:main_results; '
                'exact optima from results/classical_baselines/<ds>.json (akiba_iwata, per-graph)',
    }, 'table': table},
        open(OUT_DIR / 'main_table_repaired.json', 'w'), indent=2, default=str)

    with open(OUT_DIR / 'main_table_repaired.md', 'w') as f:
        f.write('# Feasibility-corrected re-evaluation of the main table\n\n')
        f.write('Protocol: identical test split (train_fraction=0.8, split_seed=0); randomized-hyperplane '
                'rounding (1000 hyperplanes, seed 0) identical to the paper eval; then greedy repair of '
                'uncovered edges; mu=0 at eval for +PD cells.\n\n')
        f.write('Repaired = feasible cover size after repair (lower is better). '
                'Gap = repaired - exact Akiba-Iwata optimum (mean per graph).\n\n')
        f.write('| Dataset | Paper value | Raw size | Infeasible % | Uncovered edges (mean) | Repaired | Gap vs exact opt | Split OK | Checkpoint |\n')
        f.write('|---|---|---|---|---|---|---|---|---|\n')
        n_ok = n_missing = 0
        for row in ROW_PARAMS:
            for col in METHOD_COLS:
                r = table[row][col]
                if r['status'] != 'OK':
                    n_missing += 1
                    continue
                n_ok += 1
                f.write(f"| {row} | {col} | paper={r['paper_reported']} | "
                        f"{r['raw_size_mean']:.2f} | {r['infeasible_pct']:.1f}% | "
                        f"{r['mean_uncovered_edges']:.2f} | {r['repaired_size_mean']:.2f} | "
                        f"{r['gap_repaired_vs_exact']:.2f} | {r['split_matches_classical']} | "
                        f"{r['checkpoint']} |\n")
        f.write(f'\nRe-evaluated cells: {n_ok}; MISSING checkpoints: {n_missing} '
                f'out of {len(ROW_PARAMS) * len(METHOD_COLS)} method cells.\n')
    print(f'wrote {OUT_DIR}/main_table_repaired.json / .md')

    if cli.with_ood:
        run_supplementary_ood(device)


def run_supplementary_ood(device):
    """Re-evaluate the OOD-table checkpoints (ALON=ALONGNN vs OptGNN, trained on
    synthetic [100,200], tested zero-shot on the 5 TU datasets) WITH repair.
    These are the only ALON-vs-OptGNN checkpoint pairs that exist on this
    machine; they support the OOD table (and the 'ALON consistently
    outperforms unsupervised solvers' claim), not the main table itself."""
    from model.baselines import build_liftmp_baseline
    from torch_geometric.loader import DataLoader

    out = {}
    for fam in ['ALONGNN', 'OptGNN']:
        for train_ds, rel in OOD_CKPTS[fam].items():
            ckpt = PROJECT_ROOT / rel
            param_files = list((PROJECT_ROOT / 'training_runs').glob(
                f'ood_{fam}_{train_ds}/paramhash*/params.txt'))
            params = json.load(open(param_files[0]))
            args = make_args(params, device)
            for tu in TU_DATASETS:
                torch.manual_seed(0)
                torch.cuda.manual_seed_all(0)
                args.dataset = tu
                ds = construct_dataset(args)
                ds, total_nodes = prepare_alon_dataset(ds)
                mu = torch.zeros(total_nodes, device=device)
                loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False)
                if fam == 'ALONGNN':
                    problem = get_problem(args)
                    model, _ = construct_model(args, mu)
                else:
                    problem = get_problem(args)  # score / constraint metrics
                    model = build_liftmp_baseline(args)
                sd = torch.load(ckpt, map_location='cpu', weights_only=False)
                model.load_state_dict(sd, strict=True)
                model.to(device).eval()
                sizes, unc, infeas = [], [], []
                with torch.no_grad():
                    for batch in loader:
                        batch.penalty = args.penalty
                        x_in, batch = featurize_batch(args, batch)
                        if fam == 'ALONGNN':
                            x_out = model(x_in, batch, mu)
                        else:
                            x_out = model(x_in, batch)
                        xp = random_hyperplane_projector(args, x_out, batch, problem.score,
                                                         n_hyperplanes=1000)
                        xp = torch.where(xp == 0, torch.ones_like(xp), xp).cpu().numpy()
                        ptr = batch.ptr.cpu().numpy()
                        for gi, dg in enumerate(batch.to_data_list()):
                            xv = xp[ptr[gi]:ptr[gi + 1]]
                            cover = xv > 0
                            eu, ev = undirected_edges(dg.edge_index)
                            n_unc = int((~(cover[eu] | cover[ev])).sum())
                            rep = greedy_repair(cover, eu, ev)
                            sizes.append(int(rep.sum()))
                            unc.append(n_unc)
                            infeas.append(n_unc > 0)
                out[f'{fam}_{train_ds}_{tu}'] = {
                    'checkpoint': str(ckpt.relative_to(PROJECT_ROOT)),
                    'n_graphs': len(sizes),
                    'repaired_size_mean': float(np.mean(sizes)),
                    'infeasible_pct': 100.0 * float(np.mean(infeas)),
                    'mean_uncovered_edges': float(np.mean(unc)),
                }
                print(f'{fam} {train_ds} -> {tu}: repaired={np.mean(sizes):.2f} '
                      f'infeas={100*np.mean(infeas):.1f}%')
                del model
                torch.cuda.empty_cache()
    json.dump(out, open(OUT_DIR / 'ood_tu_repaired.json', 'w'), indent=2)
    print(f'wrote {OUT_DIR}/ood_tu_repaired.json')


if __name__ == '__main__':
    main()

