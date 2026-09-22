#!/usr/bin/env python3
"""One-off: paper-protocol eval of the OptGNN RB200 ep1000 checkpoint
(1000 hyperplanes seed 0 + greedy repair, paper split), with per-graph
split verification against the ALON RB200 ep1000 eval."""
import importlib.util
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'scripts' / 'retrain_campaign'))

_spec = importlib.util.spec_from_file_location(
    'eval_repaired_cells_mod',
    ROOT / 'scripts' / 'retrain_campaign' / 'eval_repaired_cells.py')
E = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(E)

CKPT = (ROOT / 'training_runs' / 'retrain_full' / 'RB200_optgnn_ep1000')
rd = sorted(CKPT.glob('paramhash*'))[0]
params = json.load(open(rd / 'params.txt'))
OUT = ROOT / 'results' / 'retrain_full' / 'eval_RB200_optgnn_ep1000.json'

device = torch.device('cpu')
agg, test_idx = E.eval_optgnn_checkpoint(str(rd / 'final_model.pt'),
                                         params, device)

# Optional split verification against the ALON RB200 ep1000 eval (the file is
# produced by train_arm_ep1000.py + its built-in statusquo eval; skip if absent).
_t1_path = ROOT / 'results' / 'retrain_full' / 'eval_RB200_T1_ep1000.json'
if _t1_path.exists():
    t1 = json.load(open(_t1_path))
    ti_t1 = sorted(r['idx'] for r in t1['splits']['test']['runs'][0]['per_graph'])
    split_match = ti_t1 == sorted(test_idx)
else:
    split_match = None

rec = dict(agg)
rec.update({
    'checkpoint': str(rd / 'final_model.pt'),
    'model_type': params.get('model_type'),
    'arm': 'optgnn', 'dataset': 'RB200', 'epochs': params.get('epochs'),
    'split_matches_alon_ep1000': bool(split_match),
    'test_indices': list(map(int, sorted(test_idx))),
})
json.dump(rec, open(OUT, 'w'), indent=1)
print(f"[eval] RB200|optgnn ep1000: repaired={rec['repaired_size_mean']:.3f} "
      f"raw={rec['raw_size_mean']:.3f} infeas={rec['infeasible_pct']:.1f}% "
      f"n={rec['n_test_graphs']} split_match_vs_alon={split_match}")
print(f"[done] -> {OUT}")
