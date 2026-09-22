# ALON — Augmented Lagrangian Optimization Neural Network for Minimum Vertex Cover

ALON is a graph neural network solver for (minimum) vertex cover that trains
with an **augmented Lagrangian method (ALM)**: instead of learning amortized
penalties, the network carries explicit node-wise dual variables that are
updated by dual-ascent steps during training, with linear penalty annealing.
At evaluation, duals are set to zero, cover scores are decoded by
**randomized-hyperplane rounding** (1000 hyperplanes, seed 0), and any
remaining uncovered edges are fixed by **greedy repair** (repeatedly add the
vertex covering the most uncovered edges, lowest-index tie-break). The
reported quantity is always the size of the *repaired, feasible* cover
(`violation == 0` is asserted per graph).

This package contains the source needed to train the models and to run the
evaluation protocol that produces the main-table numbers, plus a CPU smoke
test (`smoke/`).

## Naming note

The paper's method is called **ALON** (Augmented Lagrangian Optimization
Network). Paper↔code name mapping:

| paper | code |
|---|---|
| ALON (Table-1 model) | class `ALONA_Fair` in `model/alon_a_fair.py`, trained by the Table-1 harness `scripts/matrix_fill/train_t1.py` |
| +AL on GIN | `--model_type ALONGIN` |
| +AL on GAT | `--model_type ALONGAT` |
| +AL on GCNN | `--model_type ALONGCNN` |
| +AL on GatedGCNN | `--model_type ALONGatedGCNN` |

The four `+AL` backbone variants instantiate class `ALON_Hybrid_Network` (in
`model/models.py`, trained by `train.py`/`train_dp.py`).

## Repository layout

```
train.py                  raw-backbone training (GIN/GAT/GCNN/GatedGCNN)
train_dp.py               ALM-variant training (ALONGIN/ALONGAT/ALONGCNN/ALONGatedGCNN)
train_alon_upgrade.py     shared dataset specs / helpers for the ALON harness
train_baseline.py         OptGNN/LiftMP baseline (implemented in our framework, model/baselines.py)
problem/                  objective, constraint, residual (losses.py), SDP/Gurobi refs
model/                    GNN architectures + ALM training loop
  baselines.py            fused OptGNN/LiftMP baseline loss + constructor
data/                     dataset construction (synthetic generators + TUDatasets)
utils/                    argument parsing, graph utilities
scripts/
  matrix_fill/train_t1.py           ALON Table-1 training harness (--arm T1)
  dual_overnight/train_arm.py       ALON training arm implementation (+ built-in eval)
  dual_overnight/overnight_{configs,model}.py
  dual_tuning/{train_sweep,configs}.py   dual-schedule configs (T1)
  retrain_campaign/eval_repaired_cells.py  protocol evaluation (incl. OptGNN checkpoints)
  retrain_full/{train_arm_ep1000,eval_rb200_optgnn_ep1000}.py  RB ep1000 train/eval
  classical_baselines.py            classical solvers (reference optima)
  eval_alon_a_fair.py              single-checkpoint protocol evaluation
  reeval_main_feasible.py           full main-table re-evaluation driver
classical_solvers/        NUMVC + FastWVC binaries (see classical_baselines.py)
ckpts/                    one small smoke-test checkpoint (ENZYMES, 232 KB)
smoke/                    expected smoke outputs
requirements.txt          exact environment (pip freeze)
```

## Dependencies

Python **3.10.12**. Exact pins in `requirements.txt`; key ones:

- `torch==2.6.0+cu124` (CPU wheels work for evaluation and the smoke test;
  bitwise-identical training needs the same build)
- `torch-geometric==2.7.0`, `torch_scatter/torch_sparse/torch_cluster`
  (+pt26cu124 builds)
- **`networkx==3.4.2`** — the synthetic graph generators (WS/BA/HK rewiring,
  ForcedRB) are RNG-sensitive to the networkx version. Install this pin
  *before* generating any dataset, or every downstream number changes.
- `numpy==2.2.6`, `scipy==1.15.3`, `cvxpy==1.7.5`
- `gurobipy==13.0.1` (optional): only for the Gurobi reference solver /
  LP-relaxation values. The default pip license is size-limited, which is fine
  for these graph sizes. Everything else runs without it.

```bash
python3.10 -m venv env && source env/bin/activate
pip install -r requirements.txt
# CPU-only machines: pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cpu
```

## Data

- **Synthetic (generated in-code, deterministic given the networkx pin and
  `seed=0`):**
  - ErdosRenyi / BarabasiAlbert / WattsStrogatz / PowerlawCluster ("HK") at
    scales [50,100], [100,200], [400,500] (2000 graphs each; e.g. WS:
    `gen_n 50 100 --gen_k 4 --gen_p 0.25`)
  - ForcedRB-200 / ForcedRB-500 (`gen_n 6 15 --gen_k 12 21` and
    `gen_n 20 34 --gen_k 10 29`, 1000 graphs each)
- **TU datasets** (MUTAG, PROTEINS, ENZYMES, IMDB-BINARY, COLLAB): standard
  [TUDatasets](https://www.chrsmrrs.com/graphkerneldatasets/), downloaded
  automatically on first use by `torch_geometric.datasets.TUDataset` into
  `datasets/` (needs network access once).
- **OptGNN baseline (`train_baseline.py`)**: the OptGNN baseline is implemented
  in our framework following the formulation of Yau et al., "Are Graph Neural
  Networks Necessary for Graph-Tailored Optimization?" (ICML 2024) — the
  original public implementation is not required; see `model/baselines.py`.
  ALON, raw-backbone and OptGNN training/evaluation all run out of the box.

Split protocol (everywhere): `train_fraction=0.8`, `split_seed=0`,
validation 10%, **test = last 10%**.

## Usage

GPU training commands assume `CUDA_DEVICE_ORDER=PCI_BUS_ID` and a free GPU.
All commands run from the repository root.

### 1. Train ALON (Table-1 harness)

```bash
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 \
python -u scripts/matrix_fill/train_t1.py --arm T1 --seed 1 --epochs 200 --dataset ER_50_100
```

`--dataset` choices: `{ER,BA,WS,HK}_{50_100,100_200,400_500}`,
`IMDB-BINARY`, `PROTEINS`, `COLLAB`, `RB200`, `RB500`. Synthetic/TU rows use
200 epochs; the RB rows use 1000 epochs (`--epochs 1000 --dataset RB200`).
At the end of training the harness automatically writes the protocol
evaluation to
`results/dual_overnight/eval_T1_<DATASET>_s<SEED>.json`: a test-split
*statusquo* run (`mu=0`, 1000 hyperplanes, seed 0, greedy repair) whose
`mean_repaired` is the reported number.

### 2. Train raw backbones / OptGNN (example cells)

```bash
# raw + ALM backbone variants (example: GIN on ER [50,100], 200 epochs)
python -u train.py --dataset ErdosRenyi --num_graphs 2000 --batch_size 16 \
    --gen_n 50 100 --gen_p 0.15 --problem_type vertex_cover --seed 1 \
    --rank 32 --num_layers 16 --epochs 200 --valid_freq 100 --model_type GIN \
    --penalty_s 1.0 --penalty_e 1.0 --lambda_freq 250 --prefix runs/gin_er

# ALM variant: same command with --model_type ALONGAT
#   --lift_ratio 0.1 --penalty_s -0.5 --penalty_e 2.5 --lambda_ratio 0.02
#   --linear_annealing --lambda_freq 1

# OptGNN (implemented in our framework; see model/baselines.py and the
# OptGNN citation above)
python -u train_baseline.py --model_type LiftMP --lift_ratio 0.2 \
    --problem_type vertex_cover --seed 1 --rank 32 --num_layers 16 \
    --epochs 200 --valid_freq 20 --num_graphs 2000 --dataset ErdosRenyi \
    --batch_size 16 --gen_n 50 100 --gen_p 0.15 --prefix runs/optgnn_er
```

TU-dataset cells: `--dataset MUTAG/ENZYMES/PROTEINS/IMDB-BINARY/COLLAB
--epochs 1000` with batch sizes 2/5/9/8/40 respectively; RB cells use
`--dataset ForcedRB --gen_n 6 15 --gen_k 12 21` (RB200) / `--gen_n 20 34
--gen_k 10 29` (RB500), `--rank 64 --epochs 1000 --num_graphs 1000`.

### 3. Evaluate a checkpoint under the protocol

The protocol (identical for every method): load the checkpoint, set duals
`mu = 0`, decode with randomized-hyperplane rounding (n_hyperplanes=1000,
seed 0), greedy-repair uncovered edges, report the repaired cover size;
`violation == 0` is asserted per graph.

```bash
# ALON checkpoints: the statusquo eval in eval_T1_*.json (written by training)
# IS the protocol evaluation. For an OptGNN checkpoint:
python scripts/retrain_campaign/eval_repaired_cells.py   # see file for cell config
# Single-checkpoint interactive evaluation (CPU works):
CUDA_VISIBLE_DEVICES= python scripts/eval_alon_a_fair.py \
    --checkpoint ckpts/alon_a_fair_C0_ENZYMES_seed0_screen_ep200/alon_a_fair.pt \
    --dataset ENZYMES --splits test --limit 5 --ks 0 --rho_sweep 0.2 --out /tmp/eval.json
```

### 4. Classical reference values

```bash
python scripts/classical_baselines.py --datasets ER_50_100 --limit 2 \
    --cutoff 5 --workers 1 --out_dir /tmp/cb_smoke
# full dataset: --all --cutoff 10 --seed 0 --workers 24
```

Greedy / matching / LP / Gurobi / NUMVC / FastWVC run out of the box (the
NUMVC and FastWVC binaries are in `classical_solvers/`; Gurobi needs the
optional `gurobipy` license). Exact per-instance optima referenced in the
paper come from the PACE-2019 exact-track solver (Akiba–Iwata), not bundled;
it is available from the public PACE 2019 vc-benchmark repository
(`build.sh`-style compilation with g++ / C++17) and is wired in
`scripts/classical_baselines.py` as the `akiba_iwata` entry.

### 5. Smoke test (CPU, < 2 minutes)

```bash
# A. classical sizes (deterministic given the networkx pin)
python scripts/classical_baselines.py --datasets ER_50_100 --limit 2 \
    --cutoff 5 --workers 1 --out_dir /tmp/cb_smoke
python - <<'EOF'
import json
got = json.load(open('/tmp/cb_smoke/ER_50_100.json'))['summary']
exp = json.load(open('smoke/expected_classical_smoke.json'))['summary']
for k in sorted(exp):
    a, b = exp[k].get('avg_size'), got.get(k, {}).get('avg_size')
    if b is None:  # solver not available in this environment
        print(f"{k:14s} expected={a} got=NOT-RUN (skip)")
        continue
    print(f"{k:14s} expected={a} got={b} {'OK' if a == b else 'MISMATCH'}")
EOF

# B. neural checkpoint under the protocol, bit-compared to the expected JSON
CUDA_VISIBLE_DEVICES= python scripts/eval_alon_a_fair.py \
    --checkpoint ckpts/alon_a_fair_C0_ENZYMES_seed0_screen_ep200/alon_a_fair.pt \
    --dataset ENZYMES --splits test --limit 5 --ks 0 --rho_sweep 0.2 \
    --out /tmp/smoke_eval.json
python - <<'EOF'
import json
got = json.load(open('/tmp/smoke_eval.json'))
exp = json.load(open('smoke/expected_output.json'))
for split in ('test',):
    exp_runs = {r['config']: r for r in exp['splits'][split]['runs']}
    got_runs = {r['config']: r for r in got['splits'][split]['runs']}
    for cfg, e in exp_runs.items():
        g = got_runs[cfg]
        print(cfg, 'rep_mean', e['mean_repaired'], '==', g['mean_repaired'],
              'OK' if e['mean_repaired'] == g['mean_repaired'] else 'MISMATCH')
EOF
```

Expected: all classical `avg_size` values and both neural `rep_mean` values
(16.2) match exactly — the reported cover sizes are convention- and
platform-independent; only the timing fields vary.

## Seed convention

- **Training / data generation / classical solvers:** seed 0 for the primary
  (original) runs; seed 1 for the independent second-seed replication of
  every Table-1 cell.
- **Split:** always `split_seed=0` (train 80% / val 10% / test = last 10%).
- **Rounding:** always seed 0 with 1000 random hyperplanes.

## Source convention

The edge residual in `problem/losses.py` is the classical **un-halved** form
`g_ij = <e1 - v_i, e1 - v_j>` (values in {0, 4} for 0/1 vertex scores),
matching the equation in the paper; numerical constants in the loss chains
(`batch.penalty`, the mu-step scaling) reabsorb the corresponding factors.
