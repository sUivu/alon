#!/usr/bin/env python3
"""matrix-fill: T1-config PDGNO training for grid rows missing from
train_pdgno_upgrade.DATASET_SPECS.

Thin wrapper around scripts/dual_overnight/train_arm.py (READ-ONLY reuse):
patches the module-level DATASET_SPECS / REF_NAME with the matrix-fill rows,
then delegates to train_arm.main().  T1 = per-epoch dual (lambda_freq 1),
penalty anneal -0.5 -> 2.5, lift_ratio 0.1, lambda_ratio 0.02,
linear_annealing -- the validated main-table PDGNO arm.

Namespace (isolated, non-colliding):
  ckpts/dual_overnight_T1_<ROW>_s<seed>/pdgno_upgrade.pt
  results/dual_overnight/eval_T1_<ROW>_s<seed>.json   (statusquo eval = new
  protocol: mu=0, 1000 hyperplanes seed 0, greedy repair, violation==0)

Usage:
  CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=2 \
  /app/pdgno-env/bin/python scripts/matrix_fill/train_t1.py \
      --arm T1 --dataset ER_50_100 --seed 0
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'scripts' / 'dual_overnight'))
sys.path.insert(0, str(ROOT / 'scripts' / 'dual_tuning'))

import train_pdgno_upgrade as U  # noqa: E402


def _spec(dataset, batch_size, gen_n=None, gen_m=None, gen_k=None,
          gen_p=None, rank=32):
    return dict(dataset=dataset, num_graphs=2000, gen_n=gen_n, gen_m=gen_m,
                gen_k=gen_k, gen_p=gen_p, batch_size=batch_size, rank=rank)


MATRIX_SPECS = {}
for short, ds, gen in [
    ('ER', 'ErdosRenyi', dict(gen_n=None, gen_p=[0.15])),
    ('BA', 'BarabasiAlbert', dict(gen_n=None, gen_m=[4])),
    ('WS', 'WattsStrogatz', dict(gen_n=None, gen_k=[4], gen_p=[0.25])),
    ('HK', 'PowerlawCluster', dict(gen_n=None, gen_m=[4], gen_p=[0.25])),
]:
    for scale in ['50_100', '100_200', '400_500']:
        lo, hi = scale.split('_')
        kw = dict(gen)
        kw['gen_n'] = [int(lo), int(hi)]
        row = f'{short}_{scale}'
        MATRIX_SPECS[row] = _spec(ds, 16, **kw)
        U.REF_NAME[row] = row  # classical_baselines stem == row name
MATRIX_SPECS['IMDB-BINARY'] = _spec('IMDB-BINARY', 8)
U.REF_NAME['IMDB-BINARY'] = 'IMDB-BINARY'

U.DATASET_SPECS.update(MATRIX_SPECS)

import train_arm  # noqa: E402

train_arm.DATASET_SPECS = U.DATASET_SPECS

if __name__ == '__main__':
    # dispatcher passes --prefix <tag> for liveness tracking; train_arm's
    # argparse does not know it (paths are derived from arm/dataset/seed)
    if '--prefix' in sys.argv:
        i = sys.argv.index('--prefix')
        del sys.argv[i:i + 2]
    train_arm.main()
