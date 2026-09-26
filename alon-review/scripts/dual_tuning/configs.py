#!/usr/bin/env python3
"""Workstream A grid: dual-hyperparameter sweep configs (isolated namespace).

Base harness: model/alon_upgrade.py D1 (pernode norm + signed residual +
alg1 dual running), imported READ-ONLY.  Each config varies the dual-side
hyperparameters only; data / budget / split / eval protocol are identical to
the upgrade campaign (200 epochs, PROTEINS, penalty anneal +0.5 -> +2.5,
statusquo eval = rounding seed 0 / 1000 hyperplanes / paper greedy repair).

Grid axes and the chosen 13-config subset (one-factor-at-a-time around the
D1 control, plus two combined 'best-guess' aggressive configs):

  T0  control            : D1 replica (validates the copied harness)
  T1  freq1              : lambda_freq 1   (Alg 1 says per-epoch)
  T2  freq5              : lambda_freq 5
  T3  rho01              : rho_init 0.05
  T4  rho05cap2          : rho_init 0.25 + cap 1.0 (aggressive rho)
  T5  tau4cap2           : tau 4 + cap 2.0 (faster growth, higher ceiling)
  T6  mucap5             : projection mu <- min(mu, 5) after each update
  T7  prox05             : proximal damping mu += rho*c - 0.5*(mu - mu_prev)
  T8  relu_c             : constraint signal relu(phi) instead of signed phi
  T9  product_c          : product residual g_ij everywhere (primal + dual)
  T10 warm50             : dual updates start after epoch 50
  T11 graph              : graph-level scaling BOTH channels (= D2 replica)
  T12 graph_agg          : graph scaling + freq 5 + rho_init 0.05 + cap 1.0
                           (tests: graph-scale REQUIRED for rho to matter)
"""

SWEEP_CONFIGS = {
    'T0': dict(norm='pernode', residual='signed', c_signal='signed',
               lambda_freq=25, rho_init=0.01, tau=2.0, rho_max=0.5,
               delta=0.5, mu_mode='none', eta_d=0.0, mu_cap=None,
               warmup=0, label='D1_replica_control'),
    'T1': dict(norm='pernode', residual='signed', c_signal='signed',
               lambda_freq=1, rho_init=0.01, tau=2.0, rho_max=0.5,
               delta=0.5, mu_mode='none', eta_d=0.0, mu_cap=None,
               warmup=0, label='freq1_per_epoch'),
    'T2': dict(norm='pernode', residual='signed', c_signal='signed',
               lambda_freq=5, rho_init=0.01, tau=2.0, rho_max=0.5,
               delta=0.5, mu_mode='none', eta_d=0.0, mu_cap=None,
               warmup=0, label='freq5'),
    'T3': dict(norm='pernode', residual='signed', c_signal='signed',
               lambda_freq=25, rho_init=0.05, tau=2.0, rho_max=0.25,
               delta=0.5, mu_mode='none', eta_d=0.0, mu_cap=None,
               warmup=0, label='rho_init_0.1'),
    'T4': dict(norm='pernode', residual='signed', c_signal='signed',
               lambda_freq=25, rho_init=0.25, tau=2.0, rho_max=1.0,
               delta=0.5, mu_mode='none', eta_d=0.0, mu_cap=None,
               warmup=0, label='rho_init_0.5_cap2'),
    'T5': dict(norm='pernode', residual='signed', c_signal='signed',
               lambda_freq=25, rho_init=0.01, tau=4.0, rho_max=1.0,
               delta=0.5, mu_mode='none', eta_d=0.0, mu_cap=None,
               warmup=0, label='tau4_cap2'),
    'T6': dict(norm='pernode', residual='signed', c_signal='signed',
               lambda_freq=25, rho_init=0.01, tau=2.0, rho_max=0.5,
               delta=0.5, mu_mode='cap', eta_d=0.0, mu_cap=5.0,
               warmup=0, label='mu_project_cap5'),
    'T7': dict(norm='pernode', residual='signed', c_signal='signed',
               lambda_freq=25, rho_init=0.01, tau=2.0, rho_max=0.5,
               delta=0.5, mu_mode='prox', eta_d=0.5, mu_cap=None,
               warmup=0, label='proximal_damp_0.5'),
    'T8': dict(norm='pernode', residual='signed', c_signal='relu',
               lambda_freq=25, rho_init=0.01, tau=2.0, rho_max=0.5,
               delta=0.5, mu_mode='none', eta_d=0.0, mu_cap=None,
               warmup=0, label='c_signal_relu'),
    'T9': dict(norm='pernode', residual='product', c_signal='product',
               lambda_freq=25, rho_init=0.01, tau=2.0, rho_max=0.5,
               delta=0.5, mu_mode='none', eta_d=0.0, mu_cap=None,
               warmup=0, label='product_residual'),
    'T10': dict(norm='pernode', residual='signed', c_signal='signed',
                lambda_freq=25, rho_init=0.01, tau=2.0, rho_max=0.5,
                delta=0.5, mu_mode='none', eta_d=0.0, mu_cap=None,
                warmup=50, label='dual_warmup_ep50'),
    'T11': dict(norm='graph', residual='signed', c_signal='signed',
                lambda_freq=25, rho_init=0.01, tau=2.0, rho_max=0.5,
                delta=0.5, mu_mode='none', eta_d=0.0, mu_cap=None,
                warmup=0, label='D2_replica_graph'),
    'T12': dict(norm='graph', residual='signed', c_signal='signed',
                lambda_freq=5, rho_init=0.05, tau=2.0, rho_max=1.0,
                delta=0.5, mu_mode='none', eta_d=0.0, mu_cap=None,
                warmup=0, label='graph_agg_freq5_rho0.1_cap2'),
}

CONFIG_ORDER = ['T0', 'T1', 'T2', 'T3', 'T4', 'T5', 'T6', 'T7', 'T8',
                'T9', 'T10', 'T11', 'T12']


def get_config(name):
    if name not in SWEEP_CONFIGS:
        raise KeyError(f'unknown sweep config {name}; have {CONFIG_ORDER}')
    return dict(SWEEP_CONFIGS[name])
