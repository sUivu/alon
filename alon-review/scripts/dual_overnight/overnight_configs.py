#!/usr/bin/env python3
"""Overnight W1 arm configs (isolated namespace; extends the dual-tuning
grid with the UNTESTED high-value cells).

  E1 dualgraph_signed : g_obj per-node normalized + g_dual graph-scaled,
                        signed residual, alg1 dual (D1 default schedule).
                        Isolates the dual channel of D2's scaling win.
  E2 dualgraph_prod_stab : D3 retest done right -- product residual + dual
                        graph scaling + PER-EPOCH dual (freq 1) + proximal
                        mu-stabilization (eta_d 0.5) + 50-epoch warm-up.
                        (D3's PROTEINS failure may have been mu divergence.)
  E3 dualgraph_prod   : product residual + dual graph scaling, D1 schedule.
  E4 dualgraph_agg    : dualgraph norm + freq 5 + rho_init 0.05 + rho_max 1.0
                        (mirror of dual-tuning T12 but obj stays pernode).
  E5 + : transfer configs -- best dual-tuning T-config, norm left as that
         config's; resolved at launch from results/dual_tuning/sweep.
"""
import sys
from pathlib import Path

_DUAL_TUNING = Path(__file__).resolve().parents[1] / 'dual_tuning'
if str(_DUAL_TUNING) not in sys.path:
    sys.path.insert(0, str(_DUAL_TUNING))

_BASE = dict(delta=0.5, mu_mode='none', eta_d=0.0, mu_cap=None, warmup=0)

OVERNIGHT_ARMS = {
    'E1': dict(_BASE, norm='dualgraph', residual='signed', c_signal='signed',
               lambda_freq=25, rho_init=0.01, tau=2.0, rho_max=0.25,
               label='dual_only_graph_scaled'),
    'E2': dict(_BASE, norm='dualgraph', residual='product',
               c_signal='product', lambda_freq=1, rho_init=0.01, tau=2.0,
               rho_max=0.25, mu_mode='prox', eta_d=0.5, warmup=50,
               label='prod_dualgraph_freq1_prox_warm50'),
    'E3': dict(_BASE, norm='dualgraph', residual='product',
               c_signal='product', lambda_freq=25, rho_init=0.01, tau=2.0,
               rho_max=0.25, label='prod_dualgraph_freq25'),
    'E4': dict(_BASE, norm='dualgraph', residual='signed', c_signal='signed',
               lambda_freq=5, rho_init=0.05, tau=2.0, rho_max=1.0,
               label='dualgraph_agg_freq5_rho0.1_cap2'),
}
ARM_ORDER = ['E1', 'E2', 'E3', 'E4']


def get_arm_cfg(name):
    """E-arms from this module; T-configs (transfers) from dual_tuning."""
    if name in OVERNIGHT_ARMS:
        return dict(OVERNIGHT_ARMS[name])
    from configs import get_config  # dual_tuning
    return get_config(name)
