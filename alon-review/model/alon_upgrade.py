#!/usr/bin/env python3
"""ALON upgrade campaign -- mechanism/parameter arms D0-D4 (isolated namespace).

Extends the fair bake-off (model/alon_a_fair.py, imported read-only) with the
factorial cells the fair study never tested, plus a dual-ascent schedule that
actually runs within a 200-epoch budget (H3):

  D0 'baseline'            : per-node norm + signed residual + released dual
                             schedule (lambda_freq 250, ratio 0.02 -> never
                             fires at 200 epochs).  Reference ALON.
  D1 'dual_running'        : D0 but dual ascent runs (lambda_freq 25, Alg-1
                             adaptive rho_t).  Isolates H3.
  D2 'graph_scaled_signed' : graph-level gradient scaling + signed residual +
                             dual running.  Isolates H1 (per-node normalize
                             makes rho provably irrelevant).
  D3 'graph_scaled_product': graph-level scaling + PRODUCT residual + dual
                             running.  H1+H2+H3 together.
  D4 'norm_product'        : per-node norm + PRODUCT residual + dual running.
                             Completes the factorial.

Graph-level scaling (norm='graph'): for channel g in {g_obj, g_dual},
    g_i <- g_i / (mean_j ||g_j|| + eps)
where the mean runs over the nodes OF THE SAME GRAPH (batch-aware).  This keeps
per-graph relative scale information (which per-node normalize destroys) while
preventing gradient blow-up across graphs of different sizes.

Dual ascent ('alg1' schedule, per paper Alg 1): every lambda_freq epochs
    mu <- relu(mu + rho_t * c_mean)          (c_mean = mean node violation over
                                              the interval, node-wise)
    rho_t <- tau * rho_t if c did not drop below delta * c_prev else rho_t
    rho_t clamped to [rho_init, rho_max]

All arms keep the node-wise dual lambda_ij = mu_i + mu_j (never materialize
edge duals) and the identical training budget.
"""
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn
from torch_scatter import scatter_add

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from model.alon_a_fair import (  # noqa: E402,F401
    EPS, FairGradLayer, edge_residual, node_violation, alg1_rho_update, _PHI,
)

# ----------------------------------------------------------------------------
# Arm specifications
# ----------------------------------------------------------------------------
UPGRADE_ARMS = {
    'D0': dict(residual='signed', norm='pernode', dual='released',
               label='baseline_alon'),
    'D1': dict(residual='signed', norm='pernode', dual='alg1',
               label='dual_running'),
    'D2': dict(residual='signed', norm='graph', dual='alg1',
               label='graph_scaled_signed'),
    'D3': dict(residual='product', norm='graph', dual='alg1',
               label='graph_scaled_product'),
    'D4': dict(residual='product', norm='pernode', dual='alg1',
               label='norm_product'),
}
ARM_ORDER = ['D0', 'D1', 'D2', 'D3', 'D4']


def arm_spec(name):
    if name not in UPGRADE_ARMS:
        raise KeyError(f'unknown arm {name}; have {list(UPGRADE_ARMS)}')
    return UPGRADE_ARMS[name]


# ----------------------------------------------------------------------------
# Graph-scaled lift layer (batch-aware graph-mean normalization)
# ----------------------------------------------------------------------------
class UpgradeLiftLayer(nn.Module):
    """h = W [x || ch_obj || ch_dual]; spherical projection at the end.

    norm='pernode': channels are F.normalize(g) per node (released style,
                    dual channel scaled by lift_ratio).
    norm='graph'  : channels are g / (graph-mean of ||g|| + eps) for BOTH
                    g_obj and g_dual (H1 fix: preserves magnitude information
                    up to one scalar per graph).
    """

    def __init__(self, in_channels, lift_ratio=0.1, norm='pernode'):
        super().__init__()
        self.norm = norm
        self.lift_ratio = lift_ratio
        self.lin = nn.Linear(3 * in_channels, in_channels)

    @staticmethod
    def _graph_scale(g, batch):
        """g_i / (mean_{j in graph(i)} ||g_j|| + eps)."""
        n = g.size(0)
        if batch is None or not hasattr(batch, 'ptr') or batch.ptr is None:
            gi = torch.zeros(n, dtype=torch.long, device=g.device)
            ng = 1
        else:
            ptr = batch.ptr
            ng = ptr.numel() - 1
            gi = torch.searchsorted(ptr[1:], torch.arange(n, device=g.device),
                                    right=True)
        norm = g.norm(dim=1)                                     # [N]
        denom = scatter_add(norm, gi, dim=0, dim_size=ng)        # [G]
        counts = torch.zeros(ng, device=g.device)
        counts.scatter_add_(0, gi, torch.ones_like(norm))
        mean_norm = denom / counts.clamp(min=1.0)
        return g / (mean_norm[gi].unsqueeze(1) + EPS)

    def forward(self, x, g_obj, g_dual, batch):
        if self.norm == 'pernode':
            obj_ch = F.normalize(g_obj, dim=1, eps=EPS)
            dual_ch = self.lift_ratio * F.normalize(g_dual, dim=1, eps=EPS)
        elif self.norm == 'graph':
            obj_ch = self._graph_scale(g_obj, batch)
            dual_ch = self._graph_scale(g_dual, batch)
        else:  # 'none'
            obj_ch, dual_ch = g_obj, g_dual
        return F.normalize(self.lin(torch.cat([x, obj_ch, dual_ch], dim=1)),
                           dim=1, eps=EPS)


class UpgradeALONGNN(nn.Module):
    """L identical lift layers; exposes the last node violation c."""

    def __init__(self, grad_layer, in_channels, num_layers=16, lift_ratio=0.1,
                 norm='pernode'):
        super().__init__()
        self.grad_layer = grad_layer
        self.layers = []
        for i in range(num_layers):
            layer = UpgradeLiftLayer(in_channels, lift_ratio=lift_ratio,
                                     norm=norm)
            self.add_module(f'layer_{i}', layer)
            self.layers.append(layer)
        self.last_c = None

    def forward(self, x, batch, mu=None):
        for layer in self.layers:
            g_obj, g_dual, c = self.grad_layer(x, batch, mu)
            x = layer(x, g_obj, g_dual, batch)
        self.last_c = c
        return x


def build_network(arm, problem, rank, num_layers, lift_ratio,
                  create_graph=True):
    spec = arm_spec(arm)
    grad = FairGradLayer(problem, residual=spec['residual'],
                         create_graph=create_graph)
    return UpgradeALONGNN(grad, in_channels=rank, num_layers=num_layers,
                         lift_ratio=lift_ratio, norm=spec['norm'])


# ----------------------------------------------------------------------------
# Dual-schedule step (training time)
# ----------------------------------------------------------------------------
def dual_step(mu_param, add_mu, n_batches, schedule, rho_t, c_mean_prev,
              rho_init=0.01, delta=0.5, tau=2.0, rho_max=0.25,
              lambda_ratio=0.01, residual='signed'):
    """One dual update at epoch lambda_freq boundaries.

    schedule='released': mu += lambda_ratio * sum_c_over_interval  (released code)
    schedule='alg1'    : mu += rho_t * mean_c_per_node; rho_t adapts per Alg 1
                         (grow xtau if violation did not drop below delta of
                         the previous interval's violation).

    rho/lambda_ratio are stored in the un-halved residual convention; product
    residuals are unchanged, so 2/PHI gives the reference step scale.

    Returns updated (mu, rho_t, c_mean).
    """
    c_mean = float(add_mu.abs().sum()) / max(n_batches, 1) / _PHI[residual]
    if schedule == 'released':
        with torch.no_grad():
            mu_param += lambda_ratio * (2.0 / _PHI[residual]) * \
                add_mu.clamp(min=0.0)
            mu_param.clamp_(min=0.0)
        return mu_param, rho_t, c_mean
    # alg1
    if c_mean_prev is not None and c_mean > delta * c_mean_prev:
        rho_t = min(tau * rho_t, rho_max)
    with torch.no_grad():
        mu_param += rho_t * (2.0 / _PHI[residual]) * add_mu / max(n_batches, 1)
        mu_param.clamp_(min=0.0)
    return mu_param, max(rho_t, rho_init), c_mean
