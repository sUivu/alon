#!/usr/bin/env python3
"""Overnight W1 arms: DUAL-ONLY graph scaling and the D3 product retest.

Extends model/alon_upgrade.py (imported READ-ONLY) with one new norm mode:

  norm='dualgraph' : g_obj stays PER-NODE normalized (magnitude-blind, D0/D1
                     style -- this channel drove PROTEINS/WS quality), while
                     g_dual gets GRAPH-level mean-norm scaling (D2's
                     feasibility win).  D2 scaled BOTH channels and paid a
                     quality cost; this isolates the dual channel.
"""
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
_DUAL_TUNING = Path(__file__).resolve().parents[1] / 'dual_tuning'
if str(_DUAL_TUNING) not in sys.path:
    sys.path.insert(0, str(_DUAL_TUNING))

from torch_scatter import scatter_add  # noqa: E402
from model.alon_a_fair import EPS  # noqa: E402
from model.alon_upgrade import (  # noqa: E402,F401
    UpgradeALONGNN, arm_spec,
)


class DualGraphLiftLayer(nn.Module):
    """h = W [x || pernode(g_obj) || graphscale(g_dual)]; spherical out."""

    def __init__(self, in_channels, lift_ratio=0.1, norm='dualgraph'):
        super().__init__()
        self.norm = norm
        self.lift_ratio = lift_ratio
        self.lin = nn.Linear(3 * in_channels, in_channels)

    @staticmethod
    def _graph_scale(g, batch):
        n = g.size(0)
        if batch is None or not hasattr(batch, 'ptr') or batch.ptr is None:
            gi = torch.zeros(n, dtype=torch.long, device=g.device)
            ng = 1
        else:
            ptr = batch.ptr
            ng = ptr.numel() - 1
            gi = torch.searchsorted(ptr[1:], torch.arange(n, device=g.device),
                                    right=True)
        norm = g.norm(dim=1)
        denom = scatter_add(norm, gi, dim=0, dim_size=ng)
        counts = torch.zeros(ng, device=g.device)
        counts.scatter_add_(0, gi, torch.ones_like(norm))
        mean_norm = denom / counts.clamp(min=1.0)
        return g / (mean_norm[gi].unsqueeze(1) + EPS)

    def forward(self, x, g_obj, g_dual, batch):
        if self.norm == 'dualgraph':
            obj_ch = F.normalize(g_obj, dim=1, eps=EPS)
            dual_ch = self._graph_scale(g_dual, batch)
        else:
            raise ValueError(self.norm)
        return F.normalize(self.lin(torch.cat([x, obj_ch, dual_ch], dim=1)),
                           dim=1, eps=EPS)


class DualGraphALONGNN(nn.Module):
    """L identical DualGraphLiftLayers; exposes the last node violation c."""

    def __init__(self, grad_layer, in_channels, num_layers=16, lift_ratio=0.1):
        super().__init__()
        self.grad_layer = grad_layer
        self.layers = []
        for i in range(num_layers):
            layer = DualGraphLiftLayer(in_channels, lift_ratio=lift_ratio)
            self.add_module(f'layer_{i}', layer)
            self.layers.append(layer)
        self.last_c = None

    def forward(self, x, batch, mu=None):
        for layer in self.layers:
            g_obj, g_dual, c = self.grad_layer(x, batch, mu)
            x = layer(x, g_obj, g_dual, batch)
        self.last_c = c
        return x


def build_overnight_network(cfg, problem, rank, num_layers, lift_ratio,
                            create_graph=True):
    """cfg: dict with residual / c_signal / norm.  For norm in
    ('pernode','graph') delegates to the upgrade harness (transfer configs);
    'dualgraph' uses the dual-only-scaled network here."""
    from model.alon_upgrade import UpgradeALONGNN
    from train_sweep import TunableGradLayer  # dual_tuning, read-only import
    grad = TunableGradLayer(problem, residual=cfg['residual'],
                            c_signal=cfg['c_signal'],
                            create_graph=create_graph)
    if cfg['norm'] == 'dualgraph':
        return DualGraphALONGNN(grad, in_channels=rank, num_layers=num_layers,
                               lift_ratio=lift_ratio)
    return UpgradeALONGNN(grad, in_channels=rank, num_layers=num_layers,
                         lift_ratio=lift_ratio, norm=cfg['norm'])
