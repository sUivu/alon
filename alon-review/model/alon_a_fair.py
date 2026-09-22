#!/usr/bin/env python3
"""
ALON-A FAIR bake-off module (isolated namespace).

WHY THIS EXISTS
---------------
Earlier comparisons froze the base primal weights ``theta`` and swapped the
gradient/constraint *representation* in front of them.  That is invalid: theta
was optimised under the old representation, so any difference measures the
mismatch, not the representation.  Here each candidate representation owns a
theta that is trained FROM SCRATCH under that representation with an identical
budget / schedule / seed / data split.  Only the representation varies.

CANDIDATES (all keep the node-wise dual lambda_ij = mu_i + mu_j; NO per-edge dual)
---------------------------------------------------------------------------------
  C0 'baseline_optgnn_style'  : signed inner-product residual
                                phi_ij = <x_i-e1, x_j-e1>/2, g_obj/g_dual
                                L2-normalised per node (released code).
  C1 'algo1_faithful'         : product residual
                                phi_ij = (1-<x_i,e1>)(1-<x_j,e1>) >= 0,
                                NO normalisation of either gradient; only the
                                final spherical projection h/||h||.
  C2 'algo1_plus_channel'     : C1 + an explicit per-node dual-magnitude
                                channel log1p(mu_i) concatenated to the lift
                                input (isolates "magnitude channel").
  C3 'code_residual_algo1_scale' : signed residual (as C0) but NO normalisation
                                (isolates the normalisation axis from the
                                residual axis).

The dual ``mu`` enters the primal ONLY through the AL term <mu, c> with
c_i = sum_{j in N(i)} phi_ij (symmetric edge_index => lambda_ij = mu_i+mu_j),
plus (C2 only) the explicit magnitude channel.  No per-edge dual is ever
materialised.

This module IMPORTS (never edits) the untouched architecture and the v1 helpers
in ``model.alon_a``.  It does not touch any other worker's files.
"""
import math
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch_scatter import scatter_add

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Reuse (no edits) the stable v1 helpers.
from model.alon_a import (  # noqa: E402,F401
    DualInitializer, structural_features, featurize, cover_constraint,
    _graph_edges,
)

EPS = 1e-12

# --- classical un-halved residual convention (2026-09) ----------------------
# edge_residual('signed') returns the un-halved inner product
#     g_ij = <e1 - v_i, e1 - v_j>          (rank-one values {0, 4})
# All quadratic/linear coefficients below are therefore expressed in the
# un-halved convention.  'product' residuals are unchanged by the conversion,
# so their coefficients are renormalised by the factors below to keep every
# historical run bit-identical.
_PHI = {'signed': 2.0, 'product': 1.0}    # residual scale change vs halved
_PHI2 = {'signed': 4.0, 'product': 1.0}   # squared scale change vs halved

# ----------------------------------------------------------------------------
# Candidate specifications
# ----------------------------------------------------------------------------
# norm      : 'pernode' (F.normalize each gradient per node, released style) or
#             'none'    (raw gradients; only the final spherical projection)
# residual  : 'signed' (released inner product) or 'product' (Algorithm 1)
# mu_channel: expose log1p(mu) as an explicit lift-input column
CANDIDATES = {
    'C0': dict(residual='signed',  norm='pernode', mu_channel=False,
               label='baseline_optgnn_style'),
    'C1': dict(residual='product', norm='none',    mu_channel=False,
               label='algo1_faithful'),
    'C2': dict(residual='product', norm='none',    mu_channel=True,
               label='algo1_plus_channel'),
    'C3': dict(residual='signed',  norm='none',    mu_channel=False,
               label='code_residual_algo1_scale'),
}
CANDIDATE_ORDER = ['C0', 'C1', 'C2', 'C3']


def spec_of(name):
    if name not in CANDIDATES:
        raise KeyError(f'unknown candidate {name}; have {list(CANDIDATES)}')
    return CANDIDATES[name]


# ----------------------------------------------------------------------------
# Residuals
# ----------------------------------------------------------------------------
def edge_residual(x, batch, residual):
    """Per-(directed)-edge constraint residual phi_ij.

    ``x`` is [N, r].  ``edge_index`` is symmetric, so each undirected edge
    appears twice; the caller halves the penalty sum and leaves the node
    scatter un-halved (that scatter already equals sum over neighbours).
    """
    ei = batch.edge_index
    if ei.numel() == 0:
        return x.new_zeros(0)
    if residual == 'product':
        x0 = x[:, 0]
        return (1.0 - x0[ei[0]]) * (1.0 - x0[ei[1]])
    elif residual == 'signed':
        e1 = torch.zeros_like(x)
        e1[:, 0] = 1.0
        xm = x - e1
        return (xm[ei[0]] * xm[ei[1]]).sum(dim=1)
    raise ValueError(residual)


def node_violation(x, batch, residual):
    """c_i = sum_{j in N(i)} phi_ij (symmetric index => lambda_ij=mu_i+mu_j)."""
    ei = batch.edge_index
    phi = edge_residual(x, batch, residual)
    if ei.numel() == 0:
        return x.new_zeros(x.size(0))
    return scatter_add(phi, ei[0], dim=0, dim_size=x.size(0))


# ----------------------------------------------------------------------------
# Grad layer: g_obj = grad(f + (rho/2) sum_undirected phi^2)
#             g_dual = grad(<mu, c>)
# ----------------------------------------------------------------------------
class FairGradLayer(nn.Module):
    def __init__(self, problem, residual='signed', create_graph=True):
        super().__init__()
        self.problem = problem
        self.residual = residual
        self.rho = 0.25
        self.create_graph = create_graph

    def set_rho(self, rho):
        self.rho = float(rho)

    def forward(self, x, batch, mu=None):
        with torch.enable_grad():
            if not x.requires_grad:
                x.requires_grad_(True)
            phi = edge_residual(x, batch, self.residual)
            if phi.numel() > 0:
                # 0.5 * sum_directed phi^2 == sum_undirected phi^2
                pen = 0.5 * (phi ** 2).sum()
            else:
                pen = x.sum() * 0.0

            # self.rho is stored in the un-halved convention; product residuals
            # are unchanged, so renormalise by 4/PHI2 to preserve behaviour.
            loss_obj = self.problem.objective(x, batch) + \
                (self.rho * 4.0 / _PHI2[self.residual] / 2.0) * pen
            g_obj = torch.autograd.grad(
                loss_obj, x, create_graph=self.create_graph,
                retain_graph=True)[0]

            c = node_violation(x, batch, self.residual)
            if mu is None:
                g_dual = torch.zeros_like(x)
            else:
                # dual term is linear in c: /PHI preserves <mu,c> exactly.
                loss_dual = (mu * c).sum() / _PHI[self.residual]
                g_dual = torch.autograd.grad(
                    loss_dual, x, create_graph=self.create_graph)[0]
            return g_obj, g_dual, c


# ----------------------------------------------------------------------------
# Lift layer: h = W [x || obj_ch || dual_ch (|| log1p(mu))]; V = h/||h||
# ----------------------------------------------------------------------------
class FairLiftLayer(nn.Module):
    def __init__(self, in_channels, lift_ratio=0.1, norm='none',
                 mu_channel=False):
        super().__init__()
        self.norm = norm
        self.lift_ratio = lift_ratio
        self.mu_channel = mu_channel
        extra = 1 if mu_channel else 0
        self.lin = nn.Linear(3 * in_channels + extra, in_channels)

    def forward(self, x, g_obj, g_dual, mu=None):
        if self.norm == 'pernode':
            obj_ch = F.normalize(g_obj, dim=1, eps=EPS)
            dual_ch = self.lift_ratio * F.normalize(g_dual, dim=1, eps=EPS)
        else:
            obj_ch = g_obj
            dual_ch = g_dual
        parts = [x, obj_ch, dual_ch]
        if self.mu_channel:
            if mu is None:
                mch = torch.zeros(x.size(0), 1, device=x.device)
            else:
                mch = torch.log1p(mu.clamp(min=0.0)).unsqueeze(1)
            parts.append(mch)
        h = self.lin(torch.cat(parts, dim=1))
        return F.normalize(h, dim=1, eps=EPS)


class FairALONGNN(nn.Module):
    """L identical lift layers; exposes the last node violation c."""

    def __init__(self, grad_layer, in_channels, num_layers=16,
                 lift_ratio=0.1, norm='none', mu_channel=False):
        super().__init__()
        self.grad_layer = grad_layer
        self.layers = []
        for i in range(num_layers):
            layer = FairLiftLayer(in_channels, lift_ratio=lift_ratio,
                                  norm=norm, mu_channel=mu_channel)
            self.add_module(f'layer_{i}', layer)
            self.layers.append(layer)
        self.last_c = None

    def forward(self, x, batch, mu=None):
        for layer in self.layers:
            g_obj, g_dual, c = self.grad_layer(x, batch, mu)
            x = layer(x, g_obj, g_dual, mu=mu)
        self.last_c = c
        return x


def build_network(candidate, problem, rank, num_layers, lift_ratio,
                  create_graph=True):
    spec = spec_of(candidate)
    grad = FairGradLayer(problem, residual=spec['residual'],
                         create_graph=create_graph)
    net = FairALONGNN(grad, in_channels=rank, num_layers=num_layers,
                     lift_ratio=lift_ratio, norm=spec['norm'],
                     mu_channel=spec['mu_channel'])
    return net


# ----------------------------------------------------------------------------
# AL training loss (used to train theta from scratch)
# ----------------------------------------------------------------------------
def al_loss(x_out, batch, mu_batch, problem, residual, penalty=1.0):
    """obj + penalty * sum_undirected phi^2 + <mu, c>.

    ``mu_batch`` is the per-batch-node dual [N]; a zero vector disables the
    dual term.  This is the released AL objective with the candidate's phi and
    the mathematically exact lambda_ij = mu_i + mu_j (no extra 1/2).
    """
    phi = edge_residual(x_out, batch, residual)
    pen = 0.5 * (phi ** 2).sum() if phi.numel() > 0 else x_out.sum() * 0.0
    obj = problem.objective(x_out, batch)
    # penalty stored in the un-halved convention; renormalise for 'product'.
    loss = obj + penalty * 4.0 / _PHI2[residual] * pen
    if mu_batch is not None:
        c = node_violation(x_out, batch, residual)
        loss = loss + (mu_batch * c).sum() / _PHI[residual]
    return loss


# ----------------------------------------------------------------------------
# ALON-A wrapper: amortized dual init + K explicit ascent steps
# ----------------------------------------------------------------------------
def alg1_rho_update(c, c_prev, rho_t, delta=0.25, tau=2.0, rho_max=None,
                    adaptive=True):
    c_norm = float(c.norm())
    if not adaptive:
        return rho_t, c_norm
    if c_prev is not None and c_norm > delta * c_prev:
        rho_next = tau * rho_t
        if rho_max is not None:
            rho_next = min(rho_next, rho_max)
    else:
        rho_next = rho_t
    return rho_next, c_norm


class ALONA_Fair(nn.Module):
    """Wraps a FairALONGNN with an amortized dual initializer and explicit
    primal-dual ascent.

        x0   = net(x_in, mu=0)
        mu0  = softplus(mu_phi([x0, struct]))          # amortized init
        x,c  = net(x_in, mu0)
        for t=1..K:  mu = relu(mu + rho_t * c)         # Alg-1 line 7
                     x,c = net(x_in, mu); rho_t adaptive
    """

    def __init__(self, net, problem, rank, hidden=32, num_dual_layers=2,
                 struct_dim=2):
        super().__init__()
        self.net = net
        self.problem = problem
        self.rank = rank
        # residual mode governs the coefficient renormalisation below
        self.residual = getattr(getattr(net, 'grad_layer', None),
                                'residual', 'signed')
        self.mu_phi = DualInitializer(rank + struct_dim, hidden, num_dual_layers)

    def net_forward(self, x_in, batch, mu):
        x = self.net(x_in, batch, mu)
        return x, self.net.last_c

    @torch.no_grad()
    def init_mu(self, x_in, batch):
        zero = torch.zeros(batch.num_nodes, device=x_in.device)
        x0, _ = self.net_forward(x_in, batch, zero)
        feat = torch.cat([x0, structural_features(batch)], dim=1)
        mu0 = F.softplus(self.mu_phi(feat, batch.edge_index))
        return mu0, x0

    def forward(self, x_in, batch, K=0, rho=0.1, use_mu_phi=True,
                adaptive=True, delta=0.25, tau=2.0, rho_max=None,
                return_info=False):
        zero = torch.zeros(batch.num_nodes, device=x_in.device)
        x0, _ = self.net_forward(x_in, batch, zero)
        if use_mu_phi:
            feat = torch.cat([x0.detach(), structural_features(batch)], dim=1)
            mu = F.softplus(self.mu_phi(feat, batch.edge_index))
        else:
            mu = zero

        x, c = self.net_forward(x_in, batch, mu)
        mu_norm_hist = [float(mu.norm())]
        resid_hist = [float(c.norm()) / _PHI[self.residual]]
        rho_hist = []
        # rho is stored in the un-halved convention; product residuals are
        # unchanged, so 2/PHI restores the historical step scale.
        rho_t = float(rho) * 2.0 / _PHI[self.residual]
        c_prev = float(c.norm()) if K > 0 else None
        for _ in range(K):
            mu = torch.clamp(mu + rho_t * c.detach(), min=0.0)
            x, c = self.net_forward(x_in, batch, mu)
            rho_t, c_prev = alg1_rho_update(c, c_prev, rho_t, delta=delta,
                                            tau=tau, rho_max=rho_max,
                                            adaptive=adaptive)
            mu_norm_hist.append(float(mu.norm()))
            resid_hist.append(float(c.norm()) / _PHI[self.residual])
            rho_hist.append(rho_t)
        if return_info:
            info = {
                'mu': mu.detach(), 'mu_norm_history': mu_norm_hist,
                'resid_history': resid_hist, 'rho_history': rho_hist,
                'mu_mean': float(mu.mean()) if mu.numel() else 0.0,
                'mu_max': float(mu.max()) if mu.numel() else 0.0,
                'mu_norm': float(mu.norm()),
            }
            return x, info
        return x

    @torch.no_grad()
    def dual_targets(self, x_in, batch, K, rho, adaptive=True, delta=0.25,
                     tau=2.0, rho_max=None):
        """No-amortization ascent from mu=0 -> (features, target_mu)."""
        zero = torch.zeros(batch.num_nodes, device=x_in.device)
        x0, _ = self.net_forward(x_in, batch, zero)
        mu = torch.zeros(batch.num_nodes, device=x_in.device)
        x, c = self.net_forward(x_in, batch, mu)
        rho_t = float(rho) * 2.0 / _PHI[self.residual]
        c_prev = float(c.norm()) if K > 0 else None
        for _ in range(K):
            mu = torch.clamp(mu + rho_t * c.detach(), min=0.0)
            x, c = self.net_forward(x_in, batch, mu)
            rho_t, c_prev = alg1_rho_update(c, c_prev, rho_t, delta=delta,
                                            tau=tau, rho_max=rho_max,
                                            adaptive=adaptive)
        feat = torch.cat([x0, structural_features(batch)], dim=1)
        return feat, mu


# ----------------------------------------------------------------------------
# Batched test-time helpers (layers are 1-hop local => exact in one batch)
# ----------------------------------------------------------------------------
def make_batch_graphs(graphs, rank, device, seed=0):
    """One PyG Data containing all ``(n, edges, idx)`` graphs."""
    from torch_geometric.data import Data
    ei = []
    x_parts = []
    off = 0
    offsets = []
    for n, edges, _idx in graphs:
        offsets.append(off)
        for u, v in edges:
            ei.append([off + u, off + v])
            ei.append([off + v, off + u])
        x_parts.append(featurize(n, rank, device, seed=seed))
        off += n
    if ei:
        ei = torch.tensor(ei, dtype=torch.long).t().contiguous().to(device)
    else:
        ei = torch.zeros((2, 0), dtype=torch.long, device=device)
    x_in = torch.cat(x_parts, dim=0) if x_parts else torch.zeros(
        (0, rank), device=device)
    data = Data(edge_index=ei, num_nodes=off,
                node_global_idx=torch.arange(off, device=device), penalty=1.0)
    return x_in, data, offsets


@torch.no_grad()
def round_repair_batch(V, graphs, offsets, seed=0, n_hyperplanes=1000,
                       penalty=1.0, verify=True):
    """Vectorized EXACT equivalent of model.alon_a.round_repair per graph.

    round_repair resets the generator to ``seed`` for every graph, and the
    hyperplane matrix only depends on rank, so H is identical across graphs; we
    therefore draw H once and score every graph in one matmul.  The uncovered
    term equals sum over undirected edges of A_u*A_v, and greedy repair follows
    the same edge order / degree tie-break as round_repair.
    """
    N, r = V.shape
    gen = torch.Generator().manual_seed(seed)
    H = torch.randn(n_hyperplanes, r, generator=gen).to(V.device)
    H = H / H.norm(dim=1, keepdim=True)
    proj = torch.sign(H @ V.t())                 # [H, N]
    A = (1.0 - proj) / 2.0
    rows = []
    for i, (n, edges, idx) in enumerate(graphs):
        s = offsets[i]
        e = s + n
        Ho = proj[:, s:e]
        sz = (Ho > 0).sum(dim=1).float()
        if edges:
            eu = torch.tensor([a for a, b in edges], device=V.device)
            ev = torch.tensor([b for a, b in edges], device=V.device)
            Ac = A[:, s:e]
            unc = (Ac[:, eu] * Ac[:, ev]).sum(dim=1)
        else:
            unc = torch.zeros(n_hyperplanes, device=V.device)
        score = -(sz + penalty * 4.0 * unc)   # fixed rounding tradeoff (unchanged)
        best = int(torch.argmax(score))
        xs = Ho[best].long().tolist()
        xs = [1 if v == 0 else v for v in xs]
        cover = set(j for j, v in enumerate(xs) if v > 0)
        raw_size = len(cover)
        deg = [0] * n
        for u, v in edges:
            deg[u] += 1
            deg[v] += 1
        raw_uncovered = sum(1 for u, v in edges
                            if u not in cover and v not in cover)
        cset = set(cover)
        for u, v in edges:
            if u not in cset and v not in cset:
                cset.add(u if deg[u] >= deg[v] else v)
        row = {
            'idx': int(idx), 'n': int(n), 'm': len(edges),
            'raw_size': raw_size, 'repaired_size': len(cset),
            'infeasible_raw': int(raw_uncovered > 0),
            'raw_uncovered': raw_uncovered,
            'repair_added': len(cset) - raw_size,
        }
        if verify:
            # Every reported cover MUST be feasible.
            viol = cover_constraint(n, edges, cset)
            row['violation_after_repair'] = int(viol)
            assert viol == 0, f'graph {idx}: repair left {viol} uncovered edges'
        rows.append(row)
    return rows


@torch.no_grad()
def eval_config(alon, x_in, big, graphs, offsets, K, rho, use_mu_phi=True,
                adaptive=True, delta=0.25, tau=2.0, rho_max=None,
                n_hyperplanes=1000, statusquo=False):
    """Run one (K, rho) config and return (rows, mu_norm_traj, resid_traj)."""
    if statusquo:
        zero = torch.zeros(big.num_nodes, device=x_in.device)
        x, c = alon.net_forward(x_in, big, zero)
        mu_norm_traj, resid_traj = [0.0], [float(c.norm())]
        mu = None
    else:
        x, info = alon(x_in, big, K=K, rho=rho, use_mu_phi=use_mu_phi,
                        adaptive=adaptive, delta=delta, tau=tau,
                        rho_max=rho_max, return_info=True)
        mu_norm_traj = info['mu_norm_history']
        resid_traj = info['resid_history']
        mu = info['mu']

    rows = round_repair_batch(x, graphs, offsets, seed=0,
                              n_hyperplanes=n_hyperplanes)
    for i, row in enumerate(rows):
        s = offsets[i]
        row['mu_norm'] = (float(mu[s:s + row['n']].norm())
                          if mu is not None else 0.0)
    return rows, mu_norm_traj, resid_traj


def summarize(name, rows, K, rho, mu_norm_traj=None, resid_traj=None,
              extra=None):
    sizes = np.array([r['repaired_size'] for r in rows], dtype=float)
    raw = np.array([r['raw_size'] for r in rows], dtype=float)
    out = {
        'config': name, 'K': K, 'rho': rho, 'n_graphs': len(rows),
        'mean_repaired': float(sizes.mean()),
        'median_repaired': float(np.median(sizes)),
        'std_repaired': float(sizes.std()),
        'mean_raw': float(raw.mean()),
        'frac_infeasible_raw': float(np.mean(
            [r['infeasible_raw'] for r in rows])),
        'mean_raw_uncovered': float(np.mean(
            [r['raw_uncovered'] for r in rows])),
        'max_violation_after_repair': int(max(
            r.get('violation_after_repair', 0) for r in rows)) if rows else 0,
        'per_graph': rows,
    }
    if mu_norm_traj is not None:
        out['mu_norm_traj'] = [float(v) for v in mu_norm_traj]
    if resid_traj is not None:
        out['resid_traj'] = [float(v) for v in resid_traj]
    if extra:
        out.update(extra)
    return out
