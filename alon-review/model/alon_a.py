#!/usr/bin/env python3
"""
ALON-A: Amortized-Warmstart ALON for Minimum Vertex Cover.

NON-INVASIVE addition to the ALON codebase.  Nothing in the existing files
(model/models.py, model/training.py, problem/*.py, data/loader.py, train*.py)
is modified: this module only IMPORTS and reuses them.

Motivation
----------
The existing ALONGNN (ALON) keeps a per-node free dual ``mu_param`` of length
``total_num_nodes`` that is indexed by node identity (``batch.node_global_idx``)
and updated ONLY for training-split nodes (model/training.py:185,260).  At
evaluation test/OOD nodes have ``mu = 0``, so ``g_dual = 0`` and the
advertised primal-dual mechanism is dead at inference.

ALON-A fixes this by REPLACING the free per-node dual with:

    mu_i = softplus( mu_phi( node_features ) )        (amortized initializer)

followed by an EXPLICIT primal-dual ascent at inference (and optionally in
training):

    x      <- ALON_theta( x_in, graph, mu )           # frozen base network
    c_i    <- sum_{j in N(i)} phi_ij(x)              # node residual (edge -> src)
    mu_i   <- max(0, mu_i + rho * c_i)               # dual ascent

The paper's node-wise reparameterization ``lambda_ij = mu_i + mu_j`` is
PRESERVED exactly: the base network consumes ``mu`` through the untouched
``DecoupledAutogradLayer`` (model/models.py:131), whose
``loss_dual = sum(mu_batch * node_penalties)/2`` and
``scatter_add`` aggregation reproduce lambda_ij = mu_i + mu_j without ever
materializing per-edge duals.  Only *how mu is produced/refined* changes.

Author: ALON-A proof-of-concept.
"""

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric.data import Data, Batch
from torch_geometric.nn import GCNConv
from torch_scatter import scatter_add

# ---- reuse existing code by importing only (no edits) -----------------------
from model.models import ALONGNN_Network, DecoupledAutogradLayer  # noqa: F401
from problem.problems import get_problem  # noqa: F401
from problem.losses import get_vertex_cover_violations  # noqa: F401
from data.loader import construct_dataset, prepare_alon_dataset  # noqa: F401

PROJECT_ROOT = Path(__file__).resolve().parent.parent


# ============================================================================
# Structural node features used by the amortized dual initializer
# ============================================================================
def structural_features(batch):
    """Degree-based, graph-size-normalized node features [N, 2].

    Works for any batch (uses batch.edge_index which is already local).  These
    features are available for unseen nodes/graphs, which is what makes the
    dual head amortized rather than identity-bound.
    """
    n = batch.num_nodes
    dev = batch.edge_index.device
    deg = torch.zeros(n, device=dev)
    if batch.edge_index.numel() > 0:
        deg = scatter_add(torch.ones(batch.edge_index.shape[1], device=dev),
                          batch.edge_index[0], dim=0, dim_size=n)
    max_deg = deg.max().clamp(min=1.0)
    deg_norm = deg / max_deg
    log_deg = torch.log1p(deg) / math.log1p(float(max_deg))
    return torch.stack([deg_norm, log_deg], dim=1)


# ============================================================================
# 1) Amortized dual initializer  mu_phi : features -> mu_i >= 0
# ============================================================================
class DualInitializer(nn.Module):
    """Small message-passing network mapping node features to mu_i >= 0.

    Input  per node: [embedding (rank), structural features (2)]
    Output per node: scalar mu_i = softplus(MLP(...))  >= 0
    A GNN (rather than a per-node MLP) is used so the map generalizes to
    unseen graph sizes / topologies -- the point of amortization.
    """

    def __init__(self, in_channels, hidden=32, num_layers=2):
        super().__init__()
        self.lin_in = nn.Linear(in_channels, hidden)
        self.convs = nn.ModuleList(
            [GCNConv(hidden, hidden) for _ in range(num_layers)])
        self.lin_out = nn.Linear(hidden, 1)

    def forward(self, feat, edge_index):
        h = F.relu(self.lin_in(feat))
        for conv in self.convs:
            h = F.relu(conv(h, edge_index))
        return F.softplus(self.lin_out(h)).squeeze(-1)


# ============================================================================
# 2) ALON-A wrapper: amortized init + explicit primal-dual ascent
# ============================================================================
class ALONA(nn.Module):
    """Wraps a frozen/live ALONGNN and adds an amortized dual + ascent loop.

    Forward equations (as implemented):

        x0        = ALON_theta(x_in, G, mu=0)                  # dual-free embedding
        mu0       = softplus(mu_phi([x0, struct_feats]))      # amortized init, >=0
        x         = ALON_theta(x_in, G, mu0)                   # warm-started primal
        for k in 1..K:                                        # explicit dual ascent
            c_i   = sum_{j in N(i)} phi_ij(x)                 # node residual
            mu_i  = max(0, mu_i + rho * c_i)
            x     = ALON_theta(x_in, G, mu)                    # re-solve primal

    ``lambda_ij = mu_i + mu_j`` is realized inside the untouched
    ``DecoupledAutogradLayer`` via scatter of the node residual, never as an
    explicit per-edge dual tensor.
    """

    def __init__(self, base_model, problem, rank, hidden=32, num_dual_layers=2,
                 struct_dim=2):
        super().__init__()
        self.base = base_model
        self.problem = problem
        self.rank = rank
        self.mu_phi = DualInitializer(rank + struct_dim, hidden, num_dual_layers)

    # -- helpers -----------------------------------------------------------
    def _base_forward(self, x_in, batch, mu_nodes):
        """Call the untouched base network with a per-node mu vector.

        The base indexes mu by ``batch.node_global_idx``; we temporarily set
        that to a local arange so no global-size buffer is needed.  Only the
        DecoupledAutogradLayer reads it, so this is behaviour-preserving.
        """
        saved = getattr(batch, 'node_global_idx', None)
        batch.node_global_idx = torch.arange(batch.num_nodes, device=x_in.device)
        try:
            out = self.base(x_in, batch, mu_nodes)
        finally:
            if saved is not None:
                batch.node_global_idx = saved
        return out

    def node_residual(self, x, batch):
        """Node-level constraint residual c_i, identical aggregation to
        model/training.py:185 (scatter edge phi onto the source node)."""
        raw = self.problem.violations(x, batch)  # edge-level phi_ij
        return scatter_add(raw, batch.edge_index[0], dim=0,
                           dim_size=x.size(0))

    @torch.no_grad()
    def init_mu(self, x_in, batch):
        """Amortized dual initialization. Returns (mu0, x0)."""
        zero = torch.zeros(batch.num_nodes, device=x_in.device)
        x0 = self._base_forward(x_in, batch, zero)
        feat = torch.cat([x0, structural_features(batch)], dim=1)
        mu0 = F.softplus(self.mu_phi(feat, batch.edge_index))
        return mu0, x0

    # -- main forward ------------------------------------------------------
    def forward(self, x_in, batch, K=0, rho=0.1, detach_dual=True,
                return_info=False):
        """Run amortized init + K explicit dual-ascent steps.

        K=0  -> amortized-init-only (no ascent).
        """
        zero = torch.zeros(batch.num_nodes, device=x_in.device)
        x0 = self._base_forward(x_in, batch, zero)
        feat = torch.cat([x0.detach() if detach_dual else x0,
                          structural_features(batch)], dim=1)
        mu = F.softplus(self.mu_phi(feat, batch.edge_index))
        x = self._base_forward(x_in, batch, mu)

        mus = [mu.detach().clone()]
        resid_norms = []
        for _ in range(K):
            c = self.node_residual(x, batch)
            mu = torch.clamp(mu + rho * c, min=0.0)
            x = self._base_forward(x_in, batch, mu)
            mus.append(mu.detach().clone())
            resid_norms.append(float(c.abs().mean()))

        if return_info:
            info = {
                'mu': mu.detach(),
                'mu_history': mus,
                'mu_mean': float(mu.mean()),
                'mu_max': float(mu.max()) if mu.numel() else 0.0,
                'mu_norm': float(mu.norm()),
                'resid_mean': resid_norms,
            }
            return x, info
        return x

    def dual_targets(self, x_in, batch, K, rho):
        """Collect supervised targets for mu_phi by running NO-GRAD ascent
        from a zero init.  Returns (features, target_mu) detached."""
        with torch.no_grad():
            zero = torch.zeros(batch.num_nodes, device=x_in.device)
            x0 = self._base_forward(x_in, batch, zero)
            mu = torch.zeros(batch.num_nodes, device=x_in.device)
            x = x0
            for _ in range(K):
                c = self.node_residual(x, batch)
                mu = torch.clamp(mu + rho * c, min=0.0)
                x = self._base_forward(x_in, batch, mu)
            feat = torch.cat([x0, structural_features(batch)], dim=1)
        return feat, mu


# ============================================================================
# Rounding + greedy repair (self-contained; does NOT depend on the
# concurrently-edited scripts/classical_baselines.py)
# ============================================================================
@torch.no_grad()
def round_repair(V, n, edges, seed=0, n_hyperplanes=1000, penalty=1.0):
    """Randomized-hyperplane rounding (same protocol as ALON's
    random_hyperplane_projector: sign(H @ V^T), best score, zeros -> 1),
    then GREEDY REPAIR of any uncovered edge.

    Returns dict with:
      raw_size        : cover size of the un-repaired rounding
      repaired_size   : cover size after greedy repair (ALWAYS feasible)
      infeasible_raw  : bool, raw rounding left >=1 edge uncovered
      raw_uncovered   : # uncovered edges before repair
      repair_added    : # vertices added by repair
      best_hyp        : winning hyperplane index
    """
    V = V.detach().cpu().float()
    gen = torch.Generator().manual_seed(seed)
    H = torch.randn(n_hyperplanes, V.shape[1], generator=gen)
    H = H / H.norm(dim=1, keepdim=True)
    proj = torch.sign(H @ V.t())            # (H, N), +/-1
    sizes = (proj > 0).sum(dim=1).float()   # objective for +/-1 assignments

    A = (1.0 - proj) / 2.0                  # A[i,u]=1 iff u NOT in cover i
    M = torch.zeros(n, n)
    for u, v in edges:
        M[u, v] = 1.0
        M[v, u] = 1.0
    uncovered = torch.einsum('in,nm,im->i', A, M, A) / 2.0  # counted twice
    score = -(sizes + penalty * 4.0 * uncovered)            # fixed rounding tradeoff (unchanged)
    best = int(torch.argmax(score))

    xs = proj[best].long().tolist()
    xs = [1 if v == 0 else v for v in xs]
    cover = set(i for i, v in enumerate(xs) if v > 0)
    raw_size = len(cover)

    deg = [0] * n
    for u, v in edges:
        deg[u] += 1
        deg[v] += 1

    raw_uncovered = 0
    for u, v in edges:
        if u not in cover and v not in cover:
            raw_uncovered += 1
    cset = set(cover)
    for u, v in edges:
        if u not in cset and v not in cset:
            cset.add(u if deg[u] >= deg[v] else v)
    return {
        'raw_size': raw_size,
        'repaired_size': len(cset),
        'infeasible_raw': raw_uncovered > 0,
        'raw_uncovered': raw_uncovered,
        'repair_added': len(cset) - raw_size,
        'best_hyp': best,
    }


def cover_constraint(n, edges, cover):
    """# uncovered edges for an integer cover set (0 == feasible)."""
    c = set(cover)
    return sum(1 for u, v in edges if u not in c and v not in c)


# ============================================================================
# Self-contained dataset split (same protocol as data.loader.construct_loaders:
# train_fraction, split_seed, random_split -> test = last 10%)
# ============================================================================
def _graph_edges(g):
    ei = g.edge_index
    s = set()
    for u, v in ei.t().tolist():
        if u != v:
            s.add((min(u, v), max(u, v)))
    return sorted(s)


def build_split_graphs(args, split='test', limit=0):
    """Return list of (n, edges, dataset_idx) using the paper's split protocol."""
    dataset = construct_dataset(args)
    dataset, total_nodes = prepare_alon_dataset(dataset)
    n = len(dataset)
    train_size = int(args.train_fraction * n)
    val_size = (n - train_size) // 2
    gen = torch.Generator().manual_seed(args.split_seed)
    perm = torch.randperm(n, generator=gen).tolist()
    if split == 'train':
        idx = perm[:train_size]
    elif split == 'val':
        idx = perm[train_size:train_size + val_size]
    elif split == 'test':
        idx = perm[train_size + val_size:]
    else:
        raise ValueError(split)
    if limit:
        idx = idx[:limit]
    graphs = []
    for i in idx:
        g = dataset[int(i)]
        graphs.append((int(g.num_nodes), _graph_edges(g), int(i)))
    return graphs, total_nodes


def make_batch(graphs, device, pctr=None):
    """Build a single PyG Batch for a list of (n, edges, idx) test graphs."""
    from torch_geometric.data import Data
    ei = []
    off = 0
    for n, edges, _idx in graphs:
        for u, v in edges:
            ei.append([off + u, off + v])
            ei.append([off + v, off + u])
        off += n
    ei = torch.tensor(ei, dtype=torch.long).t().contiguous().to(device)
    data = Data(edge_index=ei, num_nodes=off,
                node_global_idx=torch.arange(off, device=device), penalty=1.0)
    if pctr is not None:
        pctr[0] = off
    return data


def featurize(n, rank, device, seed=0):
    gen = torch.Generator().manual_seed(seed)
    x = torch.randn(n, rank, generator=gen, dtype=torch.float32)
    return F.normalize(x, dim=1).to(device)


def build_base_model(args, device):
    """Construct the untouched ALONGNN base (reusing model.models)."""
    problem = get_problem(args)
    grad_layer = DecoupledAutogradLayer(problem=problem)
    model = ALONGNN_Network(grad_layer=grad_layer, in_channels=args.rank,
                           num_layers=args.num_layers,
                           lift_ratio=args.lift_ratio).to(device)
    return model, problem


def load_base_checkpoint(model, path, device='cpu'):
    sd = torch.load(path, map_location=device, weights_only=False)
    if isinstance(sd, dict) and 'state_dict' in sd:
        sd = sd['state_dict']
    elif isinstance(sd, dict) and 'model_state_dict' in sd:
        sd = sd['model_state_dict']
    model.load_state_dict(sd, strict=False)
    return model
