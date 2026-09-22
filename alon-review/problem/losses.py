# Loss functions and their associated gradients

import numpy as np
import time

import torch
import torch.nn.functional as F
from torch_geometric.utils import dense_to_sparse, to_dense_adj, to_torch_csr_tensor, to_torch_coo_tensor
from functools import partial

# X should have shape (N, r)
def max_cut_obj(X, batch):
    # attach edge weights if they're not already present
    if not hasattr(batch, 'edge_weight') or batch.edge_weight is None:
        num_edges = batch.edge_index.shape[1]
        batch.edge_weight = torch.ones(num_edges, device=X.device)

    # compute loss
    X0 = X[batch.edge_index[0]]
    X1 = X[batch.edge_index[1]]
    edges = torch.sum(X0 * X1, dim=1)
    obj = torch.sum(edges * batch.edge_weight)
    return obj

def vertex_cover_obj(X, batch):
    # attach node weights if they're not already present
    N = batch.num_nodes
    if not hasattr(batch, 'node_weight') or batch.node_weight is None:
        batch.node_weight = torch.ones(N, device=X.device)

    # lift adopts e1 = (1,0,...,0) as 1
    # count number of vertices: \sum_{i \in [N]} w_i(1+x_i)/2
    obj = torch.inner(torch.ones(N).to(X.device) + X[:, 0], batch.node_weight) / 2.
    return obj

def vertex_cover_obj_all_dims(X, batch, use_mean=False):
    """
    Objective using all dimensions of X.
    - If use_mean=True, we use mean over rank: s_i = X[i].mean()
      (recommended: keeps scale similar to single-dim)
    - If use_mean=False, we use sum over rank: s_i = X[i].sum()
      (only if you plan to scale penalty/lr by rank)
    """
    N = batch.num_nodes
    if not hasattr(batch, 'node_weight') or batch.node_weight is None:
        batch.node_weight = torch.ones(N, device=X.device)

    if use_mean:
        s = X.mean(dim=1)          # shape [N]
    else:
        s = X.sum(dim=1)           # shape [N]

    # same algebra as before: sum_i w_i * (1 + s_i) / 2
    obj = ((1.0 + s) * batch.node_weight).sum() / 2.0
    return obj

# from torch_scatter import scatter_add

def get_vertex_cover_violations(X, batch):
    """
    Computes node-level violation values (core dimension-reduction logic)
    Returns shape: (num_nodes, )
    """
    e1 = torch.zeros_like(X)
    e1[:, 0] = 1
    Xm = e1 - X  # [N, d]

    # 1. Compute transient edge-level violations (Transient Edge Violation)
    # Note: in undirected graphs, edge_index is usually symmetric, i.e. it
    # contains both (u,v) and (v,u)
    # Classical un-halved residual: g_ij = <e1 - v_i, e1 - v_j>  (values {0,4})
    raw_edge_penalties = torch.sum(Xm[batch.edge_index[0]] * Xm[batch.edge_index[1]], dim=1)

    # 2. Dimension-reduction aggregation: accumulate edge violations onto the
    # nodes by source node (edge_index[0])
    # Because the undirected graph is symmetric, node u receives the (u,v)
    # penalty and node v receives the (v,u) penalty, fairly sharing the penalty


    return raw_edge_penalties  # returns the constraint potential C_i of each node, shape [num_nodes_in_batch]

def get_vertex_cover_violations_all_dims(X, batch, use_mean=False):
    """
    Constraint using all dims:
    phi_ij = 1 - sum(v_i) - sum(v_j) + <v_i, v_j>
    If use_mean=True, sums are means over dimensions (i.e. average).
    Returns scalar constraint = sum(phi_ij^2) over edges.
    """
    N = batch.num_nodes
    # edge_index expected shape [2, E_batch]
    ei0, ei1 = batch.edge_index[0], batch.edge_index[1]

    if use_mean:
        s = X.mean(dim=1)         # shape [N]
    else:
        s = X.sum(dim=1)          # shape [N]

    # dot products between node vectors on each edge
    dot = (X[ei0] * X[ei1]).sum(dim=1)   # shape [E_batch]

    phi = 1.0 - s[ei0] - s[ei1] + dot    # shape [E_batch]

    # penalty: sum of squared violations (same style as original)
    return phi
def vertex_cover_constraint(X, batch):
    N = batch.num_nodes

    # now calculate penalty for uncovered edges
    # phi is matrix of dimension N by N for error per edge
    # phi_ij = 1 - <x_i + x_j,e_1> + <x_i,x_j> for (i,j) \in Edges
    # phi_ij = <x_i - e1, x_j - e1> for (i, j) \in Edges
    e1 = torch.zeros_like(X) # (num_edges, hidden)
    e1[:, 0] = 1
    Xm = X - e1

    penalties = torch.sum(Xm[batch.edge_index[0]] * Xm[batch.edge_index[1]], dim=1)
    constraint = torch.sum(penalties * penalties)
    return constraint

def vertex_cover_constraint_all_dims(X, batch, use_mean=False):
    """
    Constraint using all dims:
    phi_ij = 1 - sum(v_i) - sum(v_j) + <v_i, v_j>
    If use_mean=True, sums are means over dimensions (i.e. average).
    Returns scalar constraint = sum(phi_ij^2) over edges.
    """
    N = batch.num_nodes
    # edge_index expected shape [2, E_batch]
    ei0, ei1 = batch.edge_index[0], batch.edge_index[1]

    if use_mean:
        s = X.mean(dim=1)         # shape [N]
    else:
        s = X.sum(dim=1)          # shape [N]

    # dot products between node vectors on each edge
    dot = (X[ei0] * X[ei1]).sum(dim=1)   # shape [E_batch]

    phi = 1.0 - s[ei0] - s[ei1] + dot    # shape [E_batch]

    # penalty: sum of squared violations (same style as original)
    constraint = (phi ** 2).sum()
    return constraint

# we are receiving the _complement_ of the target graph
# TODO fix this
def max_clique_loss(X, batch, penalty=2):
    return vertex_cover_loss(X, batch, penalty=penalty)

def max_cut_score(args, X, example):
    # convert numpy array to torch tensor
    if isinstance(X, np.ndarray):
        X = torch.FloatTensor(X)
    if len(X.shape) == 1:
        X = X[:, None]
    N = example.num_nodes
    E = example.edge_index.shape[1]
    return (E - max_cut_obj(X, example)) / 2.

def vertex_cover_score(args, X, example):
    # convert numpy array to torch tensor
    if isinstance(X, np.ndarray):
        X = torch.FloatTensor(X)
    if len(X.shape) == 1:
        X = X[:, None]
    # constraint() is un-halved (residual^2, 4x the old value); /4 keeps the
    # reported score numerically identical to the pre-conversion convention.
    return - (vertex_cover_obj(X, example) + vertex_cover_constraint(X, example) / 4.)

def vertex_cover_score_all_dims(args, X, example):
    # convert numpy array to torch tensor
    if isinstance(X, np.ndarray):
        X = torch.FloatTensor(X)
    if len(X.shape) == 1:
        X = X[:, None]
    return - (vertex_cover_obj_all_dims(X, example) + vertex_cover_constraint_all_dims(X, example))

# we are receiving the _complement_ of the target graph
# the score is N - k where k is the vertex cover size
def max_clique_score(args, X, example):
    if isinstance(X, np.ndarray):
        X = torch.FloatTensor(X)
    if len(X.shape) == 1:
        X = X[:, None]
    N = example.num_nodes
    return N - (vertex_cover_obj(X, example) + vertex_cover_constraint(X, batch) / 4.)
