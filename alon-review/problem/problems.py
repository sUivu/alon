from problem.losses import max_cut_obj, vertex_cover_obj, vertex_cover_constraint, get_vertex_cover_violations,get_vertex_cover_violations_all_dims, vertex_cover_constraint_all_dims,vertex_cover_obj_all_dims,vertex_cover_score_all_dims
from problem.losses import max_cut_score, vertex_cover_score, max_clique_score
from networkx.algorithms.approximation import one_exchange, min_weighted_vertex_cover
from problem.baselines import max_cut_sdp, vertex_cover_sdp
from problem.baselines import max_cut_gurobi, vertex_cover_gurobi
import torch
import numpy as np
from torch_scatter import scatter_add

def get_problem(args):
    if args.problem_type == 'max_cut':
        return MaxCutProblem
    elif args.problem_type == 'vertex_cover':
        return VertexCoverProblem
    elif args.problem_type == 'max_clique':
        return MaxCliqueProblem
    else:
        raise ValueError(f"get_problem got invalid problem_type {args.problem_type}")

# Bundle losses, constraints, and utilities for a constrained optimization problem
class OptProblem():
    @staticmethod
    def loss(X, batch):
        raise NotImplementedError()

    @staticmethod
    def objective(X, batch):
        raise NotImplementedError()

    @staticmethod
    def constraint(X, batch):
        raise NotImplementedError()

    @staticmethod
    def loss(X, batch):
        raise NotImplementedError()

    @staticmethod
    def score(args, X, example):
        raise NotImplementedError()

    @staticmethod
    def greedy(G):
        raise NotImplementedError()

    @staticmethod
    def sdp(args, example):
        raise NotImplementedError()

    @staticmethod
    def gurobi(args, example):
        raise NotImplementedError()

class MaxCutProblem(OptProblem):
    @staticmethod
    def objective(X, batch):
        return max_cut_obj(X, batch)

    @staticmethod
    def constraint(X, batch):
        return 0.

    @staticmethod
    def loss(X, batch):
        return max_cut_obj(X, batch)

    @staticmethod
    def score(args, X, example):
        return max_cut_score(args, X, example)

    @staticmethod
    def greedy(G):
        greedy_score, _ = one_exchange(G)
        return greedy_score

    @staticmethod
    def sdp(args, example):
        return max_cut_sdp(args, example)

    @staticmethod
    def gurobi(args, example):
        return max_cut_gurobi(args, example)

class VertexCoverProblem(OptProblem):
    @staticmethod
    def objective(X, batch):
        # return vertex_cover_obj_all_dims(X, batch)
        return vertex_cover_obj(X, batch)
    @staticmethod
    def constraint(X, batch):
        if isinstance(X, np.ndarray):
            X = torch.FloatTensor(X)
        if len(X.shape) == 1:
            X = X[:, None]
        # return vertex_cover_constraint_all_dims(X, batch)
        # constraint() is un-halved (4x); /4 preserves the reported metric value.
        return vertex_cover_constraint(X, batch) / 4.
    
    @staticmethod
    def violations(X, batch):
        return get_vertex_cover_violations(X, batch)

    # @staticmethod
    # def loss(X, batch, lambda_param, return_constraint=False,mode="default"):
    #     constraint_val = get_vertex_cover_violations(X, batch)
    #     lambda_batch = lambda_param[batch.edge_global_idx]
    #     lambda_term = torch.sum(lambda_batch * constraint_val)
    #     penalty_term = batch.penalty * torch.sum(constraint_val ** 2)
    #     obj = vertex_cover_obj(X, batch) 
    #     if return_constraint:
    #         return obj + penalty_term + lambda_term, constraint_val
    #     if mode=="valid":
    #         return obj + penalty_term
    #     return obj + penalty_term + lambda_term
    @staticmethod
    def loss(X, batch, mu_param, return_constraint=False, mode="default"):
        # Get node-level violation values C_i
        raw_edge_penalties_val = get_vertex_cover_violations(X, batch) 
        node_penalties = scatter_add(raw_edge_penalties_val, batch.edge_index[0], dim=0, dim_size=X.size(0))
        
        raw_edge_constraint = raw_edge_penalties_val ** 2
        node_constraint = torch.sum(scatter_add(raw_edge_constraint, batch.edge_index[0], dim=0, dim_size=X.size(0)))
        
        # Get the dual potentials mu_i of nodes in the current batch
        mu_batch = mu_param[batch.node_global_idx] 
        
        # Both ALM penalty terms are now node-dimension dot products
        # residual is un-halved (c -> 2c): /4 keeps mu*c/2 numerically identical.
        lambda_term = torch.sum(mu_batch * node_penalties) / 4
        penalty_term = batch.penalty * node_constraint / 2
        
        obj = vertex_cover_obj(X, batch) 
        
        if return_constraint:
            return obj + penalty_term + lambda_term, node_penalties
        if mode=="valid":
            return obj + penalty_term
        return obj + penalty_term + lambda_term

    @staticmethod
    def score(args, X, example):
        # return vertex_cover_score_all_dims(args, X, example)
        return vertex_cover_score(args, X, example)
    @staticmethod
    def greedy(G):
        cover = min_weighted_vertex_cover(G)
        greedy_score = -len(cover)
        return greedy_score

    @staticmethod
    def sdp(args, example):
        return vertex_cover_sdp(args, example)

    @staticmethod
    def gurobi(args, example):
        return vertex_cover_gurobi(args, example)
