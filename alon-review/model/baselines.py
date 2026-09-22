# Fused OptGNN/LiftMP baseline.
#
# The OptGNN baseline (Yau et al., "Are Graph Neural Networks Necessary for
# Graph-Tailored Optimization?", the LiftMP / lifted-multihomomorphism
# network) is re-implemented HERE on top of this package's own building
# blocks: the `AutogradLayer`/`LiftLayer`/`LiftNetwork` classes in
# model/models.py and the objective/constraint functions in
# problem/losses.py.  The original public OptGNN implementation is NOT
# required to train or evaluate this baseline.
#
# Numerical equivalence with the reference formulation: the reference code
# defines the quadratic penalty on the *halved* edge residual
# g_ij/2 (values in {0, 2} for 0/1 vertex scores).  This package's
# `vertex_cover_constraint` is the classical un-halved form
# g_ij = <e1 - v_i, e1 - v_j> (values in {0, 4}), so dividing by 4
# reproduces the reference loss exactly:
#     obj + penalty * sum((g_ij/2)^2) == obj + penalty * constraint / 4
# The gradient the network consumes through AutogradLayer is therefore
# bit-identical to the reference OptGNN penalty chain.

import torch

from model.models import AutogradLayer, LiftNetwork
from problem.losses import max_cut_obj, vertex_cover_obj, vertex_cover_constraint


def liftmp_vertex_cover_loss(X, batch, lambda_param=None):
    """OptGNN quadratic-penalty loss for vertex cover.

    `lambda_param` is accepted (and ignored) so the loss can be plugged into
    AutogradLayer: LiftMP carries no dual variables."""
    # un-halved residual (0/4 values); /4 == the reference halved-residual form
    return vertex_cover_obj(X, batch) + batch.penalty * vertex_cover_constraint(X, batch) / 4.


def liftmp_max_cut_loss(X, batch, lambda_param=None):
    """OptGNN loss for max-cut (unconstrained: objective only)."""
    return max_cut_obj(X, batch)


def get_liftmp_loss_fn(args):
    if args.problem_type == 'vertex_cover':
        return liftmp_vertex_cover_loss
    elif args.problem_type == 'max_cut':
        return liftmp_max_cut_loss
    else:
        raise ValueError(f"LiftMP baseline does not support problem_type {args.problem_type}")


def build_liftmp_baseline(args):
    """Construct the LiftMP network with the fused OptGNN loss layer.

    The returned module is a plain `LiftNetwork` (state-dict keys
    `layer_i.lin.*`), so checkpoints trained by train_baseline.py load with
    strict=True under the paper-protocol evaluation scripts."""
    grad_layer = AutogradLayer(loss_fn=get_liftmp_loss_fn(args))
    return LiftNetwork(
        grad_layer=grad_layer,
        in_channels=args.rank,
        num_layers=args.num_layers,
        repeat_lift_layers=getattr(args, 'repeat_lift_layers', None),
    )
