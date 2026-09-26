# Models
import matplotlib.pyplot as plt

import numpy as np
import time
import torch
import torch.nn.functional as F
from torch.nn import Linear, ModuleList, Parameter, Sequential
from torch.optim import Adam

from torch_geometric.nn import MessagePassing
from torch_geometric.nn.models import GAT, GIN, GCN
from torch_geometric.nn.conv import GatedGraphConv

from problem.problems import get_problem
from torch_scatter import scatter_add

def construct_model(args, lambda_param):
    if args.model_type == 'LiftMP':
        # LiftMP (OptGNN baseline): no dual variables; uses the fused penalty
        # loss from model/baselines.py (imported here to avoid a cycle).
        from model.baselines import build_liftmp_baseline
        model = build_liftmp_baseline(args)
    elif args.model_type == "ALONGNN":
        model = ALONGNN_Network(
          grad_layer=DecoupledAutogradLayer(problem=get_problem(args)),
          in_channels=args.rank,
          num_layers=args.num_layers,
          lift_ratio=args.lift_ratio,
        )
    elif args.model_type == "ALONGNN_V2":
        model = ALONGNN_Network(
          grad_layer=DecoupledAutogradLayerV2(problem=get_problem(args)),
          in_channels=args.rank,
          num_layers=args.num_layers,
          lift_ratio=args.lift_ratio,
        )
    elif args.model_type == 'GIN':
        model = GINLiftNetwork(args)
    elif args.model_type == 'GAT':
        model = GATLiftNetwork(args)
    elif args.model_type == 'GCNN':
        model = GCNLiftNetwork(args)
    elif args.model_type == 'GatedGCNN':
        model = GatedGCNLiftNetwork(args)

    # define the base GNN
    elif args.model_type == 'ALONGAT':
        grad_layer = DecoupledAutogradLayer(problem=get_problem(args))
        base = GAT(in_channels=args.rank, hidden_channels=args.hidden_channels, 
                   num_layers=args.num_layers, v2=True)
        model = ALON_Hybrid_Network(base, grad_layer, args.rank, lift_ratio=args.lift_ratio)
    elif args.model_type == 'ALONGIN':
        grad_layer = DecoupledAutogradLayer(problem=get_problem(args))
        base = GIN(in_channels=args.rank, hidden_channels=args.hidden_channels, 
                   num_layers=args.num_layers)
        model = ALON_Hybrid_Network(base, grad_layer, args.rank, lift_ratio=args.lift_ratio)
    elif args.model_type == 'ALONGCNN':
        grad_layer = DecoupledAutogradLayer(problem=get_problem(args))
        base = GCN(in_channels=args.rank, hidden_channels=args.hidden_channels, 
                   num_layers=args.num_layers)
        model = ALON_Hybrid_Network(base, grad_layer, args.rank, lift_ratio=args.lift_ratio)
    elif args.model_type == 'ALONGatedGCNN':
        grad_layer = DecoupledAutogradLayer(problem=get_problem(args))
        model = ALONGatedGCNN_Network(grad_layer, rank=args.rank, hidden_channels=args.hidden_channels, num_layers=16, lift_ratio=args.lift_ratio)
#   NOTE: GatedGCNLiftNetwork.__init__() does not accept 'hidden_channels';
#   ALONGatedGCNN instead builds on ALONGatedGCNN_Network above.
        # self.net = GatedGraphConv(out_channels=args.hidden_channels,
        #     num_layers=args.num_layers)
    else:
        raise ValueError(f'Got unexpected model_type {args.model_type}')

    if args.finetune_from is not None:
        # load in model for finetuning
        model = load_model(model, args.finetune_from)
        model.to(args.device)

    opt = Adam([{"params": model.parameters(), "lr": args.lr}])

    return model, opt

# 1. Improved AutogradLayer: gradient decoupling (original version)
# g_obj = ∇[f + ρ·c²]   (objective + quadratic penalty)
# g_dual = ∇[μ·c]       (dual term)
class DecoupledAutogradLayer(torch.nn.Module):
    def __init__(self, problem):
        super().__init__()
        self.problem = problem

    def forward(self, x, batch, mu_param=None):
        with torch.enable_grad():
            x.requires_grad_(True)
            # get node-dimension violation values
            raw_edge_penalties_val = self.problem.violations(x, batch)
            node_penalties = scatter_add(raw_edge_penalties_val, batch.edge_index[0], dim=0, dim_size=x.size(0))
            raw_edge_constraint = raw_edge_penalties_val ** 2
            node_constraint = torch.sum(scatter_add(raw_edge_constraint, batch.edge_index[0], dim=0, dim_size=x.size(0)))

            # Primal objective gradient (objective + quadratic penalty)
            # node_constraint is un-halved (4x); batch.penalty is stored /4.
            loss_obj = self.problem.objective(x, batch) + batch.penalty * node_constraint / 2
            g_obj = torch.autograd.grad(loss_obj, x, create_graph=True, retain_graph=True)[0]

            # Dual gradient (extract the mu potential of the corresponding nodes)
            # node_penalties is un-halved (2x): /4 keeps mu*c/2 numerically identical.
            mu_batch = mu_param[batch.node_global_idx]
            loss_dual = torch.sum(mu_batch * node_penalties) / 4
            g_dual = torch.autograd.grad(loss_dual, x, create_graph=True)[0]
            
            return g_obj, g_dual

# # 1b. Gradient decoupling V2: objective and constraint fully separated
# # g_obj = ∇f               (pure objective)
# # g_dual = ∇[μ·c + ρ·c²]  (dual + quadratic penalty merged)
class DecoupledAutogradLayerV2(torch.nn.Module):
    def __init__(self, problem):
        super().__init__()
        self.problem = problem

    def forward(self, x, batch, mu_param=None):
        with torch.enable_grad():
            x.requires_grad_(True)
            # get node-dimension violation values
            raw_edge_penalties_val = self.problem.violations(x, batch)
            node_penalties = scatter_add(raw_edge_penalties_val, batch.edge_index[0], dim=0, dim_size=x.size(0))
            raw_edge_constraint = raw_edge_penalties_val ** 2
            node_constraint = torch.sum(scatter_add(raw_edge_constraint, batch.edge_index[0], dim=0, dim_size=x.size(0)))

            # pure objective gradient
            loss_obj = self.problem.objective(x, batch)
            g_obj = torch.autograd.grad(loss_obj, x, create_graph=True, retain_graph=True)[0]

            # constraint gradient: dual term + quadratic penalty merged
            # linear term /4 (c -> 2c), quadratic keeps /2 with penalty stored /4.
            mu_batch = mu_param[batch.node_global_idx]
            loss_dual = (torch.sum(mu_batch * node_penalties) / 4
                         + batch.penalty * node_constraint / 2)
            g_dual = torch.autograd.grad(loss_dual, x, create_graph=True)[0]
            
            return g_obj, g_dual

# 2. Improved LiftLayer: multi-directional feature aggregation architecture
class DecoupledLiftLayer(torch.nn.Module):
    def __init__(self, in_channels, lift_ratio=0.2):
        super().__init__()
        # input dim: raw features (rank) + objective gradient (rank) + dual gradient (rank) = 3 * rank
        self.lin = Linear(3 * in_channels, in_channels)
        self.lift_ratio = lift_ratio

    def forward(self, x, batch, g_obj, g_dual):
        # normalize separately to keep feature scales consistent
        norm_obj = F.normalize(g_obj, dim=1)
        norm_dual = self.lift_ratio * F.normalize(g_dual, dim=1)
        
        # simple and efficient aggregation
        out = torch.cat((x, norm_obj, norm_dual), 1)
        out = self.lin(out)
        return F.normalize(out, dim=1)

# 3. Improved LiftNetwork: adapted for decoupled computation
class ALONGNN_Network(torch.nn.Module):
    def __init__(self, grad_layer, in_channels, num_layers=12,lift_ratio=0.2):
        super().__init__()
        self.grad_layer = grad_layer        
        self.layers = [DecoupledLiftLayer(in_channels, lift_ratio=lift_ratio) for _ in range(num_layers)]
        for i, layer in enumerate(self.layers):
            self.add_module(f"layer_{i}", layer)

    def forward(self, x, batch, lambda_param=None):
        for layer in self.layers:
            # each layer computes decoupled gradients, guiding the network in
            # real time between solution space and constraint space
            g_obj, g_dual = self.grad_layer(x, batch, lambda_param)
            # g_obj = g_obj[0]   # extract the gradient tensor
            # g_dual = g_dual[0] # extract the gradient tensor
            x = layer(x, batch, g_obj, g_dual)
        return x

# use autograd on a given loss function to compute gradients
class AutogradLayer(torch.nn.Module):
    def __init__(self, loss_fn):
        super().__init__()
        self._loss_fn = loss_fn

    def forward(self, x, batch, lambda_param=None):
        # calculate the lift loss and take the gradient w.r.t. the input x
        # the result is expected to be autodiffable, and training should be unaffected
        with torch.enable_grad():
            x.requires_grad_(True)
            loss = self._loss_fn(x, batch, lambda_param)
            grad = torch.autograd.grad(loss, x, create_graph=True)[0]
            return grad

class LiftLayer(torch.nn.Module):
    def __init__(self, grad_layer, in_channels):
        super().__init__()
        self.grad_layer = grad_layer
        self.lin = Linear(2*in_channels, in_channels)

    def forward(self, x, batch, lambda_param=None):
        grads = self.grad_layer(x, batch, lambda_param)
        norm_grads = F.normalize(grads, dim=1)
        out = torch.cat((x, norm_grads), 1)
        out = self.lin(out)
        out = F.normalize(out, dim=1)
        return out

class LiftNetwork(torch.nn.Module):
    def __init__(self, grad_layer, in_channels, num_layers=12, repeat_lift_layers=None):
        super().__init__()
        if repeat_lift_layers is not None:
            # the number of layers must equal the length of the repeat array.
            assert(len(repeat_lift_layers) == num_layers)
        self.layers = [LiftLayer(grad_layer, in_channels) for _ in range(num_layers)]
        for i, layer in enumerate(self.layers):
            self.add_module(f"layer_{i}", layer)
        
        if repeat_lift_layers is None:
            repeat_lift_layers = [1 for _ in range(num_layers)]
        self.repeat_lift_layers = repeat_lift_layers

    def forward(self, x, batch, lambda_param=None):
        for l, repeat_l in zip(self.layers, self.repeat_lift_layers):
            for _ in range(repeat_l):
                x = l(x, batch, lambda_param)
        return x

# graph isomorphism network
# TODO version with gradients, version allowing negation of neighbors
class GINLiftNetwork(torch.nn.Module):
    def __init__(self, args):
        super().__init__()
        self.net = GIN(in_channels=args.rank,
            hidden_channels=args.hidden_channels,
            dropout=args.dropout,
            norm=args.norm,
            num_layers=args.num_layers)

    def forward(self, x, batch):
        out = self.net(x=x, edge_index=batch.edge_index)
        out = F.normalize(out, dim=1)
        return out

# graph attention network
# TODO version with gradients, version allowing negation of neighbors
class GATLiftNetwork(torch.nn.Module):
    def __init__(self, args):
        super().__init__()
        self.net = GAT(in_channels=args.rank,
            hidden_channels=args.hidden_channels,
            dropout=args.dropout,
            v2=True,
            norm=args.norm,
            num_layers=args.num_layers,
            heads=args.heads)

    def forward(self, x, batch):
        out = self.net(x=x, edge_index=batch.edge_index)
        out = F.normalize(out, dim=1)
        return out

# graph convnet
# TODO version with gradients, version allowing negation of neighbors
class GCNLiftNetwork(torch.nn.Module):
    def __init__(self, args):
        super().__init__()
        self.net = GCN(in_channels=args.rank,
            hidden_channels=args.hidden_channels,
            dropout=args.dropout,
            norm=args.norm,
            num_layers=args.num_layers)

    def forward(self, x, batch):
        out = self.net(x=x, edge_index=batch.edge_index)
        out = F.normalize(out, dim=1)
        return out

# gated graph convnet
# TODO version with gradients, version allowing negation of neighbors
class GatedGCNLiftNetwork(torch.nn.Module):
    def __init__(self, args):
        super().__init__()
        # TODO - does this need more params? is out_channel correct?
        self.net = GatedGraphConv(out_channels=args.hidden_channels,
            num_layers=args.num_layers)

    def forward(self, x, batch):
        out = self.net(x=x, edge_index=batch.edge_index)
        out = F.normalize(out, dim=1)
        return out


# [MODIFIED] generic ALM hybrid architecture: inject decoupled gradients into a standard GNN
class ALON_Hybrid_Network(torch.nn.Module):
    def __init__(self, base_gnn, grad_layer, in_channels, lift_ratio=0.2):
        super().__init__()
        self.base_gnn = base_gnn  # a GAT, GIN, etc. instance
        self.grad_layer = grad_layer
        self.lift_ratio = lift_ratio

        # core: to be compatible with the ALM components, we need a mapping
        # layer that maps [x, g_obj, g_dual] back to the hidden_channels the
        # GNN expects.
        # here we assume the first layer of base_gnn takes input dim rank
        self.feature_fuse = torch.nn.Linear(3 * in_channels, in_channels)

    def forward(self, x, batch, lambda_param=None):
        # mimic OptGNN's multi-step iteration
        # if base_gnn is itself multi-layer (e.g. GAT(num_layers=12)), we can
        # iterate externally as a whole, or embed inside each layer. To align
        # strictly with ALM logic, we adopt a layer-by-layer injection strategy.

        # note: since standard PyG models (GAT/GIN) execute all layers in one
        # call, to achieve "compute gradients w.r.t. current x at every layer"
        # we must decompose the layer execution.

        current_x = x
        # iterate over base_gnn's internal layers (for GAT, usually in self.base_gnn.convs)
        # if it is a PyG model, we directly loop over its submodules
        for i in range(len(self.base_gnn.convs)):
            # 1. compute decoupled gradients for the current solution space
            g_obj, g_dual = self.grad_layer(current_x, batch, lambda_param)

            # 2. feature fusion: inject the ALM signal
            norm_obj = F.normalize(g_obj, dim=1)
            norm_dual = self.lift_ratio * F.normalize(g_dual, dim=1)
            fused_input = torch.cat((current_x, norm_obj, norm_dual), dim=1)
            current_x = self.feature_fuse(fused_input)

            # 3. topological aggregation: run one layer of the standard GNN
            # need to handle different GNN forward signatures
            if isinstance(self.base_gnn, GatedGraphConv):
                current_x = self.base_gnn.convs[i](current_x, batch.edge_index)
            else:
                current_x = self.base_gnn.convs[i](current_x, batch.edge_index)

            # 4. activation and normalization (aligning with the unit sphere constraint in the paper [cite: 1855, 1943])
            current_x = F.leaky_relu(current_x, 0.2)
            current_x = F.normalize(current_x, dim=1)
            
        return current_x

# [MODIFIED] ALM gated graph convolution network implementation
class ALONGatedGCNN_Network(torch.nn.Module):
    def __init__(self, grad_layer, rank, hidden_channels, num_layers=16, lift_ratio=0.2):
        super().__init__()
        self.grad_layer = grad_layer
        self.num_layers = num_layers
        self.lift_ratio = lift_ratio

        # 1. initial projection: map rank to the hidden dim (if different)
        self.lin_in = Linear(rank, hidden_channels)

        # 2. core gated layer: set num_layers to 1; we control the loop manually
        from torch_geometric.nn.conv import GatedGraphConv
        self.rnn_conv = GatedGraphConv(out_channels=hidden_channels, num_layers=1)

        # 3. ALM fusion layer: takes [x, g_obj, g_dual] and maps back to hidden_channels
        # note: input here is 3 * hidden_channels because gradients are also computed in hidden space
        self.feature_fuse = Linear(3 * hidden_channels, hidden_channels)

    def forward(self, x, batch, lambda_param=None):
        # initial projection
        current_x = self.lin_in(x)
        current_x = F.normalize(current_x, dim=1)

        for _ in range(self.num_layers):
            # [PHASE 2 alignment] recompute gradients within each loop step
            g_obj, g_dual = self.grad_layer(current_x, batch, lambda_param)

            # normalize and fuse
            norm_obj = F.normalize(g_obj, dim=1)
            norm_dual = self.lift_ratio * F.normalize(g_dual, dim=1)

            # inject the ALM components
            fused = torch.cat((current_x, norm_obj, norm_dual), dim=1)
            current_x = self.feature_fuse(fused)

            # run one gated update step
            current_x = self.rnn_conv(current_x, batch.edge_index)

            # maintain the unit sphere constraint [cite: 112, 156]
            current_x = F.normalize(current_x, dim=1)

        return current_x
