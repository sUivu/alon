# Models
import matplotlib.pyplot as plt

import numpy as np
import time
import os

import torch
import torch.nn.functional as F
from torch.nn import Linear, ModuleList, Parameter, Sequential
from torch.optim import Adam

from torch_geometric.nn import MessagePassing
from torch_geometric.nn.models import GAT, GIN, GCN
from torch_geometric.nn.conv import GatedGraphConv

from model.more_models import NegationGAT
from model.saving import load_model
from utils.parsing import read_params_from_folder
from problem.problems import get_problem
from torch_scatter import scatter_add

def construct_model(args, lambda_param):
    if args.model_type == 'LiftMP':
        model = LiftNetwork(
          grad_layer=AutogradLayer(loss_fn=get_problem(args).loss),
          in_channels=args.rank,
          num_layers=args.num_layers,
          repeat_lift_layers=args.repeat_lift_layers,
        )
    elif args.model_type == "ALMGNN":
        model = ALMGNN_Network(
          grad_layer=DecoupledAutogradLayer(problem=get_problem(args)),
          in_channels=args.rank,
          num_layers=args.num_layers,
          lift_ratio=args.lift_ratio,
        )
    elif args.model_type == "ALMGNN_V2":
        model = ALMGNN_Network(
          grad_layer=DecoupledAutogradLayerV2(problem=get_problem(args)),
          in_channels=args.rank,
          num_layers=args.num_layers,
          lift_ratio=args.lift_ratio,
        )
    elif args.model_type == 'FullMP':
        model = LiftProjectNetwork(
          grad_layer=AutogradLayer(loss_fn=get_problem(args).loss),
          in_channels=args.rank,
          num_layers_lift=args.num_layers - args.num_layers_project,
          num_layers_project=args.num_layers_project,
          repeat_lift_layers=args.repeat_lift_layers,
        )
    elif args.model_type == "ProjectMP":
        # must have lift network to train.
        assert args.lift_file is not None
        model = LiftProjectNetwork(
          grad_layer=AutogradLayer(loss_fn=get_problem(args)),
          in_channels=args.rank,
          num_layers_lift=args.num_layers - args.num_layers_project,
          num_layers_project=args.num_layers_project,
          lift_file=args.lift_file,
          repeat_lift_layers=args.repeat_lift_layers,
        )

    elif args.model_type == 'GIN':
        model = GINLiftNetwork(args)
    elif args.model_type == 'GAT':
        model = GATLiftNetwork(args)
    elif args.model_type == 'GCNN':
        model = GCNLiftNetwork(args)
    elif args.model_type == 'GatedGCNN':
        model = GatedGCNLiftNetwork(args)
    elif args.model_type == 'NegationGAT':
        model = NegationGAT(in_channels=args.rank, 
                            hidden_channels=args.hidden_channels, 
                            dropout=args.dropout, 
                            v2=True, norm=args.norm, 
                            num_layers=args.num_layers)

    # 定义基础 GNN
    elif args.model_type == 'ALMGAT':
        grad_layer = DecoupledAutogradLayer(problem=get_problem(args))
        base = GAT(in_channels=args.rank, hidden_channels=args.hidden_channels, 
                   num_layers=args.num_layers, v2=True)
        model = ALM_Hybrid_Network(base, grad_layer, args.rank, lift_ratio=args.lift_ratio)
    elif args.model_type == 'ALMGIN':
        grad_layer = DecoupledAutogradLayer(problem=get_problem(args))
        base = GIN(in_channels=args.rank, hidden_channels=args.hidden_channels, 
                   num_layers=args.num_layers)
        model = ALM_Hybrid_Network(base, grad_layer, args.rank, lift_ratio=args.lift_ratio)
    elif args.model_type == 'ALMGCNN':
        grad_layer = DecoupledAutogradLayer(problem=get_problem(args))
        base = GCN(in_channels=args.rank, hidden_channels=args.hidden_channels, 
                   num_layers=args.num_layers)
        model = ALM_Hybrid_Network(base, grad_layer, args.rank, lift_ratio=args.lift_ratio)
    elif args.model_type == 'ALMGatedGCNN':
        grad_layer = DecoupledAutogradLayer(problem=get_problem(args))
        model = ALMGatedGCNN_Network(grad_layer, rank=args.rank, hidden_channels=args.hidden_channels, num_layers=16, lift_ratio=args.lift_ratio)
#   NOTE: GatedGCNLiftNetwork.__init__() does not accept 'hidden_channels';
#   ALMGatedGCNN instead builds on ALMGatedGCNN_Network above.
        # self.net = GatedGraphConv(out_channels=args.hidden_channels,
        #     num_layers=args.num_layers)
    elif args.model_type == 'Nikos':
        model = LiftProjectNetwork_Nikos(
          grad_layer=construct_grad_layer(args),
          in_channels=args.rank,
          num_layers_lift=args.num_layers - args.num_layers_project,
          num_layers_project=args.num_layers_project,
        )
    else:
        raise ValueError(f'Got unexpected model_type {args.model_type}')

    if args.finetune_from is not None:
        # load in model for finetuning
        model = load_model(model, args.finetune_from)
        model.to(args.device)

    opt = Adam([{"params": model.parameters(), "lr": args.lr}])

    return model, opt

# 1. 改进 AutogradLayer：实现梯度解耦（原始版本）
# g_obj = ∇[f + ρ·c²]   (目标 + 二次惩罚)
# g_dual = ∇[μ·c]       (对偶项)
class DecoupledAutogradLayer(torch.nn.Module):
    def __init__(self, problem):
        super().__init__()
        self.problem = problem

    def forward(self, x, batch, mu_param=None):
        with torch.enable_grad():
            x.requires_grad_(True)
            # 获取节点维度的违反值
            raw_edge_penalties_val = self.problem.violations(x, batch)
            node_penalties = scatter_add(raw_edge_penalties_val, batch.edge_index[0], dim=0, dim_size=x.size(0))
            raw_edge_constraint = raw_edge_penalties_val ** 2
            node_constraint = torch.sum(scatter_add(raw_edge_constraint, batch.edge_index[0], dim=0, dim_size=x.size(0)))
            
            # Primal 目标梯度 (目标 + 二次惩罚)
            # node_constraint is un-halved (4x); batch.penalty is stored /4.
            loss_obj = self.problem.objective(x, batch) + batch.penalty * node_constraint / 2
            g_obj = torch.autograd.grad(loss_obj, x, create_graph=True, retain_graph=True)[0]

            # Dual 梯度 (提取对应节点的 mu 势能)
            # node_penalties is un-halved (2x): /4 keeps mu*c/2 numerically identical.
            mu_batch = mu_param[batch.node_global_idx]
            loss_dual = torch.sum(mu_batch * node_penalties) / 4
            g_dual = torch.autograd.grad(loss_dual, x, create_graph=True)[0]
            
            return g_obj, g_dual

# # 1b. 梯度解耦 V2：目标与约束完全分离
# # g_obj = ∇f               (纯目标)
# # g_dual = ∇[μ·c + ρ·c²]  (对偶 + 二次惩罚合并)
class DecoupledAutogradLayerV2(torch.nn.Module):
    def __init__(self, problem):
        super().__init__()
        self.problem = problem

    def forward(self, x, batch, mu_param=None):
        with torch.enable_grad():
            x.requires_grad_(True)
            # 获取节点维度的违反值
            raw_edge_penalties_val = self.problem.violations(x, batch)
            node_penalties = scatter_add(raw_edge_penalties_val, batch.edge_index[0], dim=0, dim_size=x.size(0))
            raw_edge_constraint = raw_edge_penalties_val ** 2
            node_constraint = torch.sum(scatter_add(raw_edge_constraint, batch.edge_index[0], dim=0, dim_size=x.size(0)))
            
            # 纯目标梯度
            loss_obj = self.problem.objective(x, batch)
            g_obj = torch.autograd.grad(loss_obj, x, create_graph=True, retain_graph=True)[0]

            # 约束梯度：对偶项 + 二次惩罚项合并
            # linear term /4 (c -> 2c), quadratic keeps /2 with penalty stored /4.
            mu_batch = mu_param[batch.node_global_idx]
            loss_dual = (torch.sum(mu_batch * node_penalties) / 4
                         + batch.penalty * node_constraint / 2)
            g_dual = torch.autograd.grad(loss_dual, x, create_graph=True)[0]
            
            return g_obj, g_dual

# 2. 改进 LiftLayer：多方向特征聚合架构
class DecoupledLiftLayer(torch.nn.Module):
    def __init__(self, in_channels, lift_ratio=0.2):
        super().__init__()
        # 输入维度: 原始特征(rank) + 目标梯度(rank) + 对偶梯度(rank) = 3 * rank
        self.lin = Linear(3 * in_channels, in_channels)
        self.lift_ratio = lift_ratio

    def forward(self, x, batch, g_obj, g_dual):
        # 分别归一化，确保特征尺度一致
        norm_obj = F.normalize(g_obj, dim=1)
        norm_dual = self.lift_ratio * F.normalize(g_dual, dim=1)
        
        # 简洁高效的聚合方式
        out = torch.cat((x, norm_obj, norm_dual), 1)
        out = self.lin(out)
        return F.normalize(out, dim=1)

# 3. 改进 LiftNetwork：适配解耦计算
class ALMGNN_Network(torch.nn.Module):
    def __init__(self, grad_layer, in_channels, num_layers=12,lift_ratio=0.2):
        super().__init__()
        self.grad_layer = grad_layer        
        self.layers = [DecoupledLiftLayer(in_channels, lift_ratio=lift_ratio) for _ in range(num_layers)]
        for i, layer in enumerate(self.layers):
            self.add_module(f"layer_{i}", layer)

    def forward(self, x, batch, lambda_param=None):
        for layer in self.layers:
            # 每一层计算解耦梯度，实时引导神经网络在解空间和约束空间中穿梭
            g_obj, g_dual = self.grad_layer(x, batch, lambda_param)
            # g_obj = g_obj[0]   # 提取梯度张量
            # g_dual = g_dual[0] # 提取梯度张量
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

# Nearly identical to the lift layer. The big difference is that we no longer normalize in update.
class ProjectLayer(torch.nn.Module):
    def __init__(self, grad_layer, in_channels):
        super().__init__()
        self.grad_layer = grad_layer
        self.lin = Linear(2*in_channels, in_channels)

    def forward(self, x, batch, lambda_param=None):
        grads = self.grad_layer(x, batch, lambda_param)
        out = torch.cat((x, grads), 1)
        out = self.lin(out)
        out = F.tanh(out)
        return out

class ProjectNetwork(torch.nn.Module):
    def __init__(self, grad_layer, in_channels, num_layers=8):
        super().__init__()
        self.layers = [ProjectLayer(grad_layer, in_channels) for _ in range(num_layers)]
        for i, layer in enumerate(self.layers):
            self.add_module(f"layer_{i}", layer)

    def forward(self, x, batch,lambda_param=None):
        for l in self.layers:
            x = l(x, batch, lambda_param)
        return x

class LiftProjectNetwork(torch.nn.Module):
    def __init__(self, in_channels, num_layers_lift, num_layers_project, grad_layer, lift_file=None, repeat_lift_layers=None):
        super().__init__()

        if lift_file is not None:
            print(f"loading from {lift_file}")
            # TODO: maybe check the lift arguments for consistency

            # load lift file
            class DotDict(dict):
                def __getattr__(self, key):
                    if key in self:
                        return self[key]
                    else:
                        raise AttributeError(f"'DotDict' object has no attribute '{key}'")
            self.lift_args = DotDict(read_params_from_folder(os.path.dirname(lift_file)))
            self.lift_net, _ = construct_model(self.lift_args)
            if self.lift_args.rank != in_channels:
                raise ValueError(f"As of right now, lift_args rank ({self.lift_args.rank}) must \
                                 equal the project network rank ({in_channels})")
            self.lift_net = load_model(self.lift_net, lift_file)

            # freeze it
            for param in self.lift_net.parameters():
                param.requires_grad = False
            
        else:
            self.lift_net = LiftNetwork(grad_layer, in_channels, num_layers=num_layers_lift, repeat_lift_layers=repeat_lift_layers)
        self.project_net = ProjectNetwork(grad_layer, in_channels, num_layers=num_layers_project)

    def forward(self, x, batch, lambda_param=None):
        out = self.lift_net(x, batch, lambda_param)
        # TODO randomly rotate here
        out = self.project_net(out, batch, lambda_param)
        return out
    
    # def forward(self, x, batch, lambda_param=None):
    #     for l, repeat_l in zip(self.layers, self.repeat_lift_layers):
    #         for _ in range(repeat_l):
    #             x = l(x, batch, lambda_param)
    #     return x
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

class LiftLayerv2(torch.nn.Module):
    def __init__(self, grad_layer, in_channels):
        super().__init__()
        self.grad_layer = grad_layer
        self.lin = Linear(2*in_channels, in_channels)

    def forward(self, x, batch):
        grads = self.grad_layer(x, batch)
        out = torch.cat((x, grads), 1)
        out = self.lin(out)
        out = F.normalize(out, dim=1)
        return out

class ProjectNetwork_r1(torch.nn.Module):
    def __init__(self, grad_layer, in_channels, num_layers=6):
        super().__init__()
        self.layers = [LiftLayerv2(grad_layer, in_channels) for _ in range(num_layers)]
        for i, layer in enumerate(self.layers):
            self.add_module(f"layer_{i}", layer)
        self.lin = Linear(in_channels, 1)

    def forward(self, x, batch):
        for l in self.layers:
            x = F.leaky_relu(l(x, batch)+x,0.01)
        x = F.normalize(x)
        return x

class LiftProjectNetwork_Nikos(torch.nn.Module):
    def __init__(self, in_channels, num_layers_lift, num_layers_project, grad_layer, lift_file=None):
        super().__init__()
        self.lift_net = LiftNetwork(grad_layer, in_channels, num_layers=num_layers_lift)
        self.project_net = ProjectNetwork_r1(grad_layer, in_channels, num_layers=num_layers_project)

    def forward(self, x, batch):
        out = self.lift_net(x, batch)
        outs = self.project_net(out, batch)
        return outs


# [MODIFIED] 通用 ALM 混合架构：将解耦梯度注入标准 GNN
class ALM_Hybrid_Network(torch.nn.Module):
    def __init__(self, base_gnn, grad_layer, in_channels, lift_ratio=0.2):
        super().__init__()
        self.base_gnn = base_gnn  # 传入 GAT, GIN 等实例
        self.grad_layer = grad_layer
        self.lift_ratio = lift_ratio
        
        # 核心：为了兼容 ALM 组件，我们需要一个映射层将 [x, g_obj, g_dual] 
        # 映射回 GNN 期望的 hidden_channels。
        # 这里假设 base_gnn 的第一层输入维度是 rank
        self.feature_fuse = torch.nn.Linear(3 * in_channels, in_channels)

    def forward(self, x, batch, lambda_param=None):
        # 模仿 OptGNN 的多步迭代过程
        # 如果 base_gnn 本身是多层（如 GAT(num_layers=12)），我们可以在整体外部迭代
        # 或者在每一层内部嵌入。为了严格对齐 ALM 逻辑，我们采取逐层注入策略。
        
        # 注意：由于标准 PyG 模型（GAT/GIN）是一次性执行所有层，
        # 为了实现“每一层都根据当前 x 计算梯度”，我们必须拆解层执行。
        
        current_x = x
        # 遍历 base_gnn 的内部层（以 GAT 为例，通常在 self.base_gnn.convs 中）
        # 如果是 PyG 模型，我们直接循环执行其子模块
        for i in range(len(self.base_gnn.convs)):
            # 1. 计算当前解空间的解耦梯度
            g_obj, g_dual = self.grad_layer(current_x, batch, lambda_param)
            
            # 2. 特征融合：注入 ALM 指令
            norm_obj = F.normalize(g_obj, dim=1)
            norm_dual = self.lift_ratio * F.normalize(g_dual, dim=1)
            fused_input = torch.cat((current_x, norm_obj, norm_dual), dim=1)
            current_x = self.feature_fuse(fused_input)
            
            # 3. 拓扑聚合：执行标准 GNN 的一层
            # 这里需要处理不同 GNN 的 forward 签名
            if isinstance(self.base_gnn, GatedGraphConv):
                current_x = self.base_gnn.convs[i](current_x, batch.edge_index)
            else:
                current_x = self.base_gnn.convs[i](current_x, batch.edge_index)
            
            # 4. 激活与归一化（对齐论文中的 unit sphere 约束 [cite: 1855, 1943]）
            current_x = F.leaky_relu(current_x, 0.2)
            current_x = F.normalize(current_x, dim=1)
            
        return current_x

# [MODIFIED] ALM 门控图卷积网络实现
class ALMGatedGCNN_Network(torch.nn.Module):
    def __init__(self, grad_layer, rank, hidden_channels, num_layers=16, lift_ratio=0.2):
        super().__init__()
        self.grad_layer = grad_layer
        self.num_layers = num_layers
        self.lift_ratio = lift_ratio
        
        # 1. 初始投影：将 rank 映射到隐藏层维度（如果不同）
        self.lin_in = Linear(rank, hidden_channels)
        
        # 2. 核心门控层：将 num_layers 设为 1，由我们手动控制循环
        from torch_geometric.nn.conv import GatedGraphConv
        self.rnn_conv = GatedGraphConv(out_channels=hidden_channels, num_layers=1)
        
        # 3. ALM 融合层：接收 [x, g_obj, g_dual] 并映射回 hidden_channels
        # 注意：这里输入是 3 * hidden_channels 因为梯度也是在 hidden 空间计算的
        self.feature_fuse = Linear(3 * hidden_channels, hidden_channels)

    def forward(self, x, batch, lambda_param=None):
        # 初始投影
        current_x = self.lin_in(x)
        current_x = F.normalize(current_x, dim=1)

        for _ in range(self.num_layers):
            # [PHASE 2 对齐] 在每一步循环内重新计算梯度
            g_obj, g_dual = self.grad_layer(current_x, batch, lambda_param)
            
            # 归一化与融合
            norm_obj = F.normalize(g_obj, dim=1)
            norm_dual = self.lift_ratio * F.normalize(g_dual, dim=1)
            
            # 注入 ALM 组件
            fused = torch.cat((current_x, norm_obj, norm_dual), dim=1)
            current_x = self.feature_fuse(fused)
            
            # 执行一步门控更新
            current_x = self.rnn_conv(current_x, batch.edge_index)
            
            # 保持单位球约束 [cite: 112, 156]
            current_x = F.normalize(current_x, dim=1)
            
        return current_x
# class OPT_Network(torch.nn.Module):
#     def __init__(self, base_gnn, grad_layer, in_channels, lift_ratio=0.2):
#         super().__init__()
#         self.base_gnn = base_gnn  # 传入 GAT, GIN 等实例
#         self.grad_layer = grad_layer
#         self.lift_ratio = lift_ratio
        
#         # 核心：为了兼容 ALM 组件，我们需要一个映射层将 [x, g_obj, g_dual] 
#         # 映射回 GNN 期望的 hidden_channels。
#         # 这里假设 base_gnn 的第一层输入维度是 rank
#         self.feature_fuse = torch.nn.Linear(2 * in_channels, in_channels)

#     def forward(self, x, batch, lambda_param=None):
#         # 模仿 OptGNN 的多步迭代过程
#         # 如果 base_gnn 本身是多层（如 GAT(num_layers=12)），我们可以在整体外部迭代
#         # 或者在每一层内部嵌入。为了严格对齐 ALM 逻辑，我们采取逐层注入策略。
        
#         # 注意：由于标准 PyG 模型（GAT/GIN）是一次性执行所有层，
#         # 为了实现“每一层都根据当前 x 计算梯度”，我们必须拆解层执行。
        
#         current_x = x
#         # 遍历 base_gnn 的内部层（以 GAT 为例，通常在 self.base_gnn.convs 中）
#         # 如果是 PyG 模型，我们直接循环执行其子模块
#         for i in range(len(self.base_gnn.convs)):
#             # 1. 计算当前解空间的解耦梯度
#             g_obj, g_dual = self.grad_layer(current_x, batch, lambda_param)
#             g_obj = g_obj[0]
#             g_dual = g_dual[0]
            
#             # 2. 特征融合：注入 ALM 指令
#             norm_obj = F.normalize(g_obj, dim=1)
#             # norm_dual = self.lift_ratio * F.normalize(g_dual, dim=1)
#             # fused_input = torch.cat((current_x, norm_obj, norm_dual), dim=1)
#             fused_input = torch.cat((current_x, norm_obj), dim=1)
#             current_x = self.feature_fuse(fused_input)
            
#             # 3. 拓扑聚合：执行标准 GNN 的一层
#             # 这里需要处理不同 GNN 的 forward 签名
#             if isinstance(self.base_gnn, GatedGraphConv):
#                 current_x = self.base_gnn.convs[i](current_x, batch.edge_index)
#             else:
#                 current_x = self.base_gnn.convs[i](current_x, batch.edge_index)
            
#             # 4. 激活与归一化（对齐论文中的 unit sphere 约束 [cite: 1855, 1943]）
#             current_x = F.leaky_relu(current_x, 0.2)
#             current_x = F.normalize(current_x, dim=1)
            
#         return current_x
