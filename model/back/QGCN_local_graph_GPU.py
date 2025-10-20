# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from torch_geometric.nn import MessagePassing
# from torch_geometric.nn import global_mean_pool
# from complexPyTorch.complexLayers import ComplexLinear
# from complexPyTorch.complexFunctions import complex_relu, complex_dropout
# from torch_sparse import SparseTensor
# import math
#
#
# class OptimizedComplexUnitaryEvolutionGCNConv(MessagePassing):
#     """GPU优化的复数酉演化GCN卷积层"""
#
#     def __init__(self, in_channels, out_channels,
#                  evolution_time=1.0,
#                  normalize=True,
#                  bias=True,
#                  dropout=0.0,
#                  activation='complex_relu',
#                  max_matrix_exp_terms=10):
#         super(OptimizedComplexUnitaryEvolutionGCNConv, self).__init__(aggr='add')
#
#         self.in_channels = in_channels
#         self.out_channels = out_channels
#         self.evolution_time = evolution_time
#         self.normalize = normalize
#         self.dropout = dropout
#         self.activation = activation
#         self.max_matrix_exp_terms = max_matrix_exp_terms
#
#         # 复数特征变换层
#         self.complex_lin = ComplexLinear(in_channels, out_channels)
#
#         # 可学习的演化时间参数
#         self.time_param_real = nn.Parameter(torch.tensor(evolution_time, dtype=torch.float32))
#         self.time_param_imag = nn.Parameter(torch.zeros(1, dtype=torch.float32))
#
#         # 额外的复数偏置
#         self.use_extra_bias = bias
#         if self.use_extra_bias:
#             self.extra_bias_real = nn.Parameter(torch.Tensor(out_channels))
#             self.extra_bias_imag = nn.Parameter(torch.Tensor(out_channels))
#         else:
#             self.register_parameter('extra_bias_real', None)
#             self.register_parameter('extra_bias_imag', None)
#
#         self.reset_parameters()
#
#     def reset_parameters(self):
#         if self.extra_bias_real is not None:
#             nn.init.zeros_(self.extra_bias_real)
#             nn.init.zeros_(self.extra_bias_imag)
#
#     @property
#     def complex_time_param(self):
#         """获取复数时间参数"""
#         return torch.complex(self.time_param_real, self.time_param_imag)
#
#     @property
#     def complex_extra_bias(self):
#         """获取额外的复数偏置"""
#         if self.extra_bias_real is not None:
#             return torch.complex(self.extra_bias_real, self.extra_bias_imag)
#         return None
#
#     def matrix_exp_taylor(self, A, max_terms=10):
#         """使用泰勒级数在GPU上计算矩阵指数"""
#         device = A.device
#         dtype = A.dtype
#
#         # 初始化结果矩阵
#         result = torch.eye(A.shape[-1], device=device, dtype=dtype).expand_as(A)
#         term = torch.eye(A.shape[-1], device=device, dtype=dtype).expand_as(A)
#
#         # 泰勒级数展开
#         for i in range(1, max_terms + 1):
#             term = torch.matmul(term, A) / i
#             result = result + term
#
#         return result
#
#     def build_adjacency_matrices_batch(self, edge_index, num_nodes, x):
#         """批量构建所有节点的邻接矩阵"""
#         device = x.device
#         dtype = torch.complex64
#
#         # 创建完整的邻接矩阵
#         adj_matrix = torch.zeros(num_nodes, num_nodes, device=device, dtype=dtype)
#
#         # 填充邻接矩阵
#         adj_matrix[edge_index[0], edge_index[1]] = 1.0 + 0.0j
#
#         # 为每个节点创建包含自身和邻居的子图邻接矩阵
#         # 这里我们使用一个简化的方法：直接使用度数作为子图大小
#         degrees = torch.zeros(num_nodes, device=device, dtype=torch.long)
#         degrees = degrees.scatter_add(0, edge_index[0], torch.ones_like(edge_index[0]))
#         degrees = degrees.scatter_add(0, edge_index[1], torch.ones_like(edge_index[1]))
#
#         # 限制最大邻居数以控制计算复杂度
#         max_neighbors = min(10, degrees.max().item())
#
#         # 创建批量邻接矩阵 (num_nodes, max_size, max_size)
#         max_size = max_neighbors + 1  # +1 for the node itself
#         batch_adj = torch.zeros(num_nodes, max_size, max_size, device=device, dtype=dtype)
#
#         # 为每个节点构建子图
#         for node_idx in range(num_nodes):
#             # 找到邻居
#             neighbors = []
#             mask = (edge_index[0] == node_idx) | (edge_index[1] == node_idx)
#             if mask.any():
#                 node_edges = edge_index[:, mask]
#                 for src, dst in node_edges.t():
#                     if src == node_idx:
#                         neighbors.append(dst.item())
#                     else:
#                         neighbors.append(src.item())
#
#             # 去重并限制数量
#             neighbors = list(set(neighbors))[:max_neighbors - 1]
#             subgraph_nodes = [node_idx] + neighbors
#
#             # 构建子图邻接矩阵
#             for i, node_i in enumerate(subgraph_nodes):
#                 for j, node_j in enumerate(subgraph_nodes):
#                     if adj_matrix[node_i, node_j] != 0:
#                         batch_adj[node_idx, i, j] = 1.0 + 0.0j
#
#         return batch_adj
#
#     def batch_unitary_evolution(self, batch_adj, complex_time):
#         """批量计算酉演化矩阵"""
#         device = batch_adj.device
#         num_nodes, max_size, _ = batch_adj.shape
#
#         # 计算 -i * A * t
#         evolution_arg = -1j * batch_adj * complex_time
#
#         # 使用泰勒级数计算矩阵指数
#         G_t = self.matrix_exp_taylor(evolution_arg, self.max_matrix_exp_terms)
#
#         # 计算酉演化矩阵的简化版本
#         # 这里我们直接使用G_t的归一化版本
#         # 在实际应用中，可以根据需要进行更复杂的酉化处理
#         norms = torch.norm(G_t, dim=(-2, -1), keepdim=True)
#         G_t_normalized = G_t / (norms + 1e-8)
#
#         return G_t_normalized
#
#     def batch_evolve_features(self, x, batch_adj, complex_time):
#         """批量演化节点特征"""
#         device = x.device
#         num_nodes, feature_dim = x.shape
#         max_size = batch_adj.shape[1]
#
#         # 计算演化矩阵
#         U = self.batch_unitary_evolution(batch_adj, complex_time)
#
#         # 准备特征矩阵
#         # 为每个节点准备其子图特征
#         batch_features = torch.zeros(num_nodes, max_size, feature_dim,
#                                      device=device, dtype=x.dtype)
#
#         # 简化版本：每个节点的第一个位置是自己，其他位置填充邻居特征的平均值
#         batch_features[:, 0, :] = x
#
#         # 对于其他位置，我们使用邻域特征的聚合
#         for node_idx in range(num_nodes):
#             # 找到邻居并聚合特征
#             neighbors = []
#             edge_mask = (batch_adj[node_idx, 0, 1:] != 0)
#             if edge_mask.any():
#                 # 这里简化处理，使用平均池化
#                 neighbor_features = x.mean(dim=0, keepdim=True)
#                 for i in range(1, min(max_size, edge_mask.sum().item() + 1)):
#                     batch_features[node_idx, i, :] = neighbor_features
#
#         # 应用演化矩阵
#         # U: (num_nodes, max_size, max_size)
#         # batch_features: (num_nodes, max_size, feature_dim)
#         evolved_features = torch.matmul(U, batch_features)
#
#         # 返回每个节点演化后的特征（取第一个位置）
#         return evolved_features[:, 0, :]
#
#     def complex_activation(self, x):
#         """应用复数激活函数"""
#         if self.activation == 'complex_relu':
#             return complex_relu(x)
#         elif self.activation == 'none':
#             return x
#         else:
#             return complex_relu(x)
#
#     def prepare_complex_input(self, x):
#         """将实数输入转换为复数形式"""
#         if x.dtype in [torch.complex64, torch.complex128]:
#             return x
#         else:
#             return torch.complex(x, torch.zeros_like(x))
#
#     def forward(self, x, edge_index):
#         num_nodes = x.size(0)
#
#         # 确保输入是复数形式
#         x_complex = self.prepare_complex_input(x)
#
#         # 复数特征变换
#         x_transformed = self.complex_lin(x_complex)
#
#         # 应用复数激活函数
#         x_transformed = self.complex_activation(x_transformed)
#
#         # 应用复数dropout
#         if self.training and self.dropout > 0:
#             x_transformed = complex_dropout(x_transformed, p=self.dropout, training=self.training)
#
#         # 批量构建邻接矩阵
#         batch_adj = self.build_adjacency_matrices_batch(edge_index, num_nodes, x_transformed)
#
#         # 批量演化特征
#         evolved_features = self.batch_evolve_features(x_transformed, batch_adj, self.complex_time_param)
#
#         # 添加额外的复数偏置
#         if self.complex_extra_bias is not None:
#             evolved_features = evolved_features + self.complex_extra_bias
#
#         return evolved_features
#
#
# class OptimizedComplexUnitaryGCN(nn.Module):
#     """优化的多层复数酉演化GCN网络"""
#
#     def __init__(self, input_dim, hidden_dims, output_dim,
#                  evolution_times=None, dropout=0.0,
#                  max_matrix_exp_terms=8):
#         super(OptimizedComplexUnitaryGCN, self).__init__()
#
#         # 构建层列表
#         dims = [input_dim] + hidden_dims + [output_dim]
#         self.layers = nn.ModuleList()
#
#         if evolution_times is None:
#             evolution_times = [1.0] * (len(dims) - 1)
#
#         for i in range(len(dims) - 1):
#             layer = OptimizedComplexUnitaryEvolutionGCNConv(
#                 in_channels=dims[i],
#                 out_channels=dims[i + 1],
#                 evolution_time=evolution_times[i],
#                 dropout=dropout if i < len(dims) - 2 else 0.0,
#                 max_matrix_exp_terms=max_matrix_exp_terms
#             )
#             self.layers.append(layer)
#
#     def forward(self, data):
#         x, edge_index, batch = data.x, data.edge_index, data.batch
#
#         # 前向传播
#         for layer in self.layers:
#             x = layer(x, edge_index)
#
#         # 取幅度并进行图池化
#         x = x.abs()
#         x = global_mean_pool(x, batch)
#         x = F.log_softmax(x, dim=1)
#
#         return x
#
#
# # 进一步优化的版本 - 使用消息传递机制
# class MessagePassingComplexUnitaryGCN(MessagePassing):
#     """基于消息传递的复数酉演化GCN - 更高效的GPU实现"""
#
#     def __init__(self, in_channels, out_channels, evolution_time=1.0,
#                  dropout=0.0, activation='complex_relu'):
#         super(MessagePassingComplexUnitaryGCN, self).__init__(aggr='add')
#
#         self.in_channels = in_channels
#         self.out_channels = out_channels
#         self.dropout = dropout
#         self.activation = activation
#
#         # 复数线性变换
#         self.complex_lin = ComplexLinear(in_channels, out_channels)
#
#         # 演化参数
#         self.time_param_real = nn.Parameter(torch.tensor(evolution_time, dtype=torch.float32))
#         self.time_param_imag = nn.Parameter(torch.zeros(1, dtype=torch.float32))
#
#         # 消息传递的权重
#         self.msg_lin = ComplexLinear(out_channels, out_channels)
#
#     @property
#     def complex_time_param(self):
#         return torch.complex(self.time_param_real, self.time_param_imag)
#
#     def message(self, x_j, x_i):
#         """计算消息 - 模拟量子演化的消息传递"""
#         # 应用复数时间演化
#         time_factor = torch.exp(-1j * self.complex_time_param)
#         evolved_msg = self.msg_lin(x_j) * time_factor
#         return evolved_msg
#
#     def update(self, aggr_out, x):
#         """更新节点特征"""
#         # 结合原始特征和聚合消息
#         evolution_factor = torch.exp(-1j * self.complex_time_param * 0.5)
#         updated = x * evolution_factor + aggr_out * (1 - evolution_factor)
#         return updated
#
#     def forward(self, x, edge_index):
#         # 确保输入是复数形式
#         if x.dtype not in [torch.complex64, torch.complex128]:
#             x = torch.complex(x, torch.zeros_like(x))
#
#         # 线性变换
#         x = self.complex_lin(x)
#
#         # 应用激活函数
#         if self.activation == 'complex_relu':
#             x = complex_relu(x)
#
#         # 应用dropout
#         if self.training and self.dropout > 0:
#             x = complex_dropout(x, p=self.dropout, training=self.training)
#
#         # 消息传递
#         x = self.propagate(edge_index, x=x)
#
#         return x
#
#
# # 使用示例和性能测试
# def test_optimized_models():
#     """测试优化后的模型性能"""
#     import time
#
#     print("🚀 GPU优化的复数酉演化GCN测试")
#     print("=" * 50)
#
#     # 创建测试数据
#     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#     print(f"使用设备: {device}")
#
#     num_nodes = 1000
#     num_features = 64
#     num_classes = 10
#
#     # 生成随机图数据
#     x = torch.randn(num_nodes, num_features, device=device)
#
#     # 创建随机边
#     num_edges = num_nodes * 5
#     edge_index = torch.randint(0, num_nodes, (2, num_edges), device=device)
#
#     # 创建batch（用于图级别任务）
#     batch = torch.zeros(num_nodes, dtype=torch.long, device=device)
#
#     class TestData:
#         def __init__(self, x, edge_index, batch):
#             self.x = x
#             self.edge_index = edge_index
#             self.batch = batch
#
#     data = TestData(x, edge_index, batch)
#
#     # 测试原始模型
#     print("📊 原始模型性能测试（CPU密集型）...")
#     # 这里我们不运行原始模型，因为它会很慢
#
#     # 测试优化模型1
#     print("📊 优化模型1性能测试（批量矩阵运算）...")
#     model1 = OptimizedComplexUnitaryGCN(
#         input_dim=num_features,
#         hidden_dims=[32, 16],
#         output_dim=num_classes,
#         dropout=0.1,
#         max_matrix_exp_terms=6
#     ).to(device)
#
#     # 预热
#     with torch.no_grad():
#         _ = model1(data)
#
#     # 计时
#     start_time = time.time()
#     for _ in range(10):
#         with torch.no_grad():
#             output1 = model1(data)
#     end_time = time.time()
#
#     print(f"   平均推理时间: {(end_time - start_time) / 10:.4f}s")
#     print(f"   输出形状: {output1.shape}")
#     print(f"   GPU内存使用: {torch.cuda.memory_allocated(device) / 1024 ** 2:.2f}MB")
#
#     # 测试优化模型2
#     print("📊 优化模型2性能测试（消息传递）...")
#     model2 = MessagePassingComplexUnitaryGCN(
#         in_channels=num_features,
#         out_channels=num_classes,
#         evolution_time=1.0,
#         dropout=0.1
#     ).to(device)
#
#     # 预热
#     with torch.no_grad():
#         _ = model2(x, edge_index)
#
#     # 计时
#     start_time = time.time()
#     for _ in range(10):
#         with torch.no_grad():
#             output2 = model2(x, edge_index)
#     end_time = time.time()
#
#     print(f"   平均推理时间: {(end_time - start_time) / 10:.4f}s")
#     print(f"   输出形状: {output2.shape}")
#     print(f"   GPU内存使用: {torch.cuda.memory_allocated(device) / 1024 ** 2:.2f}MB")
#
#     # 测试梯度计算
#     print("📊 梯度计算测试...")
#     model2.train()
#     optimizer = torch.optim.Adam(model2.parameters(), lr=0.01)
#
#     output = model2(x, edge_index)
#     y_true = torch.randint(0, num_classes, (num_nodes,), device=device)
#     loss = F.cross_entropy(output.real, y_true)
#
#     start_time = time.time()
#     loss.backward()
#     optimizer.step()
#     end_time = time.time()
#
#     print(f"   反向传播时间: {end_time - start_time:.4f}s")
#     print(f"   损失值: {loss.item():.4f}")
#
#     print("✅ 所有测试完成!")
#
#
# if __name__ == "__main__":
#     test_optimized_models()


import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing
from torch_geometric.nn import global_mean_pool
from complexPyTorch.complexLayers import ComplexLinear
from complexPyTorch.complexFunctions import complex_relu, complex_dropout
from torch_sparse import SparseTensor
import math


class OptimizedComplexUnitaryEvolutionGCNConv(MessagePassing):
    """GPU优化的复数酉演化GCN卷积层"""

    def __init__(self, in_channels, out_channels,
                 evolution_time=1.0,
                 normalize=True,
                 bias=True,
                 dropout=0.0,
                 activation='complex_relu',
                 max_matrix_exp_terms=10):
        super(OptimizedComplexUnitaryEvolutionGCNConv, self).__init__(aggr='add')

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.evolution_time = evolution_time
        self.normalize = normalize
        self.dropout = dropout
        self.activation = activation
        self.max_matrix_exp_terms = max_matrix_exp_terms

        # 复数特征变换层
        self.complex_lin = ComplexLinear(in_channels, out_channels)

        # 可学习的演化时间参数
        self.time_param_real = nn.Parameter(torch.tensor(evolution_time, dtype=torch.float32))
        self.time_param_imag = nn.Parameter(torch.zeros(1, dtype=torch.float32))

        # 额外的复数偏置
        self.use_extra_bias = bias
        if self.use_extra_bias:
            self.extra_bias_real = nn.Parameter(torch.Tensor(out_channels))
            self.extra_bias_imag = nn.Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('extra_bias_real', None)
            self.register_parameter('extra_bias_imag', None)

        self.reset_parameters()

    def reset_parameters(self):
        if self.extra_bias_real is not None:
            nn.init.zeros_(self.extra_bias_real)
            nn.init.zeros_(self.extra_bias_imag)

    @property
    def complex_time_param(self):
        """获取复数时间参数"""
        return torch.complex(self.time_param_real, self.time_param_imag)

    @property
    def complex_extra_bias(self):
        """获取额外的复数偏置"""
        if self.extra_bias_real is not None:
            return torch.complex(self.extra_bias_real, self.extra_bias_imag)
        return None

    def matrix_exp_taylor(self, A, max_terms=10):
        """使用泰勒级数在GPU上计算矩阵指数"""
        device = A.device
        dtype = A.dtype

        # 初始化结果矩阵
        result = torch.eye(A.shape[-1], device=device, dtype=dtype).expand_as(A)
        term = torch.eye(A.shape[-1], device=device, dtype=dtype).expand_as(A)

        # 泰勒级数展开
        for i in range(1, max_terms + 1):
            term = torch.matmul(term, A) / i
            result = result + term

        return result

    def build_adjacency_matrices_batch(self, edge_index, num_nodes, x):
        """批量构建所有节点的邻接矩阵"""
        device = x.device
        dtype = torch.complex64

        # 创建完整的邻接矩阵
        adj_matrix = torch.zeros(num_nodes, num_nodes, device=device, dtype=dtype)

        # 填充邻接矩阵
        adj_matrix[edge_index[0], edge_index[1]] = 1.0 + 0.0j

        # 为每个节点创建包含自身和邻居的子图邻接矩阵
        # 这里我们使用一个简化的方法：直接使用度数作为子图大小
        degrees = torch.zeros(num_nodes, device=device, dtype=torch.long)
        degrees = degrees.scatter_add(0, edge_index[0], torch.ones_like(edge_index[0]))
        degrees = degrees.scatter_add(0, edge_index[1], torch.ones_like(edge_index[1]))

        # 限制最大邻居数以控制计算复杂度
        max_neighbors = min(10, degrees.max().item())

        # 创建批量邻接矩阵 (num_nodes, max_size, max_size)
        max_size = max_neighbors + 1  # +1 for the node itself
        batch_adj = torch.zeros(num_nodes, max_size, max_size, device=device, dtype=dtype)

        # 为每个节点构建子图
        for node_idx in range(num_nodes):
            # 找到邻居
            neighbors = []
            mask = (edge_index[0] == node_idx) | (edge_index[1] == node_idx)
            if mask.any():
                node_edges = edge_index[:, mask]
                for src, dst in node_edges.t():
                    if src == node_idx:
                        neighbors.append(dst.item())
                    else:
                        neighbors.append(src.item())

            # 去重并限制数量
            neighbors = list(set(neighbors))[:max_neighbors - 1]
            subgraph_nodes = [node_idx] + neighbors

            # 构建子图邻接矩阵
            for i, node_i in enumerate(subgraph_nodes):
                for j, node_j in enumerate(subgraph_nodes):
                    if adj_matrix[node_i, node_j] != 0:
                        batch_adj[node_idx, i, j] = 1.0 + 0.0j

        return batch_adj

    def batch_unitary_evolution(self, batch_adj, complex_time):
        """批量计算酉演化矩阵"""
        device = batch_adj.device
        num_nodes, max_size, _ = batch_adj.shape

        # 计算 -i * A * t
        evolution_arg = -1j * batch_adj * complex_time

        # 使用泰勒级数计算矩阵指数
        G_t = self.matrix_exp_taylor(evolution_arg, self.max_matrix_exp_terms)

        # 计算酉演化矩阵的简化版本
        # 这里我们直接使用G_t的归一化版本
        # 在实际应用中，可以根据需要进行更复杂的酉化处理
        norms = torch.norm(G_t, dim=(-2, -1), keepdim=True)
        G_t_normalized = G_t / (norms + 1e-8)

        return G_t_normalized

    def batch_evolve_features(self, x, batch_adj, complex_time):
        """批量演化节点特征"""
        device = x.device
        num_nodes, feature_dim = x.shape
        max_size = batch_adj.shape[1]

        # 计算演化矩阵
        U = self.batch_unitary_evolution(batch_adj, complex_time)

        # 准备特征矩阵
        # 为每个节点准备其子图特征
        batch_features = torch.zeros(num_nodes, max_size, feature_dim,
                                     device=device, dtype=x.dtype)

        # 简化版本：每个节点的第一个位置是自己，其他位置填充邻居特征的平均值
        batch_features[:, 0, :] = x

        # 对于其他位置，我们使用邻域特征的聚合
        for node_idx in range(num_nodes):
            # 找到邻居并聚合特征
            neighbors = []
            edge_mask = (batch_adj[node_idx, 0, 1:] != 0)
            if edge_mask.any():
                # 这里简化处理，使用平均池化
                neighbor_features = x.mean(dim=0, keepdim=True)
                for i in range(1, min(max_size, edge_mask.sum().item() + 1)):
                    batch_features[node_idx, i, :] = neighbor_features

        # 应用演化矩阵
        # U: (num_nodes, max_size, max_size)
        # batch_features: (num_nodes, max_size, feature_dim)
        evolved_features = torch.matmul(U, batch_features)

        # 返回每个节点演化后的特征（取第一个位置）
        return evolved_features[:, 0, :]

    def complex_activation(self, x):
        """应用复数激活函数"""
        if self.activation == 'complex_relu':
            return complex_relu(x)
        elif self.activation == 'none':
            return x
        else:
            return complex_relu(x)

    def prepare_complex_input(self, x):
        """将实数输入转换为复数形式"""
        if x.dtype in [torch.complex64, torch.complex128]:
            return x
        else:
            return torch.complex(x, torch.zeros_like(x))

    def forward(self, x, edge_index):
        num_nodes = x.size(0)

        # 确保输入是复数形式
        x_complex = self.prepare_complex_input(x)

        # 复数特征变换
        x_transformed = self.complex_lin(x_complex)

        # 应用复数激活函数
        x_transformed = self.complex_activation(x_transformed)

        # 应用复数dropout
        if self.training and self.dropout > 0:
            x_transformed = complex_dropout(x_transformed, p=self.dropout, training=self.training)

        # 批量构建邻接矩阵
        batch_adj = self.build_adjacency_matrices_batch(edge_index, num_nodes, x_transformed)

        # 批量演化特征
        evolved_features = self.batch_evolve_features(x_transformed, batch_adj, self.complex_time_param)

        # 添加额外的复数偏置
        if self.complex_extra_bias is not None:
            evolved_features = evolved_features + self.complex_extra_bias

        return evolved_features


class OptimizedComplexUnitaryGCN(nn.Module):
    """优化的多层复数酉演化GCN网络"""

    def __init__(self, input_dim, hidden_dims, output_dim,
                 evolution_times=None, dropout=0.0,
                 max_matrix_exp_terms=8):
        super(OptimizedComplexUnitaryGCN, self).__init__()

        # 构建层列表
        dims = [input_dim] + hidden_dims + [output_dim]
        self.layers = nn.ModuleList()

        if evolution_times is None:
            evolution_times = [1.0] * (len(dims) - 1)

        for i in range(len(dims) - 1):
            layer = OptimizedComplexUnitaryEvolutionGCNConv(
                in_channels=dims[i],
                out_channels=dims[i + 1],
                evolution_time=evolution_times[i],
                dropout=dropout if i < len(dims) - 2 else 0.0,
                max_matrix_exp_terms=max_matrix_exp_terms
            )
            self.layers.append(layer)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch

        # 前向传播
        for layer in self.layers:
            x = layer(x, edge_index)

        # 取幅度并进行图池化
        x = x.abs()
        x = global_mean_pool(x, batch)
        x = F.log_softmax(x, dim=1)

        return x


# 单层消息传递组件
class MessagePassingComplexUnitaryLayer(MessagePassing):
    """基于消息传递的复数酉演化层 - 单层实现"""

    def __init__(self, in_channels, out_channels, evolution_time=1.0,
                 dropout=0.0, activation='complex_relu'):
        super(MessagePassingComplexUnitaryLayer, self).__init__(aggr='add')

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.dropout = dropout
        self.activation = activation

        # 复数线性变换
        self.complex_lin = ComplexLinear(in_channels, out_channels)

        # 演化参数
        self.time_param_real = nn.Parameter(torch.tensor(evolution_time, dtype=torch.float32))
        self.time_param_imag = nn.Parameter(torch.zeros(1, dtype=torch.float32))

        # 消息传递的权重
        self.msg_lin = ComplexLinear(out_channels, out_channels)

    @property
    def complex_time_param(self):
        return torch.complex(self.time_param_real, self.time_param_imag)

    def message(self, x_j, x_i):
        """计算消息 - 模拟量子演化的消息传递"""
        # 应用复数时间演化
        time_factor = torch.exp(-1j * self.complex_time_param)
        evolved_msg = self.msg_lin(x_j) * time_factor
        return evolved_msg

    def update(self, aggr_out, x):
        """更新节点特征"""
        # 结合原始特征和聚合消息
        evolution_factor = torch.exp(-1j * self.complex_time_param * 0.5)
        updated = x * evolution_factor + aggr_out * (1 - evolution_factor)
        return updated

    def forward(self, x, edge_index):
        # 确保输入是复数形式
        if x.dtype not in [torch.complex64, torch.complex128]:
            x = torch.complex(x, torch.zeros_like(x))

        # 线性变换
        x = self.complex_lin(x)

        # 应用激活函数
        if self.activation == 'complex_relu':
            x = complex_relu(x)

        # 应用dropout
        if self.training and self.dropout > 0:
            x = complex_dropout(x, p=self.dropout, training=self.training)

        # 消息传递
        x = self.propagate(edge_index, x=x)

        return x


# 完整的多层网络 - 修复接口问题
class MessagePassingComplexUnitaryGCN(nn.Module):
    """基于消息传递的复数酉演化GCN网络 - 统一接口"""

    def __init__(self, input_dim, hidden_dims, output_dim,
                 evolution_times=None, dropout=0.0, activation='complex_relu'):
        super(MessagePassingComplexUnitaryGCN, self).__init__()

        # 构建层列表
        dims = [input_dim] + hidden_dims + [output_dim]
        self.layers = nn.ModuleList()

        if evolution_times is None:
            evolution_times = [1.0] * (len(dims) - 1)

        for i in range(len(dims) - 1):
            layer = MessagePassingComplexUnitaryLayer(
                in_channels=dims[i],
                out_channels=dims[i + 1],
                evolution_time=evolution_times[i],
                dropout=dropout if i < len(dims) - 2 else 0.0,
                activation=activation
            )
            self.layers.append(layer)

    def forward(self, data):
        """统一的forward接口，接受data对象"""
        x, edge_index, batch = data.x, data.edge_index, data.batch

        # 前向传播
        for layer in self.layers:
            x = layer(x, edge_index)

        # 取幅度并进行图池化
        x = x.abs()
        x = global_mean_pool(x, batch)
        x = F.log_softmax(x, dim=1)

        return x


# 使用示例和性能测试
def test_optimized_models():
    """测试优化后的模型性能"""
    import time

    print("🚀 GPU优化的复数酉演化GCN测试")
    print("=" * 50)

    # 创建测试数据
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")

    num_nodes = 1000
    num_features = 64
    num_classes = 10

    # 生成随机图数据
    x = torch.randn(num_nodes, num_features, device=device)

    # 创建随机边
    num_edges = num_nodes * 5
    edge_index = torch.randint(0, num_nodes, (2, num_edges), device=device)

    # 创建batch（用于图级别任务）
    batch = torch.zeros(num_nodes, dtype=torch.long, device=device)

    class TestData:
        def __init__(self, x, edge_index, batch):
            self.x = x
            self.edge_index = edge_index
            self.batch = batch

    data = TestData(x, edge_index, batch)

    # 测试原始模型
    print("📊 原始模型性能测试（CPU密集型）...")
    # 这里我们不运行原始模型，因为它会很慢

    # 测试优化模型1
    print("📊 优化模型1性能测试（批量矩阵运算）...")
    model1 = OptimizedComplexUnitaryGCN(
        input_dim=num_features,
        hidden_dims=[32, 16],
        output_dim=num_classes,
        dropout=0.1,
        max_matrix_exp_terms=6
    ).to(device)

    # 预热
    with torch.no_grad():
        _ = model1(data)

    # 计时
    start_time = time.time()
    for _ in range(10):
        with torch.no_grad():
            output1 = model1(data)
    end_time = time.time()

    print(f"   平均推理时间: {(end_time - start_time) / 10:.4f}s")
    print(f"   输出形状: {output1.shape}")
    print(f"   GPU内存使用: {torch.cuda.memory_allocated(device) / 1024 ** 2:.2f}MB")

    # 测试优化模型2
    print("📊 优化模型2性能测试（消息传递）...")
    model2 = MessagePassingComplexUnitaryGCN(
        input_dim=num_features,
        hidden_dims=[32, 16],
        output_dim=num_classes,
        dropout=0.1
    ).to(device)

    # 预热
    with torch.no_grad():
        _ = model2(data)

    # 计时
    start_time = time.time()
    for _ in range(10):
        with torch.no_grad():
            output2 = model2(data)
    end_time = time.time()

    print(f"   平均推理时间: {(end_time - start_time) / 10:.4f}s")
    print(f"   输出形状: {output2.shape}")
    print(f"   GPU内存使用: {torch.cuda.memory_allocated(device) / 1024 ** 2:.2f}MB")

    # 测试梯度计算
    print("📊 梯度计算测试...")
    model2.train()
    optimizer = torch.optim.Adam(model2.parameters(), lr=0.01)

    output = model2(data)
    y_true = torch.randint(0, num_classes, (1,), device=device)  # 图级别标签
    loss = F.cross_entropy(output, y_true)

    start_time = time.time()
    loss.backward()
    optimizer.step()
    end_time = time.time()

    print(f"   反向传播时间: {end_time - start_time:.4f}s")
    print(f"   损失值: {loss.item():.4f}")

    print("✅ 所有测试完成!")


if __name__ == "__main__":
    test_optimized_models()