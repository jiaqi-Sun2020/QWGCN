import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import to_dense_adj
import numpy as np
from scipy.linalg import expm
import math


def unitary_dilation_operator(A, t=1):
    """酉扩张算子 - 计算演化矩阵"""
    G_t = expm(-1j * A * t)  # 获取演化矩阵
    N = A.shape[0]
    I_n = np.eye(N)  # NxN 单位矩阵
    G_dagger = G_t.conj().T  # G(t) 的共轭转置

    eigenvalues = np.linalg.eigvals(np.dot(G_t, G_dagger))
    sqrt_max_eigenvalue = np.sqrt(np.max(eigenvalues))  # 获取最大特征值

    Left_top = G_t / sqrt_max_eigenvalue
    Right_low = -G_t / sqrt_max_eigenvalue

    Right_top = np.sqrt(I_n - np.dot(G_t / sqrt_max_eigenvalue, G_dagger / sqrt_max_eigenvalue))
    Left_low = Right_top

    U = np.block([[Left_top, Right_top],  # 上半部分
                  [Left_low, Right_low]])  # 下半部分

    return U  # 返回全尺寸 U


def edge_i2Adj_M(edge_index, num_nodes):
    """将边索引转换为邻接矩阵"""
    adj_matrix = torch.zeros((num_nodes, num_nodes))
    for src, dst in edge_index.t():  # 转置 edge_index 遍历
        adj_matrix[src, dst] = 1
    return adj_matrix


class UnitaryEvolutionGCNConv(MessagePassing):
    """基于酉演化的GCN卷积层"""

    def __init__(self, in_channels, out_channels,
                 evolution_time=1.0,
                 normalize=True,
                 bias=True,
                 dropout=0.0):
        super(UnitaryEvolutionGCNConv, self).__init__(aggr='add')

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.evolution_time = evolution_time
        self.normalize = normalize
        self.dropout = dropout

        # 特征变换层
        self.lin = nn.Linear(in_channels, out_channels, bias=False)

        # 时间参数（可学习）
        self.time_param = nn.Parameter(torch.tensor(evolution_time, dtype=torch.float32))

        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.lin.weight)
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def get_node_adjacency(self, node_idx, edge_index, num_nodes):
        """获取特定节点的邻接矩阵Ai"""
        # 找到与节点node_idx相关的所有边
        mask = (edge_index[0] == node_idx) | (edge_index[1] == node_idx)
        node_edges = edge_index[:, mask]

        # 获取邻居节点
        neighbors = []
        for src, dst in node_edges.t():
            if src == node_idx:
                neighbors.append(dst.item())
            else:
                neighbors.append(src.item())

        # 去重并排序
        neighbors = sorted(list(set(neighbors)))

        if len(neighbors) == 0:
            # 如果没有邻居，返回1x1的零矩阵
            return np.array([[0.0]]), [node_idx]

        # 构建节点子图的邻接矩阵（包含目标节点）
        subgraph_nodes = [node_idx] + neighbors
        subgraph_size = len(subgraph_nodes)

        # 创建节点索引映射
        node_to_idx = {node: i for i, node in enumerate(subgraph_nodes)}

        # 构建子图邻接矩阵
        A_i = np.zeros((subgraph_size, subgraph_size))

        for src, dst in node_edges.t():
            if src.item() in node_to_idx and dst.item() in node_to_idx:
                src_idx = node_to_idx[src.item()]
                dst_idx = node_to_idx[dst.item()]
                A_i[src_idx, dst_idx] = 1.0
                A_i[dst_idx, src_idx] = 1.0  # 无向图

        return A_i, subgraph_nodes

    def evolve_node_feature(self, node_idx, node_features, edge_index, num_nodes):
        """通过酉演化更新单个节点的特征"""
        # 获取节点的邻接矩阵
        A_i, subgraph_nodes = self.get_node_adjacency(node_idx, edge_index, num_nodes)

        if A_i.shape[0] == 1:
            # 如果只有一个节点（没有邻居），直接返回原特征
            return node_features[node_idx]

        try:
            # 计算酉扩张算子
            U = unitary_dilation_operator(A_i, t=self.time_param.item())

            # 提取演化矩阵部分 (上半部分)
            evolution_matrix = U[:A_i.shape[0], :A_i.shape[0]]

            # 获取子图节点特征
            subgraph_features = node_features[subgraph_nodes]  # [subgraph_size, feature_dim]

            # 转换为复数进行演化计算
            evolution_matrix_tensor = torch.from_numpy(evolution_matrix).to(
                dtype=torch.complex64, device=node_features.device
            )
            subgraph_features_complex = subgraph_features.to(torch.complex64)

            # 特征演化: F_evolved = Evolution_Matrix @ F_neighbors
            evolved_features = torch.matmul(evolution_matrix_tensor, subgraph_features_complex)

            # 取实部作为更新后的特征
            evolved_features_real = evolved_features.real

            # 返回目标节点的更新特征（第0个位置是目标节点）
            return evolved_features_real[0]

        except Exception as e:
            # 如果演化失败，返回原始特征
            print(f"Evolution failed for node {node_idx}: {e}")
            return node_features[node_idx]

    def forward(self, x, edge_index):
        num_nodes = x.size(0)

        # 特征变换
        x_transformed = self.lin(x)
        if self.training and self.dropout > 0:
            x_transformed = F.dropout(x_transformed, p=self.dropout)

        # 为每个节点进行演化更新
        evolved_features = []

        for node_idx in range(num_nodes):
            evolved_feature = self.evolve_node_feature(
                node_idx, x_transformed, edge_index, num_nodes
            )
            evolved_features.append(evolved_feature)

        # 堆叠所有演化后的特征
        out = torch.stack(evolved_features, dim=0)

        if self.bias is not None:
            out = out + self.bias

        return out


class MultiStepUnitaryGCN(nn.Module):
    """多步酉演化GCN"""

    def __init__(self, in_channels, hidden_channels, out_channels,
                 num_layers=2, evolution_times=[1.0, 0.5]):
        super(MultiStepUnitaryGCN, self).__init__()

        self.num_layers = num_layers
        self.convs = nn.ModuleList()

        # 第一层
        self.convs.append(UnitaryEvolutionGCNConv(
            in_channels, hidden_channels,
            evolution_time=evolution_times[0] if len(evolution_times) > 0 else 1.0
        ))

        # 中间层
        for i in range(1, num_layers - 1):
            self.convs.append(UnitaryEvolutionGCNConv(
                hidden_channels, hidden_channels,
                evolution_time=evolution_times[i] if i < len(evolution_times) else 1.0
            ))

        # 最后一层
        if num_layers > 1:
            final_time = evolution_times[-1] if len(evolution_times) > 1 else 0.5
            self.convs.append(UnitaryEvolutionGCNConv(
                hidden_channels, out_channels,
                evolution_time=final_time
            ))

        self.dropout = nn.Dropout(0.5)

    def forward(self, x, edge_index):
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)

            if i < len(self.convs) - 1:  # 不在最后一层应用激活和dropout
                x = F.relu(x)
                x = self.dropout(x)

        return F.log_softmax(x, dim=1)


class AdaptiveTimeGCN(nn.Module):
    """自适应时间参数的酉演化GCN"""

    def __init__(self, in_channels, hidden_channels, out_channels):
        super(AdaptiveTimeGCN, self).__init__()

        # 时间预测网络
        self.time_predictor = nn.Sequential(
            nn.Linear(in_channels, hidden_channels // 2),
            nn.ReLU(),
            nn.Linear(hidden_channels // 2, 1),
            nn.Sigmoid()  # 输出0-1之间的时间参数
        )

        # GCN层
        self.conv1 = UnitaryEvolutionGCNConv(in_channels, hidden_channels)
        self.conv2 = UnitaryEvolutionGCNConv(hidden_channels, out_channels)

        self.dropout = nn.Dropout(0.5)

    def forward(self, x, edge_index):
        # 预测时间参数
        time_params = self.time_predictor(x)  # [num_nodes, 1]
        avg_time = time_params.mean().item()

        # 设置时间参数
        self.conv1.time_param.data = torch.tensor(avg_time, device=x.device)
        self.conv2.time_param.data = torch.tensor(avg_time * 0.5, device=x.device)

        # 前向传播
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.dropout(x)

        x = self.conv2(x, edge_index)

        return F.log_softmax(x, dim=1), time_params


# 使用示例
if __name__ == "__main__":
    # 创建示例数据
    num_nodes = 10  # 小规模测试
    num_features = 8
    num_classes = 3

    x = torch.randn(num_nodes, num_features)
    # 创建简单的边（环形图）
    edge_list = []
    for i in range(num_nodes):
        edge_list.append([i, (i + 1) % num_nodes])
        edge_list.append([i, (i - 1) % num_nodes])

    edge_index = torch.tensor(edge_list).t().contiguous()

    print(f"节点数: {num_nodes}")
    print(f"边数: {edge_index.size(1)}")
    print(f"输入特征维度: {num_features}")

    # 测试酉演化GCN
    model = MultiStepUnitaryGCN(
        in_channels=num_features,
        hidden_channels=16,
        out_channels=num_classes,
        num_layers=2,
        evolution_times=[1.0, 0.5]
    )

    print("开始前向传播...")
    output = model(x, edge_index)
    print(f"输出形状: {output.shape}")
    print(f"输出示例: {output[:3]}")

    # 测试自适应时间GCN
    print("\n测试自适应时间GCN...")
    adaptive_model = AdaptiveTimeGCN(
        in_channels=num_features,
        hidden_channels=16,
        out_channels=num_classes
    )

    adaptive_output, time_params = adaptive_model(x, edge_index)
    print(f"自适应输出形状: {adaptive_output.shape}")
    print(f"预测的时间参数: {time_params.mean().item():.4f}")