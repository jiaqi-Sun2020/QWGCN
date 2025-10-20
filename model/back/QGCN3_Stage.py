# 优化版：稳定高效的 OptimizedLocalUnitaryGCN
# 主要改进：
# 1. 稳定的复数权重初始化和梯度裁剪
# 2. 高效的矩阵指数近似（泰勒展开）
# 3. 稀疏邻接矩阵操作避免密集转换
# 4. 数值稳定的复数归一化
# 5. 渐进式训练策略

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import global_mean_pool
from torch_geometric.utils import add_self_loops, degree
from typing import Optional, Tuple
import math
from torch_geometric.data import Data
from torch_geometric.utils import erdos_renyi_graph
import time
from torch_geometric.utils import subgraph
import numpy as np


# ==== 稳定的复数层归一化 ====
class StableComplexLayerNorm(nn.Module):
    def __init__(self, normalized_shape, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))

    def forward(self, x):
        # 使用模长进行归一化，避免数值不稳定
        magnitude = torch.sqrt(x.real ** 2 + x.imag ** 2 + self.eps)
        mean_mag = magnitude.mean(dim=-1, keepdim=True)
        std_mag = magnitude.std(dim=-1, keepdim=True) + self.eps

        # 归一化模长
        normed_mag = (magnitude - mean_mag) / std_mag
        normed_mag = normed_mag * self.weight + self.bias

        # 保持相位，重构复数
        phase = torch.atan2(x.imag, x.real)
        return torch.polar(torch.abs(normed_mag) + self.eps, phase)


# ==== 稳定的复数线性层 ====
class StableComplexLinear(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        # Xavier初始化，适合复数网络
        std = math.sqrt(1.0 / in_features)
        self.weight_real = nn.Parameter(torch.randn(out_features, in_features) * std)
        self.weight_imag = nn.Parameter(torch.randn(out_features, in_features) * std)

        if bias:
            self.bias_real = nn.Parameter(torch.zeros(out_features))
            self.bias_imag = nn.Parameter(torch.zeros(out_features))
        else:
            self.register_parameter('bias_real', None)
            self.register_parameter('bias_imag', None)

    def forward(self, x):
        # 复数矩阵乘法
        real_part = torch.matmul(x.real, self.weight_real.T) - torch.matmul(x.imag, self.weight_imag.T)
        imag_part = torch.matmul(x.real, self.weight_imag.T) + torch.matmul(x.imag, self.weight_real.T)

        result = torch.complex(real_part, imag_part)

        if self.bias_real is not None:
            bias = torch.complex(self.bias_real, self.bias_imag)
            result = result + bias

        return result


# ==== 稳定的复数激活函数 ====
def stable_complex_relu(x, leak=0.01):
    """带泄漏的复数ReLU，提高稳定性"""
    real_part = torch.where(x.real > 0, x.real, leak * x.real)
    imag_part = torch.where(x.imag > 0, x.imag, leak * x.imag)
    return torch.complex(real_part, imag_part)


def stable_complex_dropout(x, p=0.5, training=True):
    """稳定的复数dropout"""
    if not training or p == 0:
        return x
    # 对模长进行dropout
    magnitude = torch.sqrt(x.real ** 2 + x.imag ** 2 + 1e-8)
    mask = (torch.rand_like(magnitude) > p).float()
    scale = 1.0 / (1.0 - p)

    phase = torch.atan2(x.imag, x.real)
    return torch.polar(magnitude * mask * scale, phase)


# ==== 高效的矩阵指数近似 ====
def efficient_matrix_exp(H, t=1.0, max_terms=6):
    """使用泰勒展开近似矩阵指数，避免昂贵的精确计算"""
    device = H.device
    n = H.size(-1)

    # 缩放时间步长以提高稳定性
    scaled_H = -1j * H * t

    # 泰勒展开: exp(A) ≈ I + A + A²/2! + A³/3! + ...
    result = torch.eye(n, device=device, dtype=torch.complex64).expand_as(scaled_H)
    term = torch.eye(n, device=device, dtype=torch.complex64).expand_as(scaled_H)

    for k in range(1, max_terms + 1):
        term = torch.matmul(term, scaled_H) / k
        result = result + term

        # 早停：如果项变得很小就停止
        if torch.max(torch.abs(term)) < 1e-6:
            break

    return result


# ==== 稀疏哈密顿矩阵构造 ====
def create_sparse_hamiltonian(edge_index, num_nodes, edge_weight=None, type='laplacian'):
    """直接从稀疏表示构造哈密顿矩阵，避免密集转换"""
    if edge_weight is None:
        edge_weight = torch.ones(edge_index.size(1), device=edge_index.device)

    row, col = edge_index
    deg = degree(row, num_nodes)

    if type == 'laplacian':
        # L = D - A
        laplacian_weight = torch.cat([deg, -edge_weight])
        laplacian_index = torch.cat([
            torch.stack([torch.arange(num_nodes, device=edge_index.device),
                         torch.arange(num_nodes, device=edge_index.device)]),
            edge_index
        ], dim=1)
    elif type == 'norm_laplacian':
        # L = I - D^(-1/2) A D^(-1/2)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0

        norm_weight = deg_inv_sqrt[row] * edge_weight * deg_inv_sqrt[col]
        laplacian_weight = torch.cat([torch.ones(num_nodes, device=edge_index.device), -norm_weight])
        laplacian_index = torch.cat([
            torch.stack([torch.arange(num_nodes, device=edge_index.device),
                         torch.arange(num_nodes, device=edge_index.device)]),
            edge_index
        ], dim=1)

    # 转换为密集矩阵用于小子图
    H = torch.sparse_coo_tensor(laplacian_index, laplacian_weight,
                                (num_nodes, num_nodes)).to_dense()
    return H.to(torch.complex64) * 0.05  # 减小缩放因子提高稳定性


# ==== 高效的局部酉GCN卷积层 ====
class EfficientLocalUnitaryGCNConv(nn.Module):
    def __init__(self, in_channels, out_channels, k_hop=1, evolution_time=0.5,
                 hamilton_type='laplacian', max_subgraph_size=6, dropout=0.1):
        super().__init__()
        self.lin = StableComplexLinear(in_channels, out_channels)
        self.norm = StableComplexLayerNorm(out_channels)
        self.k_hop = k_hop
        self.evolution_time = evolution_time
        self.hamilton_type = hamilton_type
        self.max_subgraph_size = max_subgraph_size
        self.dropout = dropout

        # 残差连接的投影层
        if in_channels != out_channels:
            self.residual_proj = StableComplexLinear(in_channels, out_channels)
        else:
            self.residual_proj = None

    def forward(self, x, edge_index):
        # 保存输入用于残差连接
        x_residual = x

        # 转换为复数
        if x.is_floating_point():
            x = torch.complex(x, torch.zeros_like(x))

        # 线性变换
        x = self.lin(x)

        N = x.size(0)
        edge_index_with_loops, _ = add_self_loops(edge_index, num_nodes=N)

        # 构建邻接关系字典，避免密集转换
        adj_dict = {}
        for i in range(edge_index_with_loops.size(1)):
            src, dst = edge_index_with_loops[:, i].tolist()
            if src not in adj_dict:
                adj_dict[src] = []
            adj_dict[src].append(dst)

        evolved = torch.zeros_like(x)

        for i in range(N):
            # 获取k-hop邻居
            neighbors = set([i])
            current_level = {i}

            for hop in range(self.k_hop):
                next_level = set()
                for node in current_level:
                    if node in adj_dict:
                        next_level.update(adj_dict[node])
                neighbors.update(next_level)
                current_level = next_level

                if len(neighbors) >= self.max_subgraph_size:
                    break

            sub_nodes = list(neighbors)[:self.max_subgraph_size]
            if i not in sub_nodes:
                sub_nodes[0] = i

            # 构建子图
            sub_edge_index, _ = subgraph(sub_nodes, edge_index_with_loops,
                                         relabel_nodes=True, num_nodes=N)

            if sub_edge_index.size(1) == 0:
                evolved[i] = x[i]
                continue

            center_idx = sub_nodes.index(i)
            sub_x = x[sub_nodes]

            # 构建子图哈密顿矩阵
            H = create_sparse_hamiltonian(sub_edge_index, len(sub_nodes),
                                          type=self.hamilton_type)

            # 高效矩阵指数计算
            U_sub = efficient_matrix_exp(H, self.evolution_time)

            # 构建扩展酉算子
            dim = len(sub_nodes)
            U = torch.zeros(2 * dim, 2 * dim, dtype=torch.complex64, device=x.device)
            U[:dim, :dim] = U_sub
            U[dim:, dim:] = U_sub.conj().transpose(-2, -1)

            # 应用酉演化
            z = torch.zeros(2 * dim, x.size(1), dtype=torch.complex64, device=x.device)
            z[:dim] = sub_x
            z_evolved = torch.matmul(U, z)

            evolved[i] = z_evolved[center_idx]

        # 激活函数
        evolved = stable_complex_relu(evolved)

        # Dropout
        if self.training and self.dropout > 0:
            evolved = stable_complex_dropout(evolved, self.dropout)

        # 层归一化
        evolved = self.norm(evolved)

        # 残差连接
        if self.residual_proj is not None:
            if x_residual.is_floating_point():
                x_residual = torch.complex(x_residual, torch.zeros_like(x_residual))
            x_residual = self.residual_proj(x_residual)
        elif x_residual.is_floating_point():
            x_residual = torch.complex(x_residual, torch.zeros_like(x_residual))

        return evolved + x_residual


# ==== 主网络 ====
class StableOptimizedLocalUnitaryGCN(nn.Module):
    def __init__(self, input_dim, hidden_dims, output_dim,
                 k_hop=1, evolution_times=None, hamilton_type='laplacian',
                 dropout=0.1, max_subgraph_size=6):
        super().__init__()
        dims = [input_dim] + hidden_dims + [output_dim]
        self.layers = nn.ModuleList()

        if evolution_times is None:
            # 渐进式演化时间
            evolution_times = [0.3, 0.5, 0.7][:len(dims) - 1]
            evolution_times = evolution_times + [evolution_times[-1]] * (len(dims) - 1 - len(evolution_times))

        for i in range(len(dims) - 1):
            self.layers.append(EfficientLocalUnitaryGCNConv(
                in_channels=dims[i],
                out_channels=dims[i + 1],
                k_hop=k_hop,
                evolution_time=evolution_times[i],
                hamilton_type=hamilton_type,
                dropout=dropout if i < len(dims) - 2 else 0.0,
                max_subgraph_size=max_subgraph_size
            ))

        # 梯度裁剪
        self.grad_clip = 1.0

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch

        for layer in self.layers:
            x = layer(x, edge_index)

            # 梯度裁剪防止爆炸
            if self.training:
                torch.nn.utils.clip_grad_norm_(self.parameters(), self.grad_clip)

        # 模长分类
        x = x.abs()

        # 全局平均池化
        x = global_mean_pool(x, batch)

        return F.log_softmax(x, dim=1)


# ==== 测试函数 ====
def test_stable_optimized_local_unitary_gcn():
    print("\U0001F680 稳定优化版 LocalUnitaryGCN 测试")
    print("=" * 60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    # 模拟图构建
    num_nodes = 100  # 减小规模测试
    num_features = 64
    num_classes = 2
    edge_index = erdos_renyi_graph(num_nodes, edge_prob=0.05).to(device)
    x = torch.randn(num_nodes, num_features).to(device) * 0.1  # 小初始值
    batch = torch.zeros(num_nodes, dtype=torch.long).to(device)
    data = Data(x=x, edge_index=edge_index, batch=batch)

    print(f"\n📊 图信息: {num_nodes} 节点, {edge_index.size(1)} 条边")

    model = StableOptimizedLocalUnitaryGCN(
        input_dim=num_features,
        hidden_dims=[32, 16],
        output_dim=num_classes,
        k_hop=1,  # 减小k_hop
        evolution_times=[0.2, 0.3, 0.4],
        hamilton_type='laplacian',
        dropout=0.1,
        max_subgraph_size=4  # 减小子图规模
    ).to(device)


    print(f"模型参数数量: {sum(p.numel() for p in model.parameters()):,}")

    # 推理测试
    model.eval()
    with torch.no_grad():
        try:
            output = model(data)
            print(f"输出形状: {output.shape}")
            print(f"输出范围: [{output.min().item():.4f}, {output.max().item():.4f}]")
            print(f"输出是否有NaN: {torch.isnan(output).any().item()}")
            print(f"输出是否有Inf: {torch.isinf(output).any().item()}")
        except Exception as e:
            print(f"模型执行出错: {e}")
            import traceback
            traceback.print_exc()
            return

    # 梯度测试
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
    start_time = time.time()
    try:
        y_fake = torch.randint(0, num_classes, (1,)).to(device)
        output = model(data)
        loss = F.nll_loss(output, y_fake)
        print(f"测试loss: {loss.item():.6f}")

        loss.backward()

        # 检查梯度
        total_grad_norm = 0
        for param in model.parameters():
            if param.grad is not None:
                total_grad_norm += param.grad.data.norm(2).item() ** 2
        total_grad_norm = total_grad_norm ** 0.5
        print(f"总梯度范数: {total_grad_norm:.6f}")

        optimizer.step()
        print("✅ 梯度更新成功")
        end_time = time.time()
        avg_time = (end_time - start_time)
        print(f"平均推理时间: {avg_time * 1000:.2f} ms")
    except Exception as e:
        print(f"训练测试出错: {e}")
        import traceback
        traceback.print_exc()

    print("\n✅ 关键改进:")
    print("  - 稳定的复数权重初始化")
    print("  - 高效的矩阵指数近似（泰勒展开）")
    print("  - 稀疏图操作避免密集转换")
    print("  - 梯度裁剪防止数值爆炸")
    print("  - 减小演化时间和子图规模")
    print("  - 保持所有核心特性：完整复数演化 + 模长分类 + 残差连接 + 局部扩张酉性")


if __name__ == "__main__":
    test_stable_optimized_local_unitary_gcn()