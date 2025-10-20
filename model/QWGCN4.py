import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import global_mean_pool
from torch_geometric.utils import degree, add_self_loops
from torch_scatter import scatter_add
import math

from complexPyTorch.complexLayers import ComplexLinear, ComplexBatchNorm1d
from complexPyTorch.complexFunctions import complex_relu, complex_dropout

"""
高效优化的量子图神经网络

核心优化策略：
1. 批量化处理：避免逐节点循环计算
2. 稀疏计算：使用稀疏矩阵操作
3. 简化量子演化：使用低阶近似和预计算
4. 缓存机制：重用计算结果
5. 并行化：利用GPU并行计算能力

物理原理保持不变：
- 哈密顿量：H = -A
- 时间演化：G(t) = exp(iAt) ≈ I + iAt + (iAt)²/2
- 量子态演化：|ψ(t)⟩ = G(t)|ψ(0)⟩
"""


class FastQuantumEvolution(nn.Module):
    """
    快速量子演化模块 - 大幅优化计算效率
    """

    def __init__(self, evolution_order=2):
        super(FastQuantumEvolution, self).__init__()

        self.evolution_order = evolution_order

        # 可学习参数
        self.evolution_time = nn.Parameter(torch.tensor(0.5))  # 减小初始值
        self.diffusion_strength = nn.Parameter(torch.tensor(1.0))

        # 用于数值稳定性
        self.eps = 1e-8

    def forward(self, x_complex, edge_index):
        """
        批量化的快速量子演化
        """
        num_nodes = x_complex.size(0)
        device = x_complex.device

        # 添加自环以确保连通性
        edge_index_with_loops, _ = add_self_loops(edge_index, num_nodes=num_nodes)

        row, col = edge_index_with_loops

        # 计算度数并归一化
        deg = degree(col, num_nodes, dtype=torch.float)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0

        # 归一化权重
        norm_weights = deg_inv_sqrt[row] * deg_inv_sqrt[col]

        # 第一阶近似：iAt * x
        # 使用稀疏矩阵乘法避免密集计算
        messages_1 = x_complex[col] * norm_weights.unsqueeze(-1)
        first_order = torch.zeros_like(x_complex)
        first_order.index_add_(0, row, messages_1)

        # 量子演化：G(t) ≈ I + it*diffusion*A*x
        evolved_x = x_complex + 1j * self.evolution_time * self.diffusion_strength * first_order

        if self.evolution_order >= 2:
            # 第二阶近似：(iAt)² * x / 2
            messages_2 = first_order[col] * norm_weights.unsqueeze(-1)
            second_order = torch.zeros_like(x_complex)
            second_order.index_add_(0, row, messages_2)

            second_coeff = (1j * self.evolution_time * self.diffusion_strength) ** 2 / 2.0
            evolved_x = evolved_x + second_coeff * second_order

        # 计算演化权重：|ψ(t)|²
        evolution_weights = (evolved_x * evolved_x.conj()).real.sum(dim=-1, keepdim=True)

        # 权重归一化
        weight_sum = evolution_weights.sum()
        if weight_sum > self.eps:
            evolution_weights = evolution_weights / weight_sum * num_nodes
        else:
            evolution_weights = torch.ones_like(evolution_weights)

        return evolution_weights, evolved_x


class EfficientMultiHop(nn.Module):
    """
    高效多跳聚合 - 使用矩阵幂次
    """

    def __init__(self, max_hops=2):
        super(EfficientMultiHop, self).__init__()

        self.max_hops = max_hops
        self.hop_weights = nn.Parameter(torch.ones(max_hops + 1))

    def forward(self, x_complex, edge_index):
        """
        基于矩阵幂次的快速多跳聚合
        """
        num_nodes = x_complex.size(0)
        device = x_complex.device

        # 添加自环
        edge_index_with_loops, _ = add_self_loops(edge_index, num_nodes=num_nodes)
        row, col = edge_index_with_loops

        # 度数归一化
        deg = degree(col, num_nodes, dtype=torch.float)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
        norm_weights = deg_inv_sqrt[row] * deg_inv_sqrt[col]

        # 存储不同跳数的结果
        hop_results = [x_complex]  # 0跳
        current_x = x_complex

        # 逐跳传播（复用中间结果）
        for hop in range(1, self.max_hops + 1):
            messages = current_x[col] * norm_weights.unsqueeze(-1)
            next_x = torch.zeros_like(x_complex)
            next_x.index_add_(0, row, messages)
            current_x = next_x
            hop_results.append(current_x)

        # 加权组合
        hop_weights_norm = F.softmax(self.hop_weights, dim=0)

        result = torch.zeros_like(x_complex)
        for hop, hop_result in enumerate(hop_results):
            result += hop_weights_norm[hop] * hop_result

        return result


class FastQWGNNLayer(nn.Module):
    """
    快速QWGNN层 - 优化版本
    """

    def __init__(self, in_features, out_features, max_hops=2, evolution_order=1):
        super(FastQWGNNLayer, self).__init__()

        self.in_features = in_features
        self.out_features = out_features

        # 快速量子演化（降低阶数）

        # 高效多跳聚合
        self.multi_hop = EfficientMultiHop(max_hops)

        # ComplexFC层
        self.complex_fc = ComplexLinear(in_features, out_features)

        # 残差连接
        if in_features != out_features:
            self.residual_proj = ComplexLinear(in_features, out_features)
        else:
            self.residual_proj = None

    def forward(self, x_complex, edge_index):
        """
        快速前向传播
        """
        # 保存残差
        residual = x_complex

        # 快速量子演化
        evolution_weights, evolved_features = self.quantum_evolution(x_complex, edge_index)

        # 多跳聚合
        aggregated_features = self.multi_hop(evolved_features, edge_index)

        # 权重应用
        weighted_features = evolution_weights * aggregated_features

        # 特征变换
        output = self.complex_fc(weighted_features)

        # 残差连接
        if self.residual_proj is not None:
            residual = self.residual_proj(residual)

        output = output + residual

        return output


class QWGCN_Fast(nn.Module):
    """
    快速量子图神经网络 - 高效版本
    """

    def __init__(self, in_features, hidden_features, out_features,
                 num_layers=2, max_hops=1, evolution_order=1, dropout_rate=0.1):
        super(QWGCN_Fast, self).__init__()

        self.num_layers = num_layers
        self.dropout_rate = dropout_rate

        # 构建层
        self.quantum_layers = nn.ModuleList()

        # 输入层
        self.quantum_layers.append(
            FastQWGNNLayer(in_features, hidden_features, max_hops, evolution_order)
        )

        # 隐藏层
        for _ in range(num_layers - 2):
            self.quantum_layers.append(
                FastQWGNNLayer(hidden_features, hidden_features, max_hops, evolution_order)
            )

        # 输出层
        if num_layers > 1:
            self.quantum_layers.append(
                FastQWGNNLayer(hidden_features, out_features, max_hops, evolution_order)
            )

        # 批归一化
        self.batch_norms = nn.ModuleList()
        for i in range(num_layers):
            if i < num_layers - 1:
                self.batch_norms.append(ComplexBatchNorm1d(hidden_features))
            else:
                self.batch_norms.append(ComplexBatchNorm1d(out_features))

    def forward(self, data):
        """快速前向传播"""
        x, edge_index, batch = data.x, data.edge_index, data.batch

        # 转换为复数
        x_real = x.float()
        x_complex = torch.complex(x_real, torch.zeros_like(x_real))

        # 逐层处理
        for i, (quantum_layer, batch_norm) in enumerate(zip(self.quantum_layers, self.batch_norms)):
            # 量子演化
            x_complex = quantum_layer(x_complex, edge_index)

            # 批归一化
            x_complex = batch_norm(x_complex)

            # 激活和Dropout（最后一层除外）
            if i < len(self.quantum_layers) - 1:
                x_complex = complex_relu(x_complex)

                if self.training and self.dropout_rate > 0:
                    x_complex = complex_dropout(x_complex, self.dropout_rate)

        # 转换为实数
        x = x_complex.abs()

        # 图级池化
        x = global_mean_pool(x, batch)

        return F.log_softmax(x, dim=1)


class AdaptiveQWGNN(nn.Module):
    """
    自适应量子GNN - 根据图大小调整计算复杂度
    """

    def __init__(self, in_features, hidden_features, out_features,
                 num_layers=2, dropout_rate=0.1):
        super(AdaptiveQWGNN, self).__init__()

        self.in_features = in_features
        self.hidden_features = hidden_features
        self.out_features = out_features
        self.num_layers = num_layers
        self.dropout_rate = dropout_rate

        # 预定义不同复杂度的模型
        self.models = nn.ModuleDict({
            'simple': QWGCN_Fast(in_features, hidden_features, out_features,
                                 num_layers, max_hops=1, evolution_order=1, dropout_rate=dropout_rate),
            'medium': QWGCN_Fast(in_features, hidden_features, out_features,
                                 num_layers, max_hops=2, evolution_order=1, dropout_rate=dropout_rate),
            'complex': QWGCN_Fast(in_features, hidden_features, out_features,
                                  num_layers, max_hops=2, evolution_order=2, dropout_rate=dropout_rate)
        })

    def forward(self, data):
        """根据图大小自适应选择模型复杂度"""
        num_nodes = data.x.size(0)
        num_edges = data.edge_index.size(1)

        # 根据图大小选择模型
        if num_nodes < 100:
            return self.models['complex'](data)
        elif num_nodes < 1000:
            return self.models['medium'](data)
        else:
            return self.models['simple'](data)


def benchmark_comparison():
    """性能基准测试"""
    from torch_geometric.data import Data
    from torch_geometric.nn import GCNConv
    import time

    print("=== 优化后性能对比测试 ===")

    # 测试不同规模
    test_configs = [
        {"nodes": 100, "edges": 300, "name": "小图"},
        {"nodes": 500, "edges": 1500, "name": "中图"},
        {"nodes": 1000, "edges": 3000, "name": "大图"}
    ]

    for config in test_configs:
        print(f"\n--- {config['name']} ({config['nodes']}节点, {config['edges']}边) ---")

        # 创建测试数据
        x = torch.randn(config['nodes'], 64)
        edge_index = torch.randint(0, config['nodes'], (2, config['edges']))
        edge_index = torch.unique(edge_index, dim=1)

        data = Data(x=x, edge_index=edge_index)
        data.batch = torch.zeros(config['nodes'], dtype=torch.long)

        # 定义对比模型
        class SimpleGCN(nn.Module):
            def __init__(self):
                super().__init__()
                self.conv1 = GCNConv(64, 128)
                self.conv2 = GCNConv(128, 10)

            def forward(self, data):
                x, edge_index, batch = data.x, data.edge_index, data.batch
                x = F.relu(self.conv1(x, edge_index))
                x = self.conv2(x, edge_index)
                x = global_mean_pool(x, batch)
                return F.log_softmax(x, dim=1)

        models = {
            "标准GCN": SimpleGCN(),
            "快速量子GNN": QWGCN_Fast(64, 128, 10, num_layers=2, max_hops=1, evolution_order=1),
            "自适应量子GNN": AdaptiveQWGNN(64, 128, 10, num_layers=2)
        }

        for name, model in models.items():
            try:
                model.eval()

                # 参数统计
                num_params = sum(p.numel() for p in model.parameters())

                # 性能测试
                with torch.no_grad():
                    # 预热
                    _ = model(data)

                    # 计时
                    start_time = time.time()
                    for _ in range(10):
                        output = model(data)
                    end_time = time.time()

                    avg_time = (end_time - start_time) / 10

                    print(f"  {name}:")
                    print(f"    参数: {num_params:,}")
                    print(f"    时间: {avg_time:.4f}s")
                    print(f"    输出: {output.shape}")

            except Exception as e:
                print(f"  {name}: 失败 - {str(e)}")


def test_optimizations():
    """测试优化效果"""
    from torch_geometric.data import Data

    print("=== 优化效果测试 ===")

    # 创建测试数据
    x = torch.randn(500, 64)
    edge_index = torch.randint(0, 500, (2, 1500))
    edge_index = torch.unique(edge_index, dim=1)

    data = Data(x=x, edge_index=edge_index)
    data.batch = torch.zeros(500, dtype=torch.long)

    x_complex = torch.complex(x, torch.zeros_like(x))

    print("测试各个优化组件:")

    # 测试快速量子演化
    print("\n1. 快速量子演化模块")
    fast_evolution = FastQuantumEvolution(evolution_order=1)

    import time
    start = time.time()
    weights, evolved = fast_evolution(x_complex, edge_index)
    end = time.time()

    print(f"   ✓ 计算时间: {end - start:.4f}s")
    print(f"   ✓ 权重形状: {weights.shape}")
    print(f"   ✓ 演化特征形状: {evolved.shape}")

    # 测试多跳聚合
    print("\n2. 高效多跳聚合")
    multi_hop = EfficientMultiHop(max_hops=2)

    start = time.time()
    aggregated = multi_hop(x_complex, edge_index)
    end = time.time()

    print(f"   ✓ 计算时间: {end - start:.4f}s")
    print(f"   ✓ 聚合特征形状: {aggregated.shape}")

    # 测试完整快速模型
    print("\n3. 完整快速模型")
    fast_model = QWGCN_Fast(64, 128, 10, num_layers=2, max_hops=1, evolution_order=1)
    fast_model.eval()

    with torch.no_grad():
        start = time.time()
        output = fast_model(data)
        end = time.time()

    print(f"   ✓ 总计算时间: {end - start:.4f}s")
    print(f"   ✓ 输出形状: {output.shape}")

    print("\n优化测试完成！")

    model_final = QWGCN_Fast(in_features=64,
                           hidden_features=128,
                           out_features=10,
                           num_layers=2,
                           max_hops=2,
                           evolution_order=1,
                           dropout_rate=0.1)

    print(model_final)



if __name__ == "__main__":
    # 测试优化效果
    test_optimizations()
    print("\n" + "=" * 60 + "\n")

    # 性能基准测试
    benchmark_comparison()