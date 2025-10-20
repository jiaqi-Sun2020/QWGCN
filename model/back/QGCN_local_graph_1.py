import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing
import numpy as np
from complexPyTorch.complexLayers import ComplexLinear
from complexPyTorch.complexFunctions import complex_relu, complex_dropout
from scipy.linalg import expm

class ComplexUnitaryEvolutionGCNConv(MessagePassing):
    """基于复数酉演化的GCN卷积层"""

    def __init__(self, in_channels, out_channels,
                 evolution_time=1.0,
                 normalize=True,
                 bias=True,
                 dropout=0.0,
                 activation='complex_relu'):
        super(ComplexUnitaryEvolutionGCNConv, self).__init__(aggr='add')

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.evolution_time = evolution_time
        self.normalize = normalize
        self.dropout = dropout
        self.activation = activation

        # 复数特征变换层
        self.complex_lin = ComplexLinear(in_channels, out_channels)

        # 时间参数（可学习，复数形式）
        time_real = torch.tensor(evolution_time, dtype=torch.float32)
        time_imag = torch.zeros_like(time_real)
        self.time_param_real = nn.Parameter(time_real)
        self.time_param_imag = nn.Parameter(time_imag)

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

    def get_node_adjacency(self, node_idx, edge_index, num_nodes):
        """获取特定节点的邻接矩阵Ai"""
        # 找到与节点node_idx相关的所有边  edge_index[0] == node_idx：哪些边以 node_idx 为起点；edge_index[1] == node_idx：哪些边以 node_idx 为终点；| 是按位或，表示只要一个条件满足就选上
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
            return np.array([[0.0]], dtype=np.complex64), [node_idx]

        # 构建节点子图的邻接矩阵（包含目标节点）
        subgraph_nodes = [node_idx] + neighbors
        subgraph_size = len(subgraph_nodes)

        # 创建节点索引映射
        node_to_idx = {node: i for i, node in enumerate(subgraph_nodes)}

        # 构建子图邻接矩阵（复数形式）
        A_i = np.zeros((subgraph_size, subgraph_size), dtype=np.complex64)

        for src, dst in node_edges.t():
            if src.item() in node_to_idx and dst.item() in node_to_idx:
                src_idx = node_to_idx[src.item()]
                dst_idx = node_to_idx[dst.item()]
                A_i[src_idx, dst_idx] = 1.0 + 0.0j
                A_i[dst_idx, src_idx] = 1.0 + 0.0j

        return A_i, subgraph_nodes

    def unitary_evolution_matrix(self, A_i, complex_time):  # 与GCN 信息交换相同的操作  A和t是根据输入的时间来计算的
        if isinstance(complex_time, torch.Tensor):
            complex_time = complex_time.item()  # 转为 Python float/complex（假设它是一个 0维 Tensor）
        G_t = expm(-1j * A_i * complex_time)  # 获取演化矩阵
        N = A_i.shape[0]
        I_n = np.eye(N)  # NxN 单位矩阵
        G_dagger = G_t.conj().T  # G(t) 的共轭转置
        # print(np.dot(G_t, G_dagger))
        eigenvalues = np.linalg.eigvals(np.dot(G_t, G_dagger))

        sqrt_max_eigenvalue = np.sqrt(np.max(eigenvalues))  # 获取最大特征值

        Left_top = G_t / sqrt_max_eigenvalue
        Right_low = -G_t / sqrt_max_eigenvalue

        Right_top = np.sqrt(I_n - np.dot(G_t / sqrt_max_eigenvalue, G_dagger / sqrt_max_eigenvalue))
        Left_low = Right_top

        U = np.block([[Left_top, Right_top],  # 上半部分
                      [Left_low, Right_low]])  # 下半部分
        # return U[:N, :N]  # 演化   U
        return U.astype(np.complex64)  # 全尺寸 U



    def _matrix_exp_approx(self, A, terms=10):
        """矩阵指数的泰勒展开近似"""
        result = np.eye(A.shape[0], dtype=A.dtype)
        term = np.eye(A.shape[0], dtype=A.dtype)

        for i in range(1, terms + 1):
            term = np.dot(term, A) / i
            result += term

        return result

    def evolve_node_feature(self, node_idx, node_features, edge_index, num_nodes):
        """通过复数酉演化更新单个节点的特征"""
        # 获取节点的邻接矩阵
        A_i, subgraph_nodes = self.get_node_adjacency(node_idx, edge_index, num_nodes)
        # print("子图：",subgraph_nodes)
        # print("邻接矩阵：",A_i)

        if A_i.shape[0] == 1:
            return node_features[node_idx]


        # 计算酉演化矩阵
        U = self.unitary_evolution_matrix(A_i, self.complex_time_param)

        # 转换为PyTorch张量
        evolution_matrix_tensor = torch.from_numpy(U).to(
            dtype=torch.complex64, device=node_features.device
        )

        # 获取子图节点特征
        subgraph_features = node_features[subgraph_nodes]

        #子图节点特征 进行堆叠
        subgraph_features = torch.cat([subgraph_features, subgraph_features], dim=0)
        # 特征演化
        evolved_features = torch.matmul(evolution_matrix_tensor, subgraph_features)

        return evolved_features[0]  #目标节点本身的值！！！



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

        # 为每个节点进行演化更新
        evolved_features = []

        for node_idx in range(num_nodes):
            evolved_feature = self.evolve_node_feature(
                node_idx, x_transformed, edge_index, num_nodes
            )
            evolved_features.append(evolved_feature)

        # 堆叠所有演化后的特征
        out = torch.stack(evolved_features, dim=0)

        # 添加额外的复数偏置
        if self.complex_extra_bias is not None:
            out = out + self.complex_extra_bias

        return out

    def get_real_output(self, x, edge_index):
        """获取实数输出"""
        complex_output = self.forward(x, edge_index)
        return complex_output.real

    def get_magnitude_output(self, x, edge_index):
        """获取幅度输出"""
        complex_output = self.forward(x, edge_index)
        return torch.abs(complex_output)

    def get_phase_output(self, x, edge_index):
        """获取相位输出"""
        complex_output = self.forward(x, edge_index)
        return torch.angle(complex_output)


class ComplexUnitaryGCN(nn.Module):
    """多层复数酉演化GCN网络"""

    def __init__(self, input_dim, hidden_dims, output_dim,
                 evolution_times=None, dropout=0.0,
                 output_type='real'):
        super(ComplexUnitaryGCN, self).__init__()

        self.output_type = output_type

        # 构建层列表
        dims = [input_dim] + hidden_dims + [output_dim]
        self.layers = nn.ModuleList()

        if evolution_times is None:
            evolution_times = [1.0] * (len(dims) - 1)

        for i in range(len(dims) - 1):
            layer = ComplexUnitaryEvolutionGCNConv(
                in_channels=dims[i],
                out_channels=dims[i + 1],
                evolution_time=evolution_times[i],
                dropout=dropout if i < len(dims) - 2 else 0.0
            )
            self.layers.append(layer)

    def forward(self, x, edge_index):
        # 前向传播
        for i, layer in enumerate(self.layers):
            x = layer(x, edge_index)

        # 根据输出类型返回结果
        if self.output_type == 'real':
            return x.real
        elif self.output_type == 'magnitude':
            return torch.abs(x)
        elif self.output_type == 'phase':
            return torch.angle(x)
        else:  # 'complex'
            return x


# 测试函数 - 修复版本
def safe_min_max(tensor):
    """安全地计算张量的最小值和最大值"""
    if tensor.dtype in [torch.complex64, torch.complex128]:
        # 对于复数张量，返回实部和虚部的范围
        real_min, real_max = tensor.real.min().item(), tensor.real.max().item()
        imag_min, imag_max = tensor.imag.min().item(), tensor.imag.max().item()
        return f"实部[{real_min:.4f}, {real_max:.4f}], 虚部[{imag_min:.4f}, {imag_max:.4f}]"
    else:
        return f"[{tensor.min().item():.4f}, {tensor.max().item():.4f}]"


if __name__ == "__main__":
    print("🔬 复数酉演化GCN测试（修复版）")
    print("=" * 50)

    # 创建示例数据
    num_nodes = 10
    num_features = 8
    num_classes = 3

    x = torch.randn(num_nodes, num_features)

    # 创建边
    edge_list = []
    for i in range(num_nodes):
        edge_list.append([i, (i + 1) % num_nodes])
        edge_list.append([i, (i - 1) % num_nodes])

    import random

    random.seed(42)
    for _ in range(5):
        i, j = random.sample(range(num_nodes), 2)
        if [i, j] not in edge_list and [j, i] not in edge_list:
            edge_list.append([i, j])
            edge_list.append([j, i])

    edge_index = torch.tensor(edge_list).t().contiguous()

    print(f"📊 图数据信息:")
    print(f"   节点数: {num_nodes}")
    print(f"   边数: {edge_index.size(1)}")
    print(f"   输入特征维度: {num_features}")
    print(f"   输出类别数: {num_classes}")
    print()

    # 测试1: 单层复数酉演化GCN
    print("🧪 测试1: 单层复数酉演化GCN")
    try:
        layer = ComplexUnitaryEvolutionGCNConv(
            in_channels=num_features,
            out_channels=num_classes,
            evolution_time=1.0,
            dropout=0.1
        )

        print(f"   模型参数数量: {sum(p.numel() for p in layer.parameters())}")

        with torch.no_grad():
            output = layer(x, edge_index)

        print(f"   输出形状: {output.shape}")
        print(f"   输出类型: {output.dtype}")
        print(f"   输出范围: {safe_min_max(output)}")
        print(f"   输出幅度范围: [{torch.abs(output).min():.4f}, {torch.abs(output).max():.4f}]")

        # 测试不同输出模式
        real_output = layer.get_real_output(x, edge_index)
        magnitude_output = layer.get_magnitude_output(x, edge_index)
        phase_output = layer.get_phase_output(x, edge_index)

        print(f"   实部输出形状: {real_output.shape}")
        print(f"   幅度输出形状: {magnitude_output.shape}")
        print(f"   相位输出形状: {phase_output.shape}")
        print("   ✅ 单层测试通过!")

    except Exception as e:
        print(f"   ❌ 单层测试失败: {e}")

    print()

    # 测试2: 多层复数酉演化GCN网络
    print("🧪 测试2: 多层复数酉演化GCN网络")
    try:
        model = ComplexUnitaryGCN(
            input_dim=num_features,
            hidden_dims=[16, 8],
            output_dim=num_classes,
            evolution_times=[1.0, 0.5, 0.2],
            dropout=0.1,
            output_type='real'
        )

        print(f"   模型层数: {len(model.layers)}")
        print(f"   模型参数数量: {sum(p.numel() for p in model.parameters())}")

        with torch.no_grad():
            output = model(x, edge_index)

        print(f"   输出形状: {output.shape}")
        print(f"   输出类型: {output.dtype}")
        print(f"   输出范围: {safe_min_max(output)}")
        print("   ✅ 多层测试通过!")

    except Exception as e:
        print(f"   ❌ 多层测试失败: {e}")

    print()

    # 测试3: 不同输出类型测试
    print("🧪 测试3: 不同输出类型测试")
    output_types = ['real', 'magnitude', 'phase', 'complex']

    for output_type in output_types:
        try:
            model = ComplexUnitaryGCN(
                input_dim=num_features,
                hidden_dims=[12],
                output_dim=num_classes,
                output_type=output_type
            )

            with torch.no_grad():
                output = model(x, edge_index)

            range_str = safe_min_max(output)
            print(f"   {output_type:>9}输出 - 形状: {output.shape}, 类型: {output.dtype}, 范围: {range_str}")

        except Exception as e:
            print(f"   {output_type:>9}输出 - ❌ 失败: {e}")

    print()

    # 测试4: 梯度计算测试
    print("🧪 测试4: 梯度计算测试")
    try:
        model = ComplexUnitaryGCN(
            input_dim=num_features,
            hidden_dims=[6],
            output_dim=num_classes,
            output_type='real'
        )

        y_true = torch.randint(0, num_classes, (num_nodes,))

        # 前向传播
        output = model(x, edge_index)

        # 计算损失
        loss = F.cross_entropy(output, y_true)
        print(f"   损失值: {loss.item():.4f}")

        # 反向传播
        loss.backward()

        # 检查梯度
        grad_norms = []
        for name, param in model.named_parameters():
            if param.grad is not None:
                grad_norm = param.grad.norm().item()
                grad_norms.append(grad_norm)
                print(f"   {name}: 梯度范数 = {grad_norm:.6f}")

        if len(grad_norms) > 0:
            print(f"   平均梯度范数: {sum(grad_norms) / len(grad_norms):.6f}")
            print("   ✅ 梯度计算正常!")
        else:
            print("   ⚠️  没有检测到梯度")

    except Exception as e:
        print(f"   ❌ 梯度测试失败: {e}")

    print()

    # 测试5: 演化时间参数分析
    print("🧪 测试5: 演化时间参数分析")
    try:
        evolution_times = [0.1, 0.5, 1.0, 2.0]

        for evo_time in evolution_times:
            layer = ComplexUnitaryEvolutionGCNConv(
                in_channels=num_features,
                out_channels=num_classes,
                evolution_time=evo_time
            )

            with torch.no_grad():
                output = layer(x, edge_index)

            magnitude_mean = torch.abs(output).mean().item()
            phase_std = torch.angle(output).std().item()

            print(f"   演化时间 {evo_time:>4.1f}: 平均幅度={magnitude_mean:.4f}, 相位标准差={phase_std:.4f}")

        print("   ✅ 演化时间分析完成!")

    except Exception as e:
        print(f"   ❌ 演化时间分析失败: {e}")

    print()
    print("🎉 测试完成!")