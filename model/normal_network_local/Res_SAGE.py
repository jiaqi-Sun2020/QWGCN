import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv, global_mean_pool
from torch_geometric.datasets import TUDataset
from torch_geometric.loader import DataLoader


class ResidualSAGE_local(nn.Module):
    """
    残差SAGE图神经网络 - 高效版本
    """

    def __init__(self, in_features, hidden_features, out_features,
                 num_layers=2, dropout_rate=0.1):
        super(ResidualSAGE_local, self).__init__()

        self.num_layers = num_layers
        self.dropout_rate = dropout_rate
        self.hidden_features = hidden_features

        # 构建层
        self.model_layers = nn.ModuleList()
        self.projection_layers = nn.ModuleList()  # 用于维度对齐的投影层

        # 输入层
        self.model_layers.append(
            SAGEConv(in_features, hidden_features)
        )
        # 输入层的投影层（如果输入维度与隐藏维度不同）
        if in_features != hidden_features:
            self.projection_layers.append(nn.Linear(in_features, hidden_features))
        else:
            self.projection_layers.append(nn.Identity())

        # 隐藏层
        for _ in range(num_layers - 2):
            self.model_layers.append(
                SAGEConv(hidden_features, hidden_features)
            )
            # 隐藏层不需要投影层，因为维度相同
            self.projection_layers.append(nn.Identity())

        # 输出层
        if num_layers > 1:
            self.model_layers.append(
                SAGEConv(hidden_features, out_features)
            )
            # 输出层的投影层（如果隐藏维度与输出维度不同）
            if hidden_features != out_features:
                self.projection_layers.append(nn.Linear(hidden_features, out_features))
            else:
                self.projection_layers.append(nn.Identity())

        # 批归一化
        self.batch_norms = nn.ModuleList()
        for i in range(num_layers):
            if i < num_layers - 1:
                self.batch_norms.append(nn.BatchNorm1d(hidden_features))
            else:
                self.batch_norms.append(nn.BatchNorm1d(out_features))

        # Dropout层
        self.dropout_layer = nn.Dropout(self.dropout_rate)

    def forward(self, data):
        """残差连接前向传播"""
        x, edge_index, batch = data.x, data.edge_index, data.batch

        # 逐层处理（包括残差连接）
        for i, (sage_layer, projection_layer, batch_norm) in enumerate(
                zip(self.model_layers, self.projection_layers, self.batch_norms)
        ):
            # 保存残差连接的输入
            residual = x

            # SAGE卷积层
            x = sage_layer(x, edge_index)

            # 批归一化
            x = batch_norm(x)

            # 残差连接：将输入通过投影层对齐维度后加到输出上
            projected_residual = projection_layer(residual)
            x = x + projected_residual

            # 激活和Dropout（最后一层除外）
            if i < len(self.model_layers) - 1:
                x = F.relu(x)

                if self.training and self.dropout_rate > 0:
                    x = self.dropout_layer(x)

        # 图级池化
        x = global_mean_pool(x, batch)

        return F.log_softmax(x, dim=1)  # 以log_softmax形式输出


class ResidualSAGEBlock(nn.Module):
    """
    可选的残差SAGE块实现 - 更模块化的设计
    """

    def __init__(self, in_features, out_features, dropout_rate=0.1):
        super(ResidualSAGEBlock, self).__init__()

        self.sage_conv = SAGEConv(in_features, out_features)
        self.batch_norm = nn.BatchNorm1d(out_features)
        self.dropout = nn.Dropout(dropout_rate)

        # 投影层用于维度对齐
        if in_features != out_features:
            self.projection = nn.Linear(in_features, out_features)
        else:
            self.projection = nn.Identity()

    def forward(self, x, edge_index):
        # 保存残差
        residual = x

        # 主路径
        out = self.sage_conv(x, edge_index)
        out = self.batch_norm(out)

        # 残差连接
        out = out + self.projection(residual)
        out = F.relu(out)
        out = self.dropout(out)

        return out


class ModularResidualSAGE(nn.Module):
    """
    使用模块化残差块的SAGE网络
    """

    def __init__(self, in_features, hidden_features, out_features,
                 num_layers=2, dropout_rate=0.1):
        super(ModularResidualSAGE, self).__init__()

        self.num_layers = num_layers

        # 构建残差块
        self.blocks = nn.ModuleList()

        # 第一层
        self.blocks.append(
            ResidualSAGEBlock(in_features, hidden_features, dropout_rate)
        )

        # 中间层
        for _ in range(num_layers - 2):
            self.blocks.append(
                ResidualSAGEBlock(hidden_features, hidden_features, dropout_rate)
            )

        # 输出层（不使用残差块，直接用卷积）
        if num_layers > 1:
            self.output_conv = SAGEConv(hidden_features, out_features)
            self.output_norm = nn.BatchNorm1d(out_features)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch

        # 通过残差块
        for block in self.blocks:
            x = block(x, edge_index)

        # 输出层
        if self.num_layers > 1:
            x = self.output_conv(x, edge_index)
            x = self.output_norm(x)

        # 图级池化
        x = global_mean_pool(x, batch)

        return F.log_softmax(x, dim=1)


if __name__ == "__main__":
    # 加载TUDataset
    dataset = TUDataset(root='./data', name='PROTEINS')
    loader = DataLoader(dataset, batch_size=32, shuffle=True)

    # 选择设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 测试两种残差SAGE模型
    print("=== 测试集成残差SAGE模型 ===")
    model1 = ResidualSAGE_local(
        in_features=dataset.num_node_features,
        hidden_features=64,
        out_features=dataset.num_classes,
        num_layers=4  # 更深的网络更能体现残差连接的优势
    ).to(device)

    print("=== 测试模块化残差SAGE模型 ===")
    model2 = ModularResidualSAGE(
        in_features=dataset.num_node_features,
        hidden_features=64,
        out_features=dataset.num_classes,
        num_layers=4
    ).to(device)

    # 获取一个批次的数据
    data = next(iter(loader)).to(device)

    # 前向传播
    out1 = model1(data)
    out2 = model2(data)

    # 打印模型信息
    total_params1 = sum(p.numel() for p in model1.parameters() if p.requires_grad)
    total_params2 = sum(p.numel() for p in model2.parameters() if p.requires_grad)

    print(f"集成残差SAGE模型参数量: {total_params1}")
    print(f"模块化残差SAGE模型参数量: {total_params2}")
    print(f"集成模型输出形状: {out1.shape}")
    print(f"模块化模型输出形状: {out2.shape}")