import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool
from torch_geometric.datasets import TUDataset
from torch_geometric.loader import DataLoader

class GCN_local(nn.Module):
    """
    快速量子图神经网络 - 高效版本
    """

    def __init__(self, in_features, hidden_features, out_features,
                 num_layers=2, dropout_rate=0.1):
        super(GCN_local, self).__init__()

        self.num_layers = num_layers
        self.dropout_rate = dropout_rate

        # 构建层
        self.model_layers = nn.ModuleList()

        # 输入层
        self.model_layers.append(
            GCNConv(in_features, hidden_features)
        )

        # 隐藏层
        for _ in range(num_layers - 2):
            self.model_layers.append(
                GCNConv(hidden_features, hidden_features)
            )

        # 输出层
        if num_layers > 1:
            self.model_layers.append(
                GCNConv(hidden_features, out_features)
            )

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
        """快速前向传播"""
        x, edge_index, batch = data.x, data.edge_index, data.batch

        # 逐层处理（包括每层的量子演化和批归一化）
        for i, (quantum_layer, batch_norm) in enumerate(zip(self.model_layers, self.batch_norms)):
            # 量子图卷积层
            x = quantum_layer(x, edge_index)

            # 批归一化
            x = batch_norm(x)

            # 激活和Dropout（最后一层除外）
            if i < len(self.model_layers) - 1:
                x = F.relu(x)

                if self.training and self.dropout_rate > 0:
                    x = self.dropout_layer(x)

        # 图级池化
        x = global_mean_pool(x, batch)

        return F.log_softmax(x, dim=1)  # 以log_softmax形式输出

if __name__ == "__main__":
    # 加载TUDataset
    dataset = TUDataset(root='./data', name='PROTEINS')
    loader = DataLoader(dataset, batch_size=32, shuffle=True)

    # 选择设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 初始化模型
    model = GCN_local(
        in_features=dataset.num_node_features,
        hidden_features=64,
        out_features=dataset.num_classes
    ).to(device)

    # 获取一个批次的数据
    data = next(iter(loader)).to(device)

    # 前向传播
    out = model(data)

    # 打印模型参数数量和输出形状
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"模型总参数量: {total_params}")
    print(model)
    print(f"输出形状: {out.shape}")  # 应该是 (batch_size, num_classes)
