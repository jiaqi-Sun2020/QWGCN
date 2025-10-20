import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
import numpy as np
from torch_geometric.nn import global_mean_pool
class GCN_layer(torch.nn.Module):
    def __init__(self,in_features = 3,out_features=6):
        super().__init__()
        self.conv1 = GCNConv(in_features, 16)
        self.conv2 = GCNConv(16, 64)
        self.conv3 = GCNConv(64, 16)
        self.conv4 = GCNConv(16, out_features)

    def forward(self, data):
        x, edge_index ,batch = data.x, data.edge_index,data.batch

        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = self.conv3(x, edge_index)
        x = F.relu(x)

        x = F.dropout(x,p=0.2, training=self.training)
        x = self.conv4(x, edge_index)

        x = global_mean_pool(x, batch)  # 图池化
        x= F.log_softmax(x, dim=1)

        return x


if __name__ == "__main__":

    model = GCN_layer(in_features=3, out_features=2)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"模型总参数量: {total_params}")
    print(model)