import torch
import torch.nn.functional as F
from torch_geometric.nn import GATConv
import numpy as np
from torch_geometric.nn import global_mean_pool

class GAT_layer(torch.nn.Module):
    def __init__(self, in_features=3, out_features=6, heads=8):
        super().__init__()
        self.heads = heads
        self.conv1 = GATConv(in_features, 16, heads=self.heads, concat=True)  # 多头注意力，输出 16*heads 维度
        self.conv2 = GATConv(16 * self.heads, out_features, heads=1, concat=False)  # 输出时只用一个头

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch

        x = self.conv1(x, edge_index)
        x = F.elu(x)  # 多头版通常用 ELU 激活更好

        x = F.dropout(x, p=0.6, training=self.training)  # GAT原论文里dropout是0.6

        x = self.conv2(x, edge_index)
        x = global_mean_pool(x, batch)
        x = F.log_softmax(x, dim=1)

        return x

# if __name__ == "__main__":
#     model = GAT_layer(in_features=2, out_features=2, heads=8)
#     print(model)
