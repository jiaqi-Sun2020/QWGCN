import torch
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv
import numpy as np
from torch_geometric.nn import global_mean_pool

class SAGE_layer(torch.nn.Module):
    def __init__(self, in_features=3, out_features=6):
        super().__init__()
        self.conv1 = SAGEConv(in_features, 16)
        self.conv2 = SAGEConv(16, out_features)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch

        x = self.conv1(x, edge_index)
        x = F.relu(x)

        x = F.dropout(x, p=0.5, training=self.training)  # 一般GraphSAGE dropout用0.5

        x = self.conv2(x, edge_index)
        x = global_mean_pool(x, batch)  # 图池化
        x = F.log_softmax(x, dim=1)


        return x

if __name__ == "__main__":
    model = SAGE_layer(in_features=2, out_features=2)
    print(model)
