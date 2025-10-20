import torch
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv, global_mean_pool


class GATv2_layer(torch.nn.Module):
    def __init__(self, in_features=3, out_features=6):
        super().__init__()
        self.conv1 = GATv2Conv(in_channels=in_features, out_channels=8, heads=4, concat=True)
        self.conv2 = GATv2Conv(in_channels=8 * 4, out_channels=out_features, heads=1, concat=False)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch

        x = self.conv1(x, edge_index)
        x = F.elu(x)
        x = F.dropout(x, training=self.training)

        x = self.conv2(x, edge_index)
        x = global_mean_pool(x, batch)
        x = F.log_softmax(x, dim=1)
        return x
