# ============================================================
# Copyright (C) 2025 Jiaqi Sun
# SPDX-License-Identifier: GPL-3.0-or-later
#
# Author: Jiaqi Sun
# Affiliation: JNU Quantum Physics Lab
# Email: 1623958071@qq.com
#
# Supervisor: Zhihao Bian
# Email: zhihaobian@jiangnan.edu.cn
#
# Description:
#   This file is part of the research codebase developed at
#   JNU Quantum Physics Lab, focusing on quantum-inspired
#   graph learning and complex-valued neural networks.
#
# ============================================================

import torch
import torch.nn.functional as F
from torch_geometric.nn import GATConv
import numpy as np
from torch_geometric.nn import global_mean_pool

class mGAT_layer(torch.nn.Module):
    def __init__(self, in_features=3, out_features=6, heads=8):
        super().__init__()
        self.heads = heads
        self.conv1 = GATConv(in_features, 16, heads=self.heads, concat=True)  # Multi-head attention, outputting 16*heads dimensions
        self.conv2 = GATConv(16 * self.heads, out_features, heads=1, concat=False)  # Use only one header when outputting

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch

        x = self.conv1(x, edge_index)
        x = F.elu(x)  # Multi-head versions are usually better activated with ELU.

        x = F.dropout(x, p=0.6, training=self.training)  # The dropout value in the original GAT paper is 0.6.

        x = self.conv2(x, edge_index)
        x = global_mean_pool(x, batch)
        x = F.log_softmax(x, dim=1)

        return x

# if __name__ == "__main__":
#     model = mGAT_layer(in_features=2, out_features=2, heads=8)
#     print(model)
