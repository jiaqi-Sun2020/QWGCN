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
from torch_geometric.nn import SAGEConv
import numpy as np
from torch_geometric.nn import global_mean_pool

class Res_SAGE(torch.nn.Module):
    def __init__(self, in_features=3, out_features=6):
        super().__init__()
        self.conv1 = SAGEConv(in_features, 16)
        self.conv2 = SAGEConv(16, out_features)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch

        x = self.conv1(x, edge_index)
        x = F.relu(x)

        x = F.dropout(x, p=0.5, training=self.training)  # Generally, GraphSAGE dropout uses 0.5.

        x = self.conv2(x, edge_index)
        x = global_mean_pool(x, batch)  # Image pooling
        x = F.log_softmax(x, dim=1)


        return x

# if __name__ == "__main__":
#     model = SAGE_layer(in_features=2, out_features=2)
#     print(model)
