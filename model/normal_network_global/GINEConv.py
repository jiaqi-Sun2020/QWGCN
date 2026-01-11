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
from torch_geometric.nn import GINEConv, global_mean_pool
from torch.nn import Sequential, Linear, ReLU


class GINE_layer(torch.nn.Module):
    def __init__(self, in_features=3, out_features=6):
        super().__init__()
        nn1 = Sequential(Linear(in_features, 16), ReLU(), Linear(16, 16))
        nn2 = Sequential(Linear(16, out_features), ReLU(), Linear(out_features, out_features))
        self.conv1 = GINEConv(nn1)
        self.conv2 = GINEConv(nn2)

    def forward(self, data):
        x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch

        x = self.conv1(x, edge_index, edge_attr)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)

        x = self.conv2(x, edge_index, edge_attr)
        x = global_mean_pool(x, batch)
        x = F.log_softmax(x, dim=1)


        return x
