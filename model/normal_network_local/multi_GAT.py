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
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, global_mean_pool
from torch_geometric.datasets import TUDataset
from torch_geometric.loader import DataLoader

class MultiGAT_local(nn.Module):
    """
    Multi-head attention graph neural network - high-efficiency version
    """

    def __init__(self, in_features, hidden_features, out_features,
                 num_layers=2, heads=2, dropout_rate=0.1):
        super(MultiGAT_local, self).__init__()

        self.num_layers = num_layers
        self.heads = heads
        self.dropout_rate = dropout_rate

        # Build layer
        self.model_layers = nn.ModuleList()

        # Input layer - Multi-head attention, concat=True
        self.model_layers.append(
            GATConv(in_features, hidden_features, heads=heads, concat=True, dropout=dropout_rate)
        )

        # Hidden layer - Multi-head attention, concat=True
        for _ in range(num_layers - 2):
            self.model_layers.append(
                GATConv(hidden_features * heads, hidden_features, heads=heads, concat=True, dropout=dropout_rate)
            )

        # Output layer - Multi-head attention, concat=False (average)
        if num_layers > 1:
            self.model_layers.append(
                GATConv(hidden_features * heads, out_features, heads=heads, concat=False, dropout=dropout_rate)
            )

        # Normalization
        self.batch_norms = nn.ModuleList()
        for i in range(num_layers):
            if i < num_layers - 1:
                # The middle layer uses concat, so the dimension is hidden_features * heads
                self.batch_norms.append(nn.BatchNorm1d(hidden_features * heads))
            else:
                # The output layer uses average, so the dimension is out_features
                self.batch_norms.append(nn.BatchNorm1d(out_features))

        # Dropout layer
        self.dropout_layer = nn.Dropout(self.dropout_rate)

    def forward(self, data):
        """Multi-head attention forward propagation"""
        x, edge_index, batch = data.x, data.edge_index, data.batch

        # Layer-by-layer processing (including multi-head attention and batch normalization at each layer)
        for i, (gat_layer, batch_norm) in enumerate(zip(self.model_layers, self.batch_norms)):
            # Multi-head graph attention convolutional layer
            x = gat_layer(x, edge_index)

            # Normalization
            x = batch_norm(x)

            # Activation and Dropout (except for the last layer)
            if i < len(self.model_layers) - 1:
                x = F.relu(x)

                if self.training and self.dropout_rate > 0:
                    x = self.dropout_layer(x)

        # # Graph Pooling
        x = global_mean_pool(x, batch)

        return F.log_softmax(x, dim=1)  # Output in log_softmax format

if __name__ == "__main__":
    # Load TUDataset
    dataset = TUDataset(root='./data', name='PROTEINS')
    loader = DataLoader(dataset, batch_size=32, shuffle=True)

    # Select Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Initialize the model
    model = MultiGAT_local(
        in_features=dataset.num_node_features,
        hidden_features=64,
        out_features=dataset.num_classes,
        heads=8,  # 8个注意力头
        num_layers=2
    ).to(device)

    # Get a batch of data
    data = next(iter(loader)).to(device)

    # Forward Propagation
    out = model(data)

    # 打印模型参数数量和输出形状
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total number of model parameters: {total_params}")
    print(model)
    print(f"Output shape: {out.shape}")  # (batch_size, num_classes)
    print(f"Number of attention heads: {model.heads}")