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

class GAT_local(nn.Module):
    """
    Fast Quantum Graph Neural Network - High-Efficiency Version
    """

    def __init__(self, in_features, hidden_features, out_features,
                 num_layers=2, dropout_rate=0.1):
        super(GAT_local, self).__init__()

        self.num_layers = num_layers
        self.dropout_rate = dropout_rate

        # Build layer
        self.model_layers = nn.ModuleList()

        # Input layer
        self.model_layers.append(
            GATConv(in_features, hidden_features)
        )

        # Hidden layer
        for _ in range(num_layers - 2):
            self.model_layers.append(
                GATConv(hidden_features, hidden_features)
            )

        # Output layer
        if num_layers > 1:
            self.model_layers.append(
                GATConv(hidden_features, out_features)
            )

        # Normalization
        self.batch_norms = nn.ModuleList()
        for i in range(num_layers):
            if i < num_layers - 1:
                self.batch_norms.append(nn.BatchNorm1d(hidden_features))
            else:
                self.batch_norms.append(nn.BatchNorm1d(out_features))

        # Dropout layer
        self.dropout_layer = nn.Dropout(self.dropout_rate)

    def forward(self, data):
        """Fast forward propagation"""
        x, edge_index, batch = data.x, data.edge_index, data.batch

        # Layer-by-layer processing (including quantum evolution and batch normalization at each layer)
        for i, (quantum_layer, batch_norm) in enumerate(zip(self.model_layers, self.batch_norms)):
            # Quantum Graph Convolutional Layer
            x = quantum_layer(x, edge_index)

            # Normalization
            x = batch_norm(x)

            # Activation and Dropout (except for the last layer)
            if i < len(self.model_layers) - 1:
                x = F.relu(x)

                if self.training and self.dropout_rate > 0:
                    x = self.dropout_layer(x)

        # Graph Pooling
        x = global_mean_pool(x, batch)

        return F.log_softmax(x, dim=1)  # Output in log_softmax format

if __name__ == "__main__":
    # Load TUDataset
    dataset = TUDataset(root='./data', name='PROTEINS')
    loader = DataLoader(dataset, batch_size=32, shuffle=True)

    # Select Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Initialize the model
    model = GAT_local(
        in_features=dataset.num_node_features,
        hidden_features=64,
        out_features=dataset.num_classes
    ).to(device)

    # Get a batch of data
    data = next(iter(loader)).to(device)

    # Forward Propagation
    out = model(data)

    # Number of parameters and output shape for printing the model
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total number of model parameters: {total_params}")
    print(model)
    print(f"Output shape: {out.shape}")  # It should be (batch_size, num_classes)
