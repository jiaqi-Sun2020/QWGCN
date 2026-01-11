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
from torch_geometric.nn import SAGEConv, global_mean_pool
from torch_geometric.datasets import TUDataset
from torch_geometric.loader import DataLoader


class ResidualSAGE_local(nn.Module):
    """
    Residual SAGE graphical neural network
    """

    def __init__(self, in_features, hidden_features, out_features,
                 num_layers=2, dropout_rate=0.1):
        super(ResidualSAGE_local, self).__init__()

        self.num_layers = num_layers
        self.dropout_rate = dropout_rate
        self.hidden_features = hidden_features

        # Build layer
        self.model_layers = nn.ModuleList()
        self.projection_layers = nn.ModuleList()  # Projection layer for dimension alignment

        # Input layer
        self.model_layers.append(
            SAGEConv(in_features, hidden_features)
        )
        # Projection layer of the input layer (if the input dimension and hidden dimension are different)
        if in_features != hidden_features:
            self.projection_layers.append(nn.Linear(in_features, hidden_features))
        else:
            self.projection_layers.append(nn.Identity())

        # Hidden layer
        for _ in range(num_layers - 2):
            self.model_layers.append(
                SAGEConv(hidden_features, hidden_features)
            )
            # Hidden layers do not need projection layers because they have the same dimensions.
            self.projection_layers.append(nn.Identity())

        # Output layer
        if num_layers > 1:
            self.model_layers.append(
                SAGEConv(hidden_features, out_features)
            )
            # Projection layer of the output layer (if the hidden dimension is different from the output dimension)
            if hidden_features != out_features:
                self.projection_layers.append(nn.Linear(hidden_features, out_features))
            else:
                self.projection_layers.append(nn.Identity())

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
        """forward propagation"""
        x, edge_index, batch = data.x, data.edge_index, data.batch

        # Layer-by-layer processing (including residual connections)
        for i, (sage_layer, projection_layer, batch_norm) in enumerate(
                zip(self.model_layers, self.projection_layers, self.batch_norms)
        ):
            # Save the input of the residual connection
            residual = x

            # SAGE Convolutional Layer
            x = sage_layer(x, edge_index)

            # Normalization
            x = batch_norm(x)

            # Residual connection: The input is added to the output after being aligned in dimensions by a projection layer.
            projected_residual = projection_layer(residual)
            x = x + projected_residual

            # Activation and Dropout (except for the last layer)
            if i < len(self.model_layers) - 1:
                x = F.relu(x)

                if self.training and self.dropout_rate > 0:
                    x = self.dropout_layer(x)

        # Graph Pooling
        x = global_mean_pool(x, batch)

        return F.log_softmax(x, dim=1)  # Output in log_softmax format


class ResidualSAGEBlock(nn.Module):
    """
    Optional residual SAGE block implementation - a more modular design
    """

    def __init__(self, in_features, out_features, dropout_rate=0.1):
        super(ResidualSAGEBlock, self).__init__()

        self.sage_conv = SAGEConv(in_features, out_features)
        self.batch_norm = nn.BatchNorm1d(out_features)
        self.dropout = nn.Dropout(dropout_rate)

        # Projection layer for dimension alignment
        if in_features != out_features:
            self.projection = nn.Linear(in_features, out_features)
        else:
            self.projection = nn.Identity()

    def forward(self, x, edge_index):
        # Preserve residuals
        residual = x

        # Main path
        out = self.sage_conv(x, edge_index)
        out = self.batch_norm(out)

        # Residual Connection
        out = out + self.projection(residual)
        out = F.relu(out)
        out = self.dropout(out)

        return out


class ModularResidualSAGE(nn.Module):
    """
    SAGE network using modular residual blocks
    """

    def __init__(self, in_features, hidden_features, out_features,
                 num_layers=2, dropout_rate=0.1):
        super(ModularResidualSAGE, self).__init__()

        self.num_layers = num_layers

        # Constructing residual blocks
        self.blocks = nn.ModuleList()

        # First layer
        self.blocks.append(
            ResidualSAGEBlock(in_features, hidden_features, dropout_rate)
        )

        # Intermediate layer
        for _ in range(num_layers - 2):
            self.blocks.append(
                ResidualSAGEBlock(hidden_features, hidden_features, dropout_rate)
            )

        # Output layer (does not use residual blocks, uses convolution directly)
        if num_layers > 1:
            self.output_conv = SAGEConv(hidden_features, out_features)
            self.output_norm = nn.BatchNorm1d(out_features)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch

        # Through residual blocks
        for block in self.blocks:
            x = block(x, edge_index)

        # Output layer
        if self.num_layers > 1:
            x = self.output_conv(x, edge_index)
            x = self.output_norm(x)

        # Graph pooling
        x = global_mean_pool(x, batch)

        return F.log_softmax(x, dim=1)


if __name__ == "__main__":
    # Load TUDataset
    dataset = TUDataset(root='./data', name='PROTEINS')
    loader = DataLoader(dataset, batch_size=32, shuffle=True)

    # Select Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Testing two residual SAGE models
    print("=== Test integration residual SAGE model ===")
    model1 = ResidualSAGE_local(
        in_features=dataset.num_node_features,
        hidden_features=64,
        out_features=dataset.num_classes,
        num_layers=4  # Deeper networks better demonstrate the advantages of residual connections.
    ).to(device)

    print("=== Testing the modular residual SAGE model ===")
    model2 = ModularResidualSAGE(
        in_features=dataset.num_node_features,
        hidden_features=64,
        out_features=dataset.num_classes,
        num_layers=4
    ).to(device)

    # Get a batch of data
    data = next(iter(loader)).to(device)

    # Forward propagation
    out1 = model1(data)
    out2 = model2(data)

    # Print model information
    total_params1 = sum(p.numel() for p in model1.parameters() if p.requires_grad)
    total_params2 = sum(p.numel() for p in model2.parameters() if p.requires_grad)

    print(f"Integrating residual SAGE model parameter count: {total_params1}")
    print(f"Modular residual SAGE model parameter count: {total_params2}")
    print(f"Integrated model output shape: {out1.shape}")
    print(f"Modular model output shape: {out2.shape}")