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
from torch_geometric.nn import GCNConv, GATConv, GraphConv, global_mean_pool, global_max_pool, global_add_pool
from torch_geometric.nn import BatchNorm, LayerNorm
import torch.nn as nn
import math


class GAT_And_GCN_layer(torch.nn.Module):
    def __init__(self, in_features=3, out_features=6):
        super().__init__()

        # Multi-scale feature extraction
        self.input_dim = in_features
        self.hidden_dim = 128
        self.num_heads = 8

        # Input projection layer
        self.input_projection = nn.Linear(in_features, self.hidden_dim)
        self.input_norm = BatchNorm(self.hidden_dim)

        # Multi-layer attention mechanism + GCN hybrid architecture
        self.gat1 = GATConv(self.hidden_dim, self.hidden_dim // self.num_heads,
                            heads=self.num_heads, dropout=0.1, concat=True)
        self.bn1 = BatchNorm(self.hidden_dim)

        self.gcn1 = GCNConv(self.hidden_dim, self.hidden_dim)
        self.gcn_bn1 = BatchNorm(self.hidden_dim)

        self.gat2 = GATConv(self.hidden_dim, self.hidden_dim // self.num_heads,
                            heads=self.num_heads, dropout=0.1, concat=True)
        self.bn2 = BatchNorm(self.hidden_dim)

        self.gcn2 = GCNConv(self.hidden_dim, self.hidden_dim)
        self.gcn_bn2 = BatchNorm(self.hidden_dim)

        # Deep feature extraction
        self.conv3 = GraphConv(self.hidden_dim, self.hidden_dim)
        self.bn3 = BatchNorm(self.hidden_dim)

        self.conv4 = GCNConv(self.hidden_dim, self.hidden_dim // 2)
        self.bn4 = BatchNorm(self.hidden_dim // 2)

        # Linear layers with residual connections
        self.residual_proj = nn.Linear(self.hidden_dim, self.hidden_dim // 2)

        # Multi-scale pooling
        self.pool_projection = nn.Linear(self.hidden_dim // 2 * 3, self.hidden_dim)

        # Advanced Classification Header - Using Attention Mechanism
        self.attention_weights = nn.Linear(self.hidden_dim, 1)

        self.classifier = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(self.hidden_dim // 2),
            nn.Dropout(0.3),

            nn.Linear(self.hidden_dim // 2, self.hidden_dim // 4),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(self.hidden_dim // 4),
            nn.Dropout(0.2),

            nn.Linear(self.hidden_dim // 4, self.hidden_dim // 8),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),

            nn.Linear(self.hidden_dim // 8, out_features)
        )

        # Weight initialization
        self._initialize_weights()

    def _initialize_weights(self):
        """Xavier/He initialization"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                if m.out_features == self.input_dim:
                    nn.init.xavier_uniform_(m.weight)
                else:
                    nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch

        # Input projection and normalization
        x = self.input_projection(x)
        x = F.relu(self.input_norm(x))
        identity1 = x

        # First layer: GAT + residual connections
        x_gat1 = self.gat1(x, edge_index)
        x_gat1 = F.elu(self.bn1(x_gat1))
        x_gat1 = F.dropout(x_gat1, p=0.1, training=self.training)

        # First level: GCN branch
        x_gcn1 = self.gcn1(identity1, edge_index)
        x_gcn1 = F.elu(self.gcn_bn1(x_gcn1))

        # Feature Fusion
        x = x_gat1 + x_gcn1 + identity1  # Three-way residual connection
        identity2 = x

        # Second layer: GAT
        x_gat2 = self.gat2(x, edge_index)
        x_gat2 = F.elu(self.bn2(x_gat2))
        x_gat2 = F.dropout(x_gat2, p=0.1, training=self.training)

        # Second level: GCN branch
        x_gcn2 = self.gcn2(identity2, edge_index)
        x_gcn2 = F.elu(self.gcn_bn2(x_gcn2))

        # Feature Fusion
        x = x_gat2 + x_gcn2 + identity2

        # Third layer: GraphConv
        x = self.conv3(x, edge_index)
        x = F.elu(self.bn3(x))
        x = F.dropout(x, p=0.15, training=self.training)

        # Fourth layer: Final convolution + residual
        residual = self.residual_proj(identity2)
        x = self.conv4(x, edge_index)
        x = F.elu(self.bn4(x))
        x = x + residual  # Residual Connection

        # Multi-scale graph pooling
        x_mean = global_mean_pool(x, batch)
        x_max = global_max_pool(x, batch)
        x_sum = global_add_pool(x, batch)

        # Pooling Feature Fusion
        x_pooled = torch.cat([x_mean, x_max, x_sum], dim=1)
        x_pooled = self.pool_projection(x_pooled)
        x_pooled = F.relu(x_pooled)
        x_pooled = F.dropout(x_pooled, p=0.2, training=self.training)

        # Attention Weighting (Optional Global Attention Mechanism)
        attention_scores = torch.sigmoid(self.attention_weights(x_pooled))
        x_pooled = x_pooled * attention_scores

        # Classification
        x = self.classifier(x_pooled)

        return F.log_softmax(x, dim=1)

if __name__ == "__main__":

    model = GAT_And_GCN_layer(in_features=2, out_features=2)
    print(model)