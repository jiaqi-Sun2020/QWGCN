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
from torch_geometric.nn import GATConv, global_mean_pool
from torch_scatter import scatter_add

class HierarchicalAttentionPooling(torch.nn.Module):
    def __init__(self, in_channels, ratio=0.5):
        super().__init__()
        self.ratio = ratio
        self.structure_learning = torch.nn.Sequential(
            torch.nn.Linear(in_channels, 1),
            torch.nn.Sigmoid()
        )
        self.transform = torch.nn.Linear(in_channels, in_channels)

    def forward(self, x, edge_index, batch):
        node_scores = self.structure_learning(x).squeeze(-1)  # [N]

        num_nodes = scatter_add(torch.ones_like(batch), batch, dim=0)  # Batch_size Number of nodes in the graph
        batch_size = num_nodes.size(0)

        selected_nodes = []
        selected_features = []
        new_batch = []

        for i in range(batch_size):
            mask = batch == i
            graph_scores = node_scores[mask]
            graph_features = x[mask]

            if len(graph_scores) == 0:
                continue

            k = max(1, int(len(graph_scores) * self.ratio))
            top_indices = torch.topk(graph_scores, k, largest=True)[1]

            # Original image node index
            orig_indices = mask.nonzero(as_tuple=False).view(-1)[top_indices]

            selected_nodes.append(orig_indices)
            selected_features.append(graph_features[top_indices])
            new_batch.extend([i] * k)

        if len(selected_features) == 0:
            return x, edge_index, batch

        selected_nodes = torch.cat(selected_nodes)
        pooled_x = torch.cat(selected_features, dim=0)
        pooled_x = self.transform(pooled_x)
        new_batch = torch.tensor(new_batch, device=x.device, dtype=torch.long)

        # rebuild edge_index_pool
        node_id_map = -1 * torch.ones(x.size(0), dtype=torch.long, device=x.device)
        node_id_map[selected_nodes] = torch.arange(len(selected_nodes), device=x.device)

        row, col = edge_index
        mask = (node_id_map[row] >= 0) & (node_id_map[col] >= 0)
        row = node_id_map[row[mask]]
        col = node_id_map[col[mask]]
        edge_index_pool = torch.stack([row, col], dim=0)

        return pooled_x, edge_index_pool, new_batch


class HGP_SL_layer(torch.nn.Module):
    def __init__(self, in_features, hidden_dim=64, out_features=2, heads=4, pool_ratio=0.5):
        super().__init__()
        self.gat1 = GATConv(in_features, hidden_dim, heads=heads, concat=True, dropout=0.6)
        self.gat2 = GATConv(hidden_dim * heads, hidden_dim, heads=heads, concat=True, dropout=0.6)

        self.structure_learning = torch.nn.Sequential(
            torch.nn.Linear(hidden_dim * heads, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, 1),
            torch.nn.Sigmoid()
        )

        self.pool1 = HierarchicalAttentionPooling(hidden_dim * heads, ratio=pool_ratio)
        self.pool2 = HierarchicalAttentionPooling(hidden_dim * heads, ratio=pool_ratio)

        self.classifier = torch.nn.Sequential(
            torch.nn.Linear(hidden_dim * heads, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.5),
            torch.nn.Linear(hidden_dim, out_features)
        )

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch

        # Multi-layer GAT for extracting node features
        x = self.gat1(x, edge_index)
        x = F.elu(x)
        x = F.dropout(x, p=0.6, training=self.training)

        x = self.gat2(x, edge_index)
        x = F.elu(x)

        # The structure learning module calculates node importance (which can be used for pooling).
        importance = self.structure_learning(x).squeeze(-1)

        # First-level pooling
        x, edge_index, batch = self.pool1(x, edge_index, batch)

        # Second-level pooling
        x, edge_index, batch = self.pool2(x, edge_index, batch)

        # Classification after global pooling
        x = global_mean_pool(x, batch)
        x = self.classifier(x)
        x = F.log_softmax(x, dim=1)
        return x


if __name__ == "__main__":
    model = HGP_SL_layer(in_features=2, out_features=2)
    print(model)
    print(f"参数数量: {sum(p.numel() for p in model.parameters()):,}")