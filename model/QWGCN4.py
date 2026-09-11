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

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import global_mean_pool
from torch_geometric.utils import degree, add_self_loops
from torch_scatter import scatter_add
import math

from complexPyTorch.complexLayers import ComplexLinear, ComplexBatchNorm1d
from complexPyTorch.complexFunctions import complex_relu, complex_dropout

"""
Highly Efficient and Optimized Quantum Graph Neural Networks

Core Optimization Strategies:

1. Batch Processing: Avoiding node-by-node iterative computation

2. Sparse Computation: Using sparse matrix operations

3. Simplified Quantum Evolution: Using low-order approximations and pre-computation

4. Caching Mechanism: Reusing computation results

5. Parallelization: Leveraging the parallel computing capabilities of GPUs

Physical convention used by the corrected implementation:

- Graph operator: A_tilde is the symmetrically normalized adjacency with self-loops

- Time evolution: G(t) = exp(-i A_tilde t)
  ≈ I - i A_tilde t - t² A_tilde² / 2

- Quantum State Evolution: |ψ(t)⟩ = G(t)|ψ(0)⟩
"""


class FastQuantumEvolution(nn.Module):
    """
    Fast Quantum Evolution Module - Significantly Optimized Computational Efficiency
    """

    def __init__(self, evolution_order=2, evolution_time=1.0, diffusion_strength=1.0):
        super(FastQuantumEvolution, self).__init__()

        if evolution_order not in (1, 2):
            raise ValueError("evolution_order must be 1 or 2")
        if evolution_time < 0:
            raise ValueError("evolution_time must be non-negative")
        if diffusion_strength < 0:
            raise ValueError("diffusion_strength must be non-negative")

        self.evolution_order = evolution_order

        # The manuscript protocol fixes t=1 and diffusion=1. Non-persistent
        # buffers follow the module device without changing legacy checkpoints.
        self.register_buffer(
            "evolution_time", torch.tensor(float(evolution_time)), persistent=False
        )
        self.register_buffer(
            "diffusion_strength",
            torch.tensor(float(diffusion_strength)),
            persistent=False,
        )

        # For numerical stability
        self.eps = 1e-8

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        # Historical checkpoints stored these values as learnable parameters.
        # Discard those keys so loading cannot silently change the fixed protocol.
        state_dict.pop(prefix + "evolution_time", None)
        state_dict.pop(prefix + "diffusion_strength", None)
        super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )

    def forward(self, x_complex, edge_index, batch=None):
        """
        Rapid quantum evolution in batches
        """
        num_nodes = x_complex.size(0)
        # Add self-loops to ensure connectivity
        edge_index_with_loops, _ = add_self_loops(edge_index, num_nodes=num_nodes)

        row, col = edge_index_with_loops

        # Calculate the degree and normalize it.
        deg = degree(col, num_nodes, dtype=torch.float)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0

        # Normalized weights
        norm_weights = deg_inv_sqrt[row] * deg_inv_sqrt[col]

        # First-order graph term: A_tilde x
        # Use sparse matrix multiplication to avoid dense computation
        messages_1 = x_complex[col] * norm_weights.unsqueeze(-1)
        first_order = torch.zeros_like(x_complex)
        first_order.index_add_(0, row, messages_1)

        # Corrected Taylor convention: G(t) ≈ I - i t d A_tilde.
        evolved_x = x_complex - 1j * self.evolution_time * self.diffusion_strength * first_order

        if self.evolution_order >= 2:
            # (-i t d A_tilde)^2 x / 2 = -t^2 d^2 A_tilde^2 x / 2.
            messages_2 = first_order[col] * norm_weights.unsqueeze(-1)
            second_order = torch.zeros_like(x_complex)
            second_order.index_add_(0, row, messages_2)

            second_coeff = (-1j * self.evolution_time * self.diffusion_strength) ** 2 / 2.0
            evolved_x = evolved_x + second_coeff * second_order

        # Calculate the evolution weights: |ψ(t)|²
        evolution_weights = (evolved_x * evolved_x.conj()).real.sum(dim=-1, keepdim=True)

        # Normalize independently inside every graph in the batch. This prevents
        # one graph from changing another graph's output through a global sum.
        if batch is None:
            weight_sum = evolution_weights.sum()
            safe_sum = weight_sum.clamp_min(self.eps)
            normalized = evolution_weights / safe_sum * num_nodes
            evolution_weights = torch.where(
                weight_sum > self.eps,
                normalized,
                torch.ones_like(evolution_weights),
            )
        else:
            if batch.dim() != 1 or batch.numel() != num_nodes:
                raise ValueError("batch must contain one graph index per node")

            batch = batch.to(device=evolution_weights.device, dtype=torch.long)
            if batch.numel() == 0:
                return evolution_weights, evolved_x

            num_graphs = int(batch.max().item()) + 1
            graph_weight_sum = torch.zeros(
                (num_graphs, 1),
                dtype=evolution_weights.dtype,
                device=evolution_weights.device,
            )
            graph_weight_sum.index_add_(0, batch, evolution_weights)

            graph_node_count = torch.zeros_like(graph_weight_sum)
            graph_node_count.index_add_(0, batch, torch.ones_like(evolution_weights))

            per_node_sum = graph_weight_sum[batch]
            per_node_count = graph_node_count[batch]
            normalized = (
                evolution_weights
                / per_node_sum.clamp_min(self.eps)
                * per_node_count
            )
            evolution_weights = torch.where(
                per_node_sum > self.eps,
                normalized,
                torch.ones_like(evolution_weights),
            )

        return evolution_weights, evolved_x


class EfficientMultiHop(nn.Module):
    """
    Efficient multi-hop aggregation - using matrix exponentiation
    """

    def __init__(self, max_hops=2):
        super(EfficientMultiHop, self).__init__()

        self.max_hops = max_hops
        self.hop_weights = nn.Parameter(torch.ones(max_hops + 1))

    def forward(self, x_complex, edge_index):
        """
        Fast multi-hop aggregation based on matrix powers
        """
        num_nodes = x_complex.size(0)
        device = x_complex.device

        # Add self-loop
        edge_index_with_loops, _ = add_self_loops(edge_index, num_nodes=num_nodes)
        row, col = edge_index_with_loops

        # Degree normalization
        deg = degree(col, num_nodes, dtype=torch.float)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
        norm_weights = deg_inv_sqrt[row] * deg_inv_sqrt[col]

        # 存Store results with different hop counts
        hop_results = [x_complex]  # 0跳
        current_x = x_complex

        # hop-by-hop propagation (reusing intermediate results)
        for hop in range(1, self.max_hops + 1):
            messages = current_x[col] * norm_weights.unsqueeze(-1)
            next_x = torch.zeros_like(x_complex)
            next_x.index_add_(0, row, messages)
            current_x = next_x
            hop_results.append(current_x)

        # Weighted combination
        hop_weights_norm = F.softmax(self.hop_weights, dim=0)

        result = torch.zeros_like(x_complex)
        for hop, hop_result in enumerate(hop_results):
            result += hop_weights_norm[hop] * hop_result

        return result


class FastQWGNNLayer(nn.Module):
    """
    Fast QWGNN Layers - Optimized Version
    """

    def __init__(self, in_features, out_features, max_hops=2,
                 evolution_order=2, evolution_time=1.0, diffusion_strength=1.0):
        super(FastQWGNNLayer, self).__init__()

        self.in_features = in_features
        self.out_features = out_features

        # Rapid quantum evolution (reducing order)
        self.quantum_evolution = FastQuantumEvolution(
            evolution_order=evolution_order,
            evolution_time=evolution_time,
            diffusion_strength=diffusion_strength,
        )
        # High-efficiency multi-hop aggregation
        self.multi_hop = EfficientMultiHop(max_hops)

        # ComplexFC layer
        self.complex_fc = ComplexLinear(in_features, out_features)

        # Residual Connection
        if in_features != out_features:
            self.residual_proj = ComplexLinear(in_features, out_features)
        else:
            self.residual_proj = None

    def forward(self, x_complex, edge_index, batch=None):
        """
        Fast forward propagation
        """
        # Preserve residuals
        residual = x_complex

        # Rapid Quantum Evolution
        evolution_weights, evolved_features = self.quantum_evolution(
            x_complex, edge_index, batch
        )

        # Multi-hop aggregation
        aggregated_features = self.multi_hop(evolved_features, edge_index)

        # Weighting Application
        weighted_features = evolution_weights * aggregated_features

        # Feature Transformation
        output = self.complex_fc(weighted_features)

        # Residual Connection
        if self.residual_proj is not None:
            residual = self.residual_proj(residual)

        output = output + residual

        return output


class QWGCN_Fast(nn.Module):
    """
    Fast Quantum Graph Neural Network - High-Efficiency Version
    """

    def __init__(self, in_features, hidden_features, out_features,
                 num_layers=2, max_hops=1, evolution_order=2,
                 evolution_time=1.0, diffusion_strength=1.0,
                 dropout_rate=0.1):
        super(QWGCN_Fast, self).__init__()

        if num_layers < 1:
            raise ValueError("num_layers must be at least 1")

        self.num_layers = num_layers
        self.evolution_order = evolution_order
        self.evolution_time = float(evolution_time)
        self.diffusion_strength = float(diffusion_strength)
        self.dropout_rate = dropout_rate

        # Build layer
        self.quantum_layers = nn.ModuleList()

        if num_layers == 1:
            self.quantum_layers.append(
                FastQWGNNLayer(
                    in_features, out_features, max_hops, evolution_order,
                    evolution_time, diffusion_strength,
                )
            )
        else:
            # Input layer
            self.quantum_layers.append(
                FastQWGNNLayer(
                    in_features, hidden_features, max_hops, evolution_order,
                    evolution_time, diffusion_strength,
                )
            )

            # Hidden layers
            for _ in range(num_layers - 2):
                self.quantum_layers.append(
                    FastQWGNNLayer(
                        hidden_features, hidden_features, max_hops,
                        evolution_order, evolution_time, diffusion_strength,
                    )
                )

            # Output layer
            self.quantum_layers.append(
                FastQWGNNLayer(
                    hidden_features, out_features, max_hops, evolution_order,
                    evolution_time, diffusion_strength,
                )
            )

        # Normalization
        self.batch_norms = nn.ModuleList()
        for i in range(len(self.quantum_layers)):
            if i < num_layers - 1:
                self.batch_norms.append(ComplexBatchNorm1d(hidden_features))
            else:
                self.batch_norms.append(ComplexBatchNorm1d(out_features))

    def forward(self, data):
        """Fast forward propagation"""
        x, edge_index, batch = data.x, data.edge_index, data.batch

        # Convert to complex number
        x_real = x.float()
        x_complex = torch.complex(x_real, torch.zeros_like(x_real))

        # Processing layer by layer
        for i, (quantum_layer, batch_norm) in enumerate(zip(self.quantum_layers, self.batch_norms)):
            # Quantum Evolution
            x_complex = quantum_layer(x_complex, edge_index, batch)

            # Normalization
            x_complex = batch_norm(x_complex)

            # Activation and Dropout (except for the last layer)
            if i < len(self.quantum_layers) - 1:
                x_complex = complex_relu(x_complex)

                if self.training and self.dropout_rate > 0:
                    x_complex = complex_dropout(x_complex, self.dropout_rate)

        # Convert to real number
        x = x_complex.abs()

        # Graph Pooling
        x = global_mean_pool(x, batch)

        return F.log_softmax(x, dim=1)


class AdaptiveQWGNN(nn.Module):
    """
    Adaptive Quantum GNN - Adjusting computational complexity based on graph size
    """

    def __init__(self, in_features, hidden_features, out_features,
                 num_layers=2, dropout_rate=0.1):
        super(AdaptiveQWGNN, self).__init__()

        self.in_features = in_features
        self.hidden_features = hidden_features
        self.out_features = out_features
        self.num_layers = num_layers
        self.dropout_rate = dropout_rate

        # Predefined models of varying complexity
        self.models = nn.ModuleDict({
            'simple': QWGCN_Fast(in_features, hidden_features, out_features,
                                 num_layers, max_hops=1, evolution_order=1, dropout_rate=dropout_rate),
            'medium': QWGCN_Fast(in_features, hidden_features, out_features,
                                 num_layers, max_hops=2, evolution_order=1, dropout_rate=dropout_rate),
            'complex': QWGCN_Fast(in_features, hidden_features, out_features,
                                  num_layers, max_hops=2, evolution_order=2, dropout_rate=dropout_rate)
        })

    def forward(self, data):
        """The model complexity is adaptively selected based on the graph size."""
        num_nodes = data.x.size(0)
        num_edges = data.edge_index.size(1)

        # Select the model based on the image size.
        if num_nodes < 100:
            return self.models['complex'](data)
        elif num_nodes < 1000:
            return self.models['medium'](data)
        else:
            return self.models['simple'](data)


def benchmark_comparison():
    """Performance benchmark testing"""
    from torch_geometric.data import Data
    from torch_geometric.nn import GCNConv
    import time

    print("=== Performance comparison test after optimization ===")

    # Test different scales
    test_configs = [
        {"nodes": 100, "edges": 300, "name": "Small graph"},
        {"nodes": 500, "edges": 1500, "name": "Middle graph"},
        {"nodes": 1000, "edges": 3000, "name": "Large graph"}
    ]

    for config in test_configs:
        print(f"\n--- {config['name']} ({config['nodes']}node, {config['edges']}side) ---")

        # Create test data
        x = torch.randn(config['nodes'], 64)
        edge_index = torch.randint(0, config['nodes'], (2, config['edges']))
        edge_index = torch.unique(edge_index, dim=1)

        data = Data(x=x, edge_index=edge_index)
        data.batch = torch.zeros(config['nodes'], dtype=torch.long)

        # Define the contrast model
        class SimpleGCN(nn.Module):
            def __init__(self):
                super().__init__()
                self.conv1 = GCNConv(64, 128)
                self.conv2 = GCNConv(128, 10)

            def forward(self, data):
                x, edge_index, batch = data.x, data.edge_index, data.batch
                x = F.relu(self.conv1(x, edge_index))
                x = self.conv2(x, edge_index)
                x = global_mean_pool(x, batch)
                return F.log_softmax(x, dim=1)

        models = {
            "Standard GCN": SimpleGCN(),
            "Fast Quantum GNN": QWGCN_Fast(64, 128, 10, num_layers=2, max_hops=1, evolution_order=1),
            "Adaptive Quantum GNN": AdaptiveQWGNN(64, 128, 10, num_layers=2)
        }

        for name, model in models.items():
            try:
                model.eval()

                # Parameter statistics
                num_params = sum(p.numel() for p in model.parameters())

                # Performance testing
                with torch.no_grad():
                    # preheating
                    _ = model(data)

                    # Timing
                    start_time = time.time()
                    for _ in range(10):
                        output = model(data)
                    end_time = time.time()

                    avg_time = (end_time - start_time) / 10

                    print(f"  {name}:")
                    print(f"    parameter: {num_params:,}")
                    print(f"    time: {avg_time:.4f}s")
                    print(f"    Output: {output.shape}")

            except Exception as e:
                print(f"  {name}: fail - {str(e)}")


def test_optimizations():
    """Test optimization effect"""
    from torch_geometric.data import Data

    print("=== Optimization effect test ===")

    # Create test data
    x = torch.randn(500, 64)
    edge_index = torch.randint(0, 500, (2, 1500))
    edge_index = torch.unique(edge_index, dim=1)

    data = Data(x=x, edge_index=edge_index)
    data.batch = torch.zeros(500, dtype=torch.long)

    x_complex = torch.complex(x, torch.zeros_like(x))

    print("Test each optimization component:")

    # Testing rapid quantum evolution
    print("\n1. Fast Quantum Evolution Module")
    fast_evolution = FastQuantumEvolution(evolution_order=1)

    import time
    start = time.time()
    weights, evolved = fast_evolution(x_complex, edge_index)
    end = time.time()

    print(f"   ✓ Calculation time: {end - start:.4f}s")
    print(f"   ✓ Weight Shape: {weights.shape}")
    print(f"   ✓ Evolutionary features shape: {evolved.shape}")

    # Testing multi-hop aggregation
    print("\n2. High-efficiency multi-hop aggregation")
    multi_hop = EfficientMultiHop(max_hops=2)

    start = time.time()
    aggregated = multi_hop(x_complex, edge_index)
    end = time.time()

    print(f"   ✓ Calculation time: {end - start:.4f}s")
    print(f"   ✓ Aggregate feature shape: {aggregated.shape}")

    # Test the complete fast model
    print("\n3. Complete Fast Model")
    fast_model = QWGCN_Fast(64, 128, 10, num_layers=2, max_hops=1, evolution_order=1)
    fast_model.eval()

    with torch.no_grad():
        start = time.time()
        output = fast_model(data)
        end = time.time()

    print(f"   ✓ Total computation time: {end - start:.4f}s")
    print(f"   ✓ Output shape: {output.shape}")

    print("\nOptimization test completed！")

    model_final = QWGCN_Fast(in_features=64,
                           hidden_features=128,
                           out_features=10,
                           num_layers=2,
                           max_hops=2,
                           evolution_order=1,
                           dropout_rate=0.1)

    print(model_final)



if __name__ == "__main__":
    # Test optimization effect
    test_optimizations()
    print("\n" + "=" * 60 + "\n")

    # Performance benchmark testing
    benchmark_comparison()
