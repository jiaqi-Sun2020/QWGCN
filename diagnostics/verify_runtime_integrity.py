"""Deterministic contract checks for the revised QWGCN runtime.

Run from the corrected repository root:
    python -B diagnostics/verify_runtime_integrity.py --output \
        ../_local_review/project2_runtime_integrity/output/verification_report.json

The real QWGCN4.py source is imported. Lightweight compatibility stubs replace
unavailable PyG/complexPyTorch packages so this check validates control flow,
tensor shapes, evolution configuration, and batch isolation without training.
"""

import argparse
import importlib.util
import json
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.nn as nn
import torch.nn.functional as F


PROJECT = Path(__file__).resolve().parents[1]
MODEL_PATH = PROJECT / "model" / "QWGCN4.py"
FACTORY_PATH = PROJECT / "model" / "model_get.py"


def install_dependency_stubs():
    """Install the minimal APIs used by QWGCN4.py for deterministic checks."""

    def add_self_loops(edge_index, num_nodes):
        nodes = torch.arange(num_nodes, device=edge_index.device)
        loops = torch.stack((nodes, nodes))
        return torch.cat((edge_index, loops), dim=1), None

    def degree(index, num_nodes, dtype=None):
        values = torch.bincount(index, minlength=num_nodes)
        return values.to(dtype=dtype or torch.float32)

    def global_mean_pool(x, batch):
        num_graphs = int(batch.max().item()) + 1 if batch.numel() else 0
        sums = torch.zeros(
            (num_graphs, x.size(-1)), dtype=x.dtype, device=x.device
        )
        sums.index_add_(0, batch, x)
        counts = torch.zeros(
            (num_graphs, 1), dtype=x.real.dtype, device=x.device
        )
        counts.index_add_(
            0, batch, torch.ones((x.size(0), 1), device=x.device)
        )
        return sums / counts.clamp_min(1)

    class ComplexLinear(nn.Module):
        def __init__(self, in_features, out_features):
            super().__init__()
            self.fc_r = nn.Linear(in_features, out_features)
            self.fc_i = nn.Linear(in_features, out_features)

        def forward(self, x):
            return torch.complex(
                self.fc_r(x.real) - self.fc_i(x.imag),
                self.fc_r(x.imag) + self.fc_i(x.real),
            )

    class ComplexBatchNorm1d(nn.Module):
        def __init__(self, features):
            super().__init__()
            self.weight = nn.Parameter(torch.ones(features, 3))
            self.bias = nn.Parameter(torch.zeros(features, 2))
            self.register_buffer(
                "running_mean", torch.zeros(features, dtype=torch.complex64)
            )
            self.register_buffer("running_covar", torch.zeros(features, 3))
            self.register_buffer(
                "num_batches_tracked", torch.tensor(0, dtype=torch.long)
            )

        def forward(self, x):
            return x

    def complex_relu(x):
        return torch.complex(F.relu(x.real), F.relu(x.imag))

    def complex_dropout(x, p):
        return torch.complex(F.dropout(x.real, p), F.dropout(x.imag, p))

    torch_geometric = types.ModuleType("torch_geometric")
    torch_geometric_nn = types.ModuleType("torch_geometric.nn")
    torch_geometric_utils = types.ModuleType("torch_geometric.utils")
    torch_geometric_nn.global_mean_pool = global_mean_pool
    torch_geometric_utils.degree = degree
    torch_geometric_utils.add_self_loops = add_self_loops
    torch_geometric.nn = torch_geometric_nn
    torch_geometric.utils = torch_geometric_utils

    complex_package = types.ModuleType("complexPyTorch")
    complex_layers = types.ModuleType("complexPyTorch.complexLayers")
    complex_functions = types.ModuleType("complexPyTorch.complexFunctions")
    complex_layers.ComplexLinear = ComplexLinear
    complex_layers.ComplexBatchNorm1d = ComplexBatchNorm1d
    complex_functions.complex_relu = complex_relu
    complex_functions.complex_dropout = complex_dropout

    torch_scatter = types.ModuleType("torch_scatter")
    torch_scatter.scatter_add = lambda src, index, dim=0: src.index_add(
        dim, index, src
    )

    sys.modules.update(
        {
            "torch_geometric": torch_geometric,
            "torch_geometric.nn": torch_geometric_nn,
            "torch_geometric.utils": torch_geometric_utils,
            "complexPyTorch": complex_package,
            "complexPyTorch.complexLayers": complex_layers,
            "complexPyTorch.complexFunctions": complex_functions,
            "torch_scatter": torch_scatter,
        }
    )


def load_model_module():
    install_dependency_stubs()
    spec = importlib.util.spec_from_file_location("qwgcn4_contract", MODEL_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def undirected_edges(pairs):
    edges = []
    for source, target in pairs:
        edges.extend(((source, target), (target, source)))
    return torch.tensor(edges, dtype=torch.long).t().contiguous()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path)
    parser.add_argument("--checkpoint", type=Path)
    args = parser.parse_args()

    torch.manual_seed(20260907)
    model_module = load_model_module()

    edge_a = undirected_edges([(0, 1)])
    x_a = torch.tensor(
        [[1.0 + 0.2j, 0.4 - 0.1j], [0.3 + 0.7j, 1.2 + 0.5j]],
        dtype=torch.complex64,
    )
    batch_a = torch.zeros(2, dtype=torch.long)

    first_order = model_module.FastQuantumEvolution(evolution_order=1)
    second_order = model_module.FastQuantumEvolution(evolution_order=2)
    weights_first, evolved_first = first_order(x_a, edge_a, batch_a)
    weights_second, evolved_second = second_order(x_a, edge_a, batch_a)

    assert first_order.evolution_order == 1
    assert second_order.evolution_order == 2
    assert not torch.allclose(evolved_first, evolved_second)
    assert dict(second_order.named_parameters()) == {}
    assert set(dict(second_order.named_buffers())) == {
        "evolution_time",
        "diffusion_strength",
    }
    assert second_order.state_dict() == {}
    assert second_order.evolution_time.item() == 1.0
    assert second_order.diffusion_strength.item() == 1.0

    legacy_result = second_order.load_state_dict(
        {
            "evolution_time": torch.tensor(0.37),
            "diffusion_strength": torch.tensor(1.42),
        },
        strict=True,
    )
    assert not legacy_result.missing_keys and not legacy_result.unexpected_keys
    assert second_order.evolution_time.item() == 1.0
    assert second_order.diffusion_strength.item() == 1.0

    layer = model_module.FastQWGNNLayer(2, 3, evolution_order=1)
    assert layer.quantum_evolution.evolution_order == 1

    # Batch isolation: graph A must keep identical normalized weights when an
    # unrelated, differently scaled graph B is added to the same batch.
    edge_b_local = undirected_edges([(0, 1), (1, 2)])
    edge_b = edge_b_local + 2
    edge_combined = torch.cat((edge_a, edge_b), dim=1)
    x_b = 20 * torch.tensor(
        [[1.0 + 0.0j, 2.0 + 1.0j], [0.2 + 0.4j, 1.0 - 0.3j],
         [1.5 - 0.2j, 0.1 + 0.7j]],
        dtype=torch.complex64,
    )
    x_combined = torch.cat((x_a, x_b))
    batch_combined = torch.tensor([0, 0, 1, 1, 1], dtype=torch.long)
    weights_a, _ = second_order(x_a, edge_a, batch_a)
    weights_combined, _ = second_order(
        x_combined, edge_combined, batch_combined
    )
    assert torch.allclose(weights_a, weights_combined[:2], atol=1e-6)
    assert torch.allclose(weights_combined[:2].sum(), torch.tensor(2.0))
    assert torch.allclose(weights_combined[2:].sum(), torch.tensor(3.0))

    data = SimpleNamespace(
        x=x_combined.real,
        edge_index=edge_combined,
        batch=batch_combined,
    )
    full_model = model_module.QWGCN_Fast(
        in_features=2,
        hidden_features=4,
        out_features=2,
        num_layers=2,
        evolution_order=2,
        dropout_rate=0.0,
    ).eval()
    output = full_model(data)
    trainable_parameter_count = sum(
        parameter.numel() for parameter in full_model.parameters()
    )
    assert output.shape == (2, 2)
    assert torch.isfinite(output).all()
    assert torch.allclose(output.exp().sum(dim=1), torch.ones(2), atol=1e-6)
    assert all(
        block.quantum_evolution.evolution_order == 2
        for block in full_model.quantum_layers
    )
    assert not any(
        name.endswith(("evolution_time", "diffusion_strength"))
        for name, _ in full_model.named_parameters()
    )

    single_layer = model_module.QWGCN_Fast(
        in_features=2,
        hidden_features=4,
        out_features=2,
        num_layers=1,
        evolution_order=2,
        dropout_rate=0.0,
    ).eval()
    single_output = single_layer(data)
    assert single_output.shape == (2, 2)

    factory_source = FACTORY_PATH.read_text(encoding="utf-8-sig")
    assert 'getattr(args, "evolution_order", 2)' in factory_source
    assert 'getattr(args, "evolution_time", 1.0)' in factory_source
    assert 'getattr(args, "diffusion_strength", 1.0)' in factory_source

    checkpoint_result = "not requested"
    if args.checkpoint:
        checkpoint = torch.load(
            args.checkpoint, map_location="cpu", weights_only=True
        )
        checkpoint_state = checkpoint.get("model_state_dict", checkpoint)
        compatible_model = model_module.QWGCN_Fast(
            in_features=3,
            hidden_features=64,
            out_features=2,
            num_layers=2,
            max_hops=2,
            evolution_order=2,
            evolution_time=1.0,
            diffusion_strength=1.0,
            dropout_rate=0.0,
        )
        load_result = compatible_model.load_state_dict(
            checkpoint_state, strict=True
        )
        compatible_parameter_count = sum(
            parameter.numel() for parameter in compatible_model.parameters()
        )
        assert not load_result.missing_keys
        assert not load_result.unexpected_keys
        assert compatible_parameter_count == 1880
        assert all(
            block.quantum_evolution.evolution_time.item() == 1.0
            and block.quantum_evolution.diffusion_strength.item() == 1.0
            for block in compatible_model.quantum_layers
        )
        checkpoint_result = {
            "path": str(args.checkpoint.resolve()),
            "strict_load": True,
            "legacy_trainable_dynamics_discarded": True,
            "fixed_evolution_time_after_load": 1.0,
            "fixed_diffusion_strength_after_load": 1.0,
            "historical_trainable_parameter_count": 1884,
            "revised_trainable_parameter_count": compatible_parameter_count,
            "removed_trainable_dynamics_parameters": 4,
        }

    report = {
        "status": "PASS",
        "source": str(MODEL_PATH),
        "torch": torch.__version__,
        "checks": {
            "requested_order_reaches_evolution_module": True,
            "paper_defaults": {
                "evolution_order": 2,
                "evolution_time": 1.0,
                "diffusion_strength": 1.0,
                "dynamics_are_trainable": False,
                "dynamics_are_serialized": False,
            },
            "full_forward_output_shape": list(output.shape),
            "fixture_trainable_parameter_count": trainable_parameter_count,
            "single_layer_output_shape": list(single_output.shape),
            "log_probability_rows_sum_to_one": True,
            "batch_isolation_max_abs_delta": float(
                (weights_a - weights_combined[:2]).abs().max()
            ),
            "per_graph_weight_sums": [
                float(weights_combined[:2].sum()),
                float(weights_combined[2:].sum()),
            ],
            "first_vs_second_order_max_abs_delta": float(
                (evolved_first - evolved_second).abs().max()
            ),
            "legacy_checkpoint_compatibility": checkpoint_result,
        },
        "limitation": (
            "PyG and complexPyTorch compatibility APIs are stubbed; this is a "
            "runtime-contract test, not a training or library-integration test."
        ),
    }

    rendered = json.dumps(report, indent=2, ensure_ascii=False)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()
