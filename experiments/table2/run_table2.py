"""Run a reproducible, paired-seed Table II benchmark on full PROTEINS.

The legacy training path cannot reproduce Table II because it lacks an
independent-run manifest and tests the terminal model rather than restoring the
best validation checkpoint.  This isolated runner makes those contracts
explicit without changing the deferred PROTEINS_SMALL split implementation.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import platform
import random
import sys
import time
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import confusion_matrix, f1_score, mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split
from torch import nn
from torch_geometric.datasets import TUDataset
from torch_geometric.loader import DataLoader


PROJECT_AI_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_AI_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_AI_ROOT))

from model.QWGCN4 import QWGCN_Fast  # noqa: E402
from model.normal_network_local.GAT import GAT_local  # noqa: E402
from model.normal_network_local.GCN import GCN_local  # noqa: E402
from model.normal_network_local.GraphSAGE import SAGE_local  # noqa: E402
from model.normal_network_local.Res_SAGE import ResidualSAGE_local  # noqa: E402
from model.normal_network_local.TransformerConv import Transformer_local  # noqa: E402
from model.normal_network_local.multi_GAT import MultiGAT_local  # noqa: E402


PAPER_MODELS = [
    "GAT",
    "GCN",
    "GraphSAGE",
    "multi_GAT",
    "TransformerConv",
    "Res_SAGE",
    "QWGCN",
]

MODEL_SOURCE_FILES = {
    "GAT": PROJECT_AI_ROOT / "model/normal_network_local/GAT.py",
    "GCN": PROJECT_AI_ROOT / "model/normal_network_local/GCN.py",
    "GraphSAGE": PROJECT_AI_ROOT / "model/normal_network_local/GraphSAGE.py",
    "multi_GAT": PROJECT_AI_ROOT / "model/normal_network_local/multi_GAT.py",
    "TransformerConv": PROJECT_AI_ROOT / "model/normal_network_local/TransformerConv.py",
    "Res_SAGE": PROJECT_AI_ROOT / "model/normal_network_local/Res_SAGE.py",
    "QWGCN": PROJECT_AI_ROOT / "model/QWGCN4.py",
}


@dataclass(frozen=True)
class Protocol:
    dataset: str
    split_seed: int
    train_fraction: float
    validation_fraction: float
    test_fraction: float
    hidden_features: int
    num_layers: int
    multi_gat_heads: int
    qwgcn_max_hops: int
    qwgcn_evolution_order: int
    qwgcn_evolution_time: float
    qwgcn_diffusion_strength: float
    dropout: float
    learning_rate: float
    weight_decay: float
    batch_size: int
    epochs: int
    evaluate_every: int
    patience_evaluations: int
    optimizer: str
    scheduler: str
    checkpoint_selection: str
    metric_definition: str
    aggregation: str


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_hash(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def seed_everything(seed: int) -> None:
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True, warn_only=True)


def dataset_fingerprint(dataset: TUDataset) -> str:
    summary = [
        {
            "index": index,
            "label": int(graph.y.item()),
            "nodes": int(graph.num_nodes),
            "directed_edges": int(graph.edge_index.shape[1]),
            "feature_shape": list(graph.x.shape) if graph.x is not None else None,
        }
        for index, graph in enumerate(dataset)
    ]
    return canonical_hash(summary)


def stratified_split(dataset: TUDataset, split_seed: int) -> dict[str, list[int]]:
    indices = np.arange(len(dataset), dtype=int)
    labels = np.asarray([int(dataset[index].y.item()) for index in indices], dtype=int)
    train_idx, temp_idx = train_test_split(
        indices,
        test_size=0.30,
        random_state=split_seed,
        stratify=labels,
    )
    val_idx, test_idx = train_test_split(
        temp_idx,
        test_size=0.50,
        random_state=split_seed,
        stratify=labels[temp_idx],
    )
    return {
        "train": sorted(map(int, train_idx)),
        "validation": sorted(map(int, val_idx)),
        "test": sorted(map(int, test_idx)),
    }


def split_manifest(dataset: TUDataset, split: dict[str, list[int]], split_seed: int) -> dict[str, Any]:
    manifest: dict[str, Any] = {
        "policy": "single fixed stratified 70/15/15 split shared by every model and training seed",
        "split_seed": split_seed,
        "dataset_size": len(dataset),
        "dataset_fingerprint": dataset_fingerprint(dataset),
        "indices": split,
        "partitions": {},
    }
    all_indices: list[int] = []
    for name, indices in split.items():
        labels = [int(dataset[index].y.item()) for index in indices]
        manifest["partitions"][name] = {
            "count": len(indices),
            "label_counts": dict(sorted(Counter(labels).items())),
            "indices_sha256": canonical_hash(indices),
        }
        all_indices.extend(indices)
    if len(all_indices) != len(set(all_indices)) or sorted(all_indices) != list(range(len(dataset))):
        raise RuntimeError("Split integrity failure: partitions overlap or do not cover the dataset exactly.")
    return manifest


def make_model(model_name: str, dataset: TUDataset, protocol: Protocol) -> nn.Module:
    common = {
        "in_features": int(dataset.num_node_features),
        "hidden_features": protocol.hidden_features,
        "out_features": int(dataset.num_classes),
        "num_layers": protocol.num_layers,
        "dropout_rate": protocol.dropout,
    }
    factories: dict[str, Callable[[], nn.Module]] = {
        "GAT": lambda: GAT_local(**common),
        "GCN": lambda: GCN_local(**common),
        "GraphSAGE": lambda: SAGE_local(**common),
        "multi_GAT": lambda: MultiGAT_local(**common, heads=protocol.multi_gat_heads),
        "TransformerConv": lambda: Transformer_local(**common),
        "Res_SAGE": lambda: ResidualSAGE_local(**common),
        "QWGCN": lambda: QWGCN_Fast(
            **common,
            max_hops=protocol.qwgcn_max_hops,
            evolution_order=protocol.qwgcn_evolution_order,
            evolution_time=protocol.qwgcn_evolution_time,
            diffusion_strength=protocol.qwgcn_diffusion_strength,
        ),
    }
    return factories[model_name]()


def make_loaders(
    dataset: TUDataset,
    split: dict[str, list[int]],
    batch_size: int,
    training_seed: int,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    generator = torch.Generator()
    generator.manual_seed(training_seed)
    train_loader = DataLoader(
        [dataset[index] for index in split["train"]],
        batch_size=batch_size,
        shuffle=True,
        generator=generator,
        num_workers=0,
    )
    validation_loader = DataLoader(
        [dataset[index] for index in split["validation"]],
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
    )
    test_loader = DataLoader(
        [dataset[index] for index in split["test"]],
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
    )
    return train_loader, validation_loader, test_loader


def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> dict[str, Any]:
    model.eval()
    loss_sum = 0.0
    sample_count = 0
    predictions: list[torch.Tensor] = []
    positive_class_probabilities: list[torch.Tensor] = []
    labels: list[torch.Tensor] = []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            output = model(batch)
            if output.ndim != 2 or output.shape[1] != 2:
                raise RuntimeError(f"Table II requires binary class scores, received {tuple(output.shape)}")
            target = batch.y
            loss = F.nll_loss(output, target)
            count = int(target.numel())
            loss_sum += float(loss.item()) * count
            sample_count += count
            predictions.append(output.argmax(dim=1).cpu())
            positive_class_probabilities.append(output.exp()[:, 1].cpu())
            labels.append(target.cpu())

    pred = torch.cat(predictions).numpy()
    probability_class_1 = torch.cat(positive_class_probabilities).numpy()
    truth = torch.cat(labels).numpy()
    accuracy = float(np.mean(truth == pred))
    return {
        "loss": loss_sum / sample_count,
        "f1": float(f1_score(truth, pred, average="macro", zero_division=0)),
        # Probability MSE is the binary Brier score. The legacy hard-label MSE
        # equals 1-accuracy exactly and therefore carries no independent signal.
        "mse": float(mean_squared_error(truth, probability_class_1)),
        "mae": float(mean_absolute_error(truth, probability_class_1)),
        "accuracy": accuracy,
        "error_rate": 1.0 - accuracy,
        "sample_count": sample_count,
        "confusion_matrix": confusion_matrix(truth, pred).tolist(),
        "predictions": pred.tolist(),
        "positive_class_probabilities": probability_class_1.tolist(),
        "labels": truth.tolist(),
    }


def train_epoch(model: nn.Module, loader: DataLoader, optimizer: torch.optim.Optimizer, device: torch.device) -> dict[str, float]:
    model.train()
    loss_sum = 0.0
    correct = 0
    sample_count = 0
    for batch in loader:
        batch = batch.to(device)
        optimizer.zero_grad(set_to_none=True)
        output = model(batch)
        loss = F.nll_loss(output, batch.y)
        if not torch.isfinite(loss):
            raise FloatingPointError(f"Non-finite training loss: {loss.item()}")
        loss.backward()
        optimizer.step()
        count = int(batch.y.numel())
        loss_sum += float(loss.item()) * count
        correct += int((output.argmax(dim=1) == batch.y).sum().item())
        sample_count += count
    return {"loss": loss_sum / sample_count, "accuracy": correct / sample_count}


def state_dict_to_cpu(model: nn.Module) -> dict[str, torch.Tensor]:
    return {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}


def write_history(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_predictions(path: Path, indices: list[int], metrics: dict[str, Any]) -> None:
    rows = [
        {
            "dataset_index": index,
            "label": label,
            "prediction": prediction,
            "probability_class_1": probability,
            "correct": label == prediction,
        }
        for index, label, prediction, probability in zip(
            indices,
            metrics["labels"],
            metrics["predictions"],
            metrics["positive_class_probabilities"],
        )
    ]
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def public_metrics(metrics: dict[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in metrics.items()
        if key not in {"predictions", "positive_class_probabilities", "labels"}
    }


def run_one(
    model_name: str,
    training_seed: int,
    dataset: TUDataset,
    split: dict[str, list[int]],
    protocol: Protocol,
    protocol_hash: str,
    output_root: Path,
    device: torch.device,
) -> dict[str, Any]:
    run_dir = output_root / "runs" / model_name / f"seed_{training_seed}"
    run_dir.mkdir(parents=True, exist_ok=True)
    result_path = run_dir / "result.json"
    if result_path.exists():
        prior = json.loads(result_path.read_text(encoding="utf-8"))
        if prior.get("protocol_hash") != protocol_hash:
            raise RuntimeError(f"Existing run has a different protocol: {result_path}")
        print(f"SKIP completed {model_name} seed={training_seed}", flush=True)
        return prior

    seed_everything(training_seed)
    train_loader, validation_loader, test_loader = make_loaders(
        dataset, split, protocol.batch_size, training_seed
    )
    model = make_model(model_name, dataset, protocol).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=protocol.learning_rate, weight_decay=protocol.weight_decay
    )
    parameter_count = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)

    started = time.perf_counter()
    history: list[dict[str, Any]] = []
    best_state: dict[str, torch.Tensor] | None = None
    best_validation: dict[str, Any] | None = None
    best_epoch: int | None = None
    evaluations_without_improvement = 0
    final_epoch = 0

    for epoch in range(1, protocol.epochs + 1):
        training = train_epoch(model, train_loader, optimizer, device)
        final_epoch = epoch
        should_evaluate = epoch % protocol.evaluate_every == 0 or epoch == protocol.epochs
        if should_evaluate:
            validation = evaluate(model, validation_loader, device)
            row = {
                "epoch": epoch,
                "training_loss": training["loss"],
                "training_accuracy": training["accuracy"],
                "validation_loss": validation["loss"],
                "validation_f1": validation["f1"],
                "validation_mse": validation["mse"],
                "validation_mae": validation["mae"],
                "validation_accuracy": validation["accuracy"],
            }
            history.append(row)
            improved = (
                best_validation is None
                or validation["accuracy"] > best_validation["accuracy"] + 1e-12
                or (
                    abs(validation["accuracy"] - best_validation["accuracy"]) <= 1e-12
                    and validation["loss"] < best_validation["loss"] - 1e-12
                )
            )
            if improved:
                best_validation = public_metrics(validation)
                best_epoch = epoch
                best_state = state_dict_to_cpu(model)
                evaluations_without_improvement = 0
                torch.save(
                    {
                        "model_state_dict": best_state,
                        "epoch": epoch,
                        "validation": best_validation,
                        "training_seed": training_seed,
                        "protocol_hash": protocol_hash,
                    },
                    run_dir / "best.pt",
                )
            else:
                evaluations_without_improvement += 1

            print(
                f"{model_name} seed={training_seed} epoch={epoch}/{protocol.epochs} "
                f"val_acc={validation['accuracy']:.4f} best={best_validation['accuracy']:.4f}",
                flush=True,
            )
            if evaluations_without_improvement >= protocol.patience_evaluations:
                break

    if best_state is None or best_validation is None or best_epoch is None:
        raise RuntimeError("No validation evaluation was performed.")

    torch.save(
        {
            "model_state_dict": state_dict_to_cpu(model),
            "epoch": final_epoch,
            "training_seed": training_seed,
            "protocol_hash": protocol_hash,
        },
        run_dir / "last.pt",
    )
    model.load_state_dict(best_state)
    model.to(device)
    test_metrics = evaluate(model, test_loader, device)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    elapsed_seconds = time.perf_counter() - started

    write_history(run_dir / "history.csv", history)
    write_predictions(run_dir / "test_predictions.csv", split["test"], test_metrics)
    result = {
        "status": "complete",
        "model": model_name,
        "training_seed": training_seed,
        "split_seed": protocol.split_seed,
        "protocol_hash": protocol_hash,
        "parameter_count": parameter_count,
        "best_epoch": best_epoch,
        "final_epoch": final_epoch,
        "best_validation": best_validation,
        "test": public_metrics(test_metrics),
        "elapsed_seconds": elapsed_seconds,
        "device": str(device),
        "model_source": str(MODEL_SOURCE_FILES[model_name].resolve()),
        "model_source_sha256": sha256_file(MODEL_SOURCE_FILES[model_name]),
    }
    result_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    del model, optimizer
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print(
        f"DONE {model_name} seed={training_seed}: F1={test_metrics['f1']:.4f} "
        f"MSE={test_metrics['mse']:.4f} ACC={test_metrics['accuracy']:.4f}",
        flush=True,
    )
    return result


def aggregate_results(results: list[dict[str, Any]], models: list[str], seeds: list[int]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for model in models:
        model_results = sorted(
            [result for result in results if result["model"] == model],
            key=lambda result: result["training_seed"],
        )
        observed_seeds = [int(result["training_seed"]) for result in model_results]
        if observed_seeds != sorted(seeds):
            raise RuntimeError(f"Incomplete paired-seed results for {model}: {observed_seeds} != {sorted(seeds)}")
        row: dict[str, Any] = {
            "model": model,
            "run_count": len(model_results),
            "training_seeds": " ".join(map(str, observed_seeds)),
        }
        for field in ("f1", "mse", "mae", "accuracy", "error_rate"):
            values = np.asarray([result["test"][field] for result in model_results], dtype=float)
            row[f"{field}_mean"] = float(values.mean())
            row[f"{field}_sample_sd"] = float(values.std(ddof=1)) if len(values) >= 2 else None
        rows.append(row)
    return rows


def write_aggregate(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def environment_manifest(device: torch.device) -> dict[str, Any]:
    return {
        "python": sys.version,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "torch_geometric": __import__("torch_geometric").__version__,
        "numpy": np.__version__,
        "device": str(device),
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "deterministic_algorithms_warn_only": True,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, default=PROJECT_AI_ROOT / "dataset")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--models", nargs="+", choices=PAPER_MODELS, default=PAPER_MODELS)
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44, 45, 46])
    parser.add_argument("--split-seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=3000)
    parser.add_argument("--evaluate-every", type=int, default=20)
    parser.add_argument("--patience-evaluations", type=int, default=100)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=5e-4)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--hidden-features", type=int, default=64)
    parser.add_argument("--multi-gat-heads", type=int, default=2)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--smoke", action="store_true", help="Run one seed and two epochs for contract validation")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if len(set(args.seeds)) != len(args.seeds):
        raise ValueError("Training seeds must be unique.")
    if args.smoke:
        args.seeds = [args.seeds[0]]
        args.epochs = 2
        args.evaluate_every = 1
        args.patience_evaluations = 2
    if not 0.0 <= args.dropout < 1.0:
        raise ValueError("dropout must satisfy 0 <= dropout < 1")

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available.")

    protocol = Protocol(
        dataset="TUDataset/PROTEINS (all 1113 graphs)",
        split_seed=args.split_seed,
        train_fraction=0.70,
        validation_fraction=0.15,
        test_fraction=0.15,
        hidden_features=args.hidden_features,
        num_layers=2,
        multi_gat_heads=args.multi_gat_heads,
        qwgcn_max_hops=2,
        qwgcn_evolution_order=2,
        qwgcn_evolution_time=1.0,
        qwgcn_diffusion_strength=1.0,
        dropout=args.dropout,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        batch_size=args.batch_size,
        epochs=args.epochs,
        evaluate_every=args.evaluate_every,
        patience_evaluations=args.patience_evaluations,
        optimizer="AdamW",
        scheduler="none (constant learning rate; no scheduler was specified in the manuscript/config)",
        checkpoint_selection="highest validation accuracy; tie-break lower validation NLL; test best once",
        metric_definition=(
            "macro F1 and accuracy from argmax labels; MSE/MAE from positive-class probability "
            "(binary Brier-style error); error_rate stored separately as 1-accuracy"
        ),
        aggregation="arithmetic mean and sample standard deviation across paired training seeds (ddof=1)",
    )
    protocol_payload = {
        "protocol": asdict(protocol),
        "models": args.models,
        "training_seeds": sorted(args.seeds),
        "source_hashes": {model: sha256_file(MODEL_SOURCE_FILES[model]) for model in args.models},
        "runner_sha256": sha256_file(Path(__file__)),
    }
    protocol_hash = canonical_hash(protocol_payload)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = args.output_dir / "experiment_manifest.json"
    if manifest_path.exists():
        existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        if existing.get("protocol_hash") != protocol_hash:
            raise RuntimeError(
                f"Output directory already contains a different protocol: {manifest_path}. "
                "Use a new output directory."
            )

    dataset = TUDataset(root=str(args.dataset_root), name="PROTEINS")
    if len(dataset) != 1113:
        raise RuntimeError(f"Expected 1113 PROTEINS graphs, found {len(dataset)}")
    if int(dataset.num_node_features) <= 0 or int(dataset.num_classes) != 2:
        raise RuntimeError("Unexpected PROTEINS feature/class schema.")
    split = stratified_split(dataset, args.split_seed)
    split_info = split_manifest(dataset, split, args.split_seed)
    (args.output_dir / "split_manifest.json").write_text(
        json.dumps(split_info, indent=2) + "\n", encoding="utf-8"
    )
    experiment_manifest = {
        "schema_version": 1,
        "protocol_hash": protocol_hash,
        **protocol_payload,
        "dataset": {
            "root": str(args.dataset_root.resolve()),
            "graph_count": len(dataset),
            "num_node_features": int(dataset.num_node_features),
            "num_classes": int(dataset.num_classes),
            "fingerprint": split_info["dataset_fingerprint"],
        },
        "environment": environment_manifest(device),
        "smoke": bool(args.smoke),
    }
    manifest_path.write_text(json.dumps(experiment_manifest, indent=2) + "\n", encoding="utf-8")

    results: list[dict[str, Any]] = []
    for model_name in args.models:
        for seed in sorted(args.seeds):
            results.append(
                run_one(
                    model_name,
                    seed,
                    dataset,
                    split,
                    protocol,
                    protocol_hash,
                    args.output_dir,
                    device,
                )
            )

    aggregate = aggregate_results(results, args.models, sorted(args.seeds))
    write_aggregate(args.output_dir / "table2_aggregate.csv", aggregate)
    (args.output_dir / "table2_aggregate.json").write_text(
        json.dumps(
            {
                "protocol_hash": protocol_hash,
                "run_count": len(results),
                "models": aggregate,
                "note": "Sample SD is undefined for a one-seed smoke run and is therefore null.",
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"COMPLETE: {len(results)} runs, protocol={protocol_hash}", flush=True)


if __name__ == "__main__":
    main()
