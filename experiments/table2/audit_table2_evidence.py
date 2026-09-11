"""Audit whether archived QWGCN runs reproduce manuscript Table II.

This module deliberately separates *observed evidence* from *reported claims*.
It never aggregates checkpoints from one optimization trajectory as if they were
independent runs and never combines different MultiGAT architectures.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np


MODEL_LABELS = {
    "GAT": "GAT_local",
    "GCN": "GCN_local",
    "GraphSAGE": "GraphSAGE_local",
    "multi_GAT": "MultiGAT_local",
    "TransformerConv": "TransformerConv_local",
    "Res_SAGE": "Res_SAGE_local",
    "QWGCN": "QWGCN",
}

DISPLAY_ORDER = [
    "GAT",
    "GCN",
    "GraphSAGE",
    "multi_GAT",
    "TransformerConv",
    "Res_SAGE",
    "QWGCN",
]


@dataclass(frozen=True)
class PaperValue:
    model: str
    f1_mean: float
    f1_sd: float
    mse_mean: float
    mse_sd: float
    acc_mean: float
    acc_sd: float


@dataclass(frozen=True)
class ArchivedRun:
    model: str
    configured_model: str
    variant: str
    log_path: str
    log_sha256: str
    has_final_test: bool
    f1: float | None
    mse: float | None
    mae: float | None
    acc: float | None
    parameter_count: int | None
    attention_heads: int | None
    seed_documented: bool
    final_epoch: int | None
    best_val_acc: float | None
    best_val_epoch: int | None
    final_model_is_best_checkpoint: bool


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def strip_tex_markup(value: str) -> str:
    value = value.replace(r"\textbf{", "").replace("}", "")
    value = value.replace(r"\_", "_")
    return value.strip()


def parse_metric_cell(cell: str) -> tuple[float, float]:
    numbers = re.findall(r"\d+\.\d+", strip_tex_markup(cell))
    if len(numbers) != 2:
        raise ValueError(f"Expected mean and SD in Table II cell: {cell!r}")
    return float(numbers[0]), float(numbers[1])


def parse_paper_table(tex_path: Path) -> list[PaperValue]:
    text = tex_path.read_text(encoding="utf-8")
    match = re.search(
        r"\\label\{tab:local_model_and_compare\}.*?\\begin\{tabular\}.*?"
        r"(?P<body>.*?)\\end\{tabular\}",
        text,
        flags=re.DOTALL,
    )
    if not match:
        raise ValueError("Could not locate Table II (tab:local_model_and_compare).")

    rows: list[PaperValue] = []
    for raw_line in match.group("body").splitlines():
        if "&" not in raw_line or r"\pm" not in raw_line:
            continue
        cells = [cell.strip() for cell in raw_line.split("&")]
        if len(cells) != 4:
            continue
        model = strip_tex_markup(cells[0])
        if model not in MODEL_LABELS:
            continue
        f1_mean, f1_sd = parse_metric_cell(cells[1])
        mse_mean, mse_sd = parse_metric_cell(cells[2])
        acc_mean, acc_sd = parse_metric_cell(cells[3])
        rows.append(
            PaperValue(
                model=model,
                f1_mean=f1_mean,
                f1_sd=f1_sd,
                mse_mean=mse_mean,
                mse_sd=mse_sd,
                acc_mean=acc_mean,
                acc_sd=acc_sd,
            )
        )

    if [row.model for row in rows] != DISPLAY_ORDER:
        raise ValueError(f"Unexpected Table II model order: {[row.model for row in rows]}")
    return rows


def configured_model_for(log_path: Path) -> str:
    config_path = log_path.parent / "config.json"
    if not config_path.exists():
        return "unknown"
    return str(json.loads(config_path.read_text(encoding="utf-8"))["model"])


def display_model(configured_model: str) -> str:
    reverse = {value: key for key, value in MODEL_LABELS.items()}
    if configured_model not in reverse:
        raise ValueError(f"Unmapped configured model: {configured_model}")
    return reverse[configured_model]


def last_float(pattern: str, text: str) -> float | None:
    matches = re.findall(pattern, text)
    return float(matches[-1]) if matches else None


def last_int(pattern: str, text: str) -> int | None:
    matches = re.findall(pattern, text)
    return int(matches[-1]) if matches else None


def parse_archived_log(log_path: Path) -> ArchivedRun:
    text = log_path.read_text(encoding="utf-8", errors="replace")
    configured_model = configured_model_for(log_path)
    model = display_model(configured_model)

    final_matches = re.findall(
        r"Final Test Results\s*\|\s*F1 Score:\s*([0-9.]+)\s*\|\s*"
        r"MSE:\s*([0-9.]+)\s*\|\s*MAE:\s*([0-9.]+)\s*\|\s*ACC:\s*([0-9.]+)",
        text,
    )
    final = tuple(map(float, final_matches[-1])) if final_matches else None

    parameter_count = last_int(r"总参数量:\s*(\d+)", text)
    attention_heads = last_int(r"heads=(\d+)", text)
    variant = f"heads={attention_heads}, params={parameter_count}" if attention_heads else f"params={parameter_count}"
    if parameter_count is None:
        variant = "failed-before-model-construction"

    val_matches = re.findall(
        r"Epoch\s+(\d+)\s*\|\s*\|.*?F1 Score:\s*([0-9.]+).*?"
        r"MSE:\s*([0-9.]+).*?ACC:\s*([0-9.]+)",
        text,
    )
    best_val_acc: float | None = None
    best_val_epoch: int | None = None
    if val_matches:
        parsed = [(int(epoch), float(acc)) for epoch, _f1, _mse, acc in val_matches]
        best_val_epoch, best_val_acc = max(parsed, key=lambda item: (item[1], -item[0]))

    epoch_matches = re.findall(r"^Epoch\s+(\d+),\s+Loss", text, flags=re.MULTILINE)
    final_epoch = int(epoch_matches[-1]) if epoch_matches else None

    return ArchivedRun(
        model=model,
        configured_model=configured_model,
        variant=variant,
        log_path=str(log_path.resolve()),
        log_sha256=sha256_file(log_path),
        has_final_test=final is not None,
        f1=final[0] if final else None,
        mse=final[1] if final else None,
        mae=final[2] if final else None,
        acc=final[3] if final else None,
        parameter_count=parameter_count,
        attention_heads=attention_heads,
        seed_documented=bool(re.search(r"(?:random[_ ]?seed|seed)\s*[:=]\s*\d+", text, flags=re.I)),
        final_epoch=final_epoch,
        best_val_acc=best_val_acc,
        best_val_epoch=best_val_epoch,
        # Legacy train.py saves a best checkpoint but evaluates the in-memory final model.
        final_model_is_best_checkpoint=False,
    )


def audit_rows(
    paper_values: list[PaperValue], archived_runs: list[ArchivedRun]
) -> list[dict[str, object]]:
    paper_by_model = {row.model: row for row in paper_values}
    rows: list[dict[str, object]] = []
    for run in archived_runs:
        if not run.has_final_test:
            continue
        paper = paper_by_model[run.model]
        for metric in ("f1", "mse", "acc"):
            archived_value = float(getattr(run, metric))
            paper_mean = float(getattr(paper, f"{metric}_mean"))
            paper_sd = float(getattr(paper, f"{metric}_sd"))
            delta = archived_value - paper_mean
            rows.append(
                {
                    "model": run.model,
                    "configured_model": run.configured_model,
                    "variant": run.variant,
                    "metric": metric.upper(),
                    "paper_mean": paper_mean,
                    "paper_sd": paper_sd,
                    "paper_low_1sd": paper_mean - paper_sd,
                    "paper_high_1sd": paper_mean + paper_sd,
                    "archived_final": archived_value,
                    "delta_archived_minus_paper": delta,
                    "absolute_z_by_reported_sd": abs(delta) / paper_sd if paper_sd else None,
                    "within_reported_1sd": abs(delta) <= paper_sd + 5e-8,
                    "log_path": run.log_path,
                    "log_sha256": run.log_sha256,
                }
            )
    return rows


def independent_evidence_rows(
    paper_values: list[PaperValue], archived_runs: list[ArchivedRun]
) -> list[dict[str, object]]:
    grouped: dict[str, list[ArchivedRun]] = defaultdict(list)
    for run in archived_runs:
        grouped[run.model].append(run)

    rows: list[dict[str, object]] = []
    for paper in paper_values:
        runs = grouped[paper.model]
        valid = [run for run in runs if run.has_final_test]
        variants = sorted({run.variant for run in valid})
        documented_seeds = sorted({run.log_path for run in valid if run.seed_documented})
        rows.append(
            {
                "model": paper.model,
                "paper_statistics": "mean +/- SD",
                "text_logs": len(runs),
                "valid_final_tests": len(valid),
                "distinct_architectures": len(variants),
                "variants": "; ".join(variants),
                "runs_with_documented_seed": len(documented_seeds),
                "independent_runs_needed_for_sd": ">=2 (protocol should predefine >=5)",
                "mean_sd_reproducible": False,
                "reason": (
                    "Different architectures cannot be pooled"
                    if len(variants) > 1
                    else "No documented independent seed/run matrix or aggregation artifact"
                ),
            }
        )
    return rows


def best_models_from_archived(archived_runs: Iterable[ArchivedRun]) -> dict[str, object]:
    valid = [run for run in archived_runs if run.has_final_test]
    max_f1 = max(float(run.f1) for run in valid)
    min_mse = min(float(run.mse) for run in valid)
    max_acc = max(float(run.acc) for run in valid)

    def winners(field: str, target: float) -> list[str]:
        return sorted(
            {
                f"{run.model} ({run.variant})"
                for run in valid
                if abs(float(getattr(run, field)) - target) < 5e-8
            }
        )

    return {
        "F1": {"value": max_f1, "winners": winners("f1", max_f1)},
        "MSE": {"value": min_mse, "winners": winners("mse", min_mse)},
        "ACC": {"value": max_acc, "winners": winners("acc", max_acc)},
    }


def mse_accuracy_identity_audit(
    paper_values: list[PaperValue], archived_runs: list[ArchivedRun]
) -> dict[str, object]:
    """Check the binary hard-label identity MSE == 1 - accuracy.

    The archived code computes MSE from argmax class labels. For binary labels,
    every squared error is exactly the 0/1 misclassification indicator. Thus
    run-wise MSE equals 1-accuracy, their means sum to one, and their sample
    standard deviations must be identical when calculated over the same runs.
    """

    valid_runs = [run for run in archived_runs if run.has_final_test]
    archive_failures = [
        run.log_path
        for run in valid_runs
        if not np.isclose(float(run.mse), 1.0 - float(run.acc), atol=5e-4, rtol=0.0)
    ]
    paper_mean_failures = [
        row.model
        for row in paper_values
        if not np.isclose(row.mse_mean + row.acc_mean, 1.0, atol=5e-4, rtol=0.0)
    ]
    paper_sd_mismatches = [
        {
            "model": row.model,
            "mse_sd": row.mse_sd,
            "acc_sd": row.acc_sd,
            "absolute_difference": abs(row.mse_sd - row.acc_sd),
        }
        for row in paper_values
        if not np.isclose(row.mse_sd, row.acc_sd, atol=5e-4, rtol=0.0)
    ]
    return {
        "mathematical_basis": "binary argmax predictions imply MSE = 1 - accuracy for every run",
        "archived_runs_checked": len(valid_runs),
        "archived_identity_holds": not archive_failures,
        "archived_identity_failures": archive_failures,
        "paper_mean_identity_holds": not paper_mean_failures,
        "paper_mean_identity_failures": paper_mean_failures,
        "paper_sd_identity_holds": not paper_sd_mismatches,
        "paper_sd_mismatches": paper_sd_mismatches,
    }


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise ValueError(f"Refusing to write empty CSV: {path}")
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def make_performance_chart(
    paper_values: list[PaperValue],
    archived_runs: list[ArchivedRun],
    output_path: Path,
) -> None:
    paper_by_model = {row.model: row for row in paper_values}
    model_y = {model: index for index, model in enumerate(DISPLAY_ORDER)}
    metrics = [("f1", "F1 (higher is better)"), ("mse", "MSE (lower is better)"), ("acc", "Accuracy (higher is better)")]
    limits = {"f1": (0.56, 0.73), "mse": (0.25, 0.38), "acc": (0.62, 0.75)}

    fig, axes = plt.subplots(1, 3, figsize=(16, 6.8), sharey=True)
    paper_color = "#315C8C"
    archive_color = "#C96A2B"
    for axis, (metric, title) in zip(axes, metrics):
        for model in DISPLAY_ORDER:
            paper = paper_by_model[model]
            mean = float(getattr(paper, f"{metric}_mean"))
            sd = float(getattr(paper, f"{metric}_sd"))
            y = model_y[model]
            axis.errorbar(
                mean,
                y,
                xerr=sd,
                fmt="o",
                color=paper_color,
                ecolor=paper_color,
                capsize=4,
                markersize=6,
                linewidth=1.6,
                label="Paper mean +/- reported SD" if y == 0 else None,
                zorder=2,
            )

        offsets: Counter[str] = Counter()
        for run in archived_runs:
            if not run.has_final_test:
                continue
            offsets[run.model] += 1
            count_for_model = sum(1 for item in archived_runs if item.model == run.model and item.has_final_test)
            offset_index = offsets[run.model] - 1
            offset = (offset_index - (count_for_model - 1) / 2) * 0.14
            marker = "D" if run.model == "multi_GAT" else "s"
            axis.scatter(
                float(getattr(run, metric)),
                model_y[run.model] + offset,
                marker=marker,
                facecolors="white",
                edgecolors=archive_color,
                linewidths=1.7,
                s=52,
                label="Archived final test (one log)" if run.log_path == next(item.log_path for item in archived_runs if item.has_final_test) else None,
                zorder=3,
            )

        axis.set_title(title, fontsize=11, fontweight="semibold", color="#202124")
        axis.set_xlim(*limits[metric])
        axis.grid(axis="x", color="#D9DEE5", linewidth=0.8)
        axis.set_axisbelow(True)
        axis.spines[["top", "right"]].set_visible(False)
        axis.tick_params(labelsize=9, colors="#3F4650")

    axes[0].set_yticks(range(len(DISPLAY_ORDER)), DISPLAY_ORDER)
    axes[0].invert_yaxis()
    fig.suptitle("Table II: manuscript statistics versus archived final-test records", fontsize=15, fontweight="bold", y=0.98)
    fig.text(
        0.5,
        0.925,
        "Focused axes; each square/diamond is one archived log, not an independent-run aggregate. MultiGAT diamonds are different heads variants.",
        ha="center",
        fontsize=9.5,
        color="#4B535D",
    )
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 0.025))
    fig.subplots_adjust(left=0.11, right=0.985, top=0.86, bottom=0.14, wspace=0.13)
    fig.savefig(output_path, dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def markdown_report(
    tex_path: Path,
    archive_dir: Path,
    paper_values: list[PaperValue],
    archived_runs: list[ArchivedRun],
    evidence_rows: list[dict[str, object]],
    winners: dict[str, object],
    metric_identity: dict[str, object],
) -> str:
    valid_runs = [run for run in archived_runs if run.has_final_test]
    failed_runs = [run for run in archived_runs if not run.has_final_test]
    within = sum(bool(row["within_reported_1sd"]) for row in evidence_rows)
    lines = [
        "# Table II reproducibility audit",
        "",
        "## Conclusion",
        "",
        "**The archived files cannot reproduce any reported mean +/- standard deviation in Table II.** "
        "They contain no documented independent seed/run matrix and no aggregation artifact. "
        "The eight valid final-test records are individual terminal-model evaluations; the legacy training path did not restore the best validation checkpoint before testing.",
        "",
        "The archived snapshot also does not support the paper's claim that QWGCN wins all three metrics: "
        f"its best observed F1 is {winners['F1']['value']:.4f}, while the best archived MSE is {winners['MSE']['value']:.4f} "
        f"and the best archived accuracy is {winners['ACC']['value']:.4f}.",
        "",
        "## Coverage",
        "",
        f"- Manuscript source: `{tex_path.resolve()}` (SHA-256 `{sha256_file(tex_path)}`)",
        f"- Archive root: `{archive_dir.resolve()}`",
        f"- Table rows parsed: {len(paper_values)}",
        f"- Text logs inspected: {len(archived_runs)}",
        f"- Logs with final-test records: {len(valid_runs)}",
        f"- Logs failing before final test: {len(failed_runs)}",
        f"- Metric observations within the paper's reported mean +/- 1 SD interval: {within}/{len(evidence_rows)} (compatibility is not reproducibility)",
        "",
        "## Archived performance leaders",
        "",
        f"- F1: {winners['F1']['value']:.4f} — {', '.join(winners['F1']['winners'])}",
        f"- MSE: {winners['MSE']['value']:.4f} — {', '.join(winners['MSE']['winners'])}",
        f"- Accuracy: {winners['ACC']['value']:.4f} — {', '.join(winners['ACC']['winners'])}",
        "",
        "## Blocking provenance gaps",
        "",
        "1. A mean and sample SD require multiple independent runs under one fixed protocol; seeds are absent from all valid logs.",
        "2. The two MultiGAT logs use different architectures (`heads=8`/12,326 parameters and `heads=2`/1,550 parameters), so they cannot be pooled.",
        "3. Checkpoints named `best...pth` are successive improvements within one run, not independent replicates.",
        "4. The legacy training code evaluates the terminal in-memory model after saving `last`; it never reloads the best-validation checkpoint before the final test.",
        "5. Baseline constructors use their internal dropout default (0.1) because `model_get.py` does not forward the configured 0.2, while QWGCN does receive 0.2.",
        "6. The manuscript does not specify the Table II run count, seed list, split policy, checkpoint-selection rule, or aggregation convention.",
        "7. The archived code computes MSE from binary argmax class labels, so every run must satisfy `MSE = 1 - Acc`, and MSE/Acc sample SDs must be identical over the same runs. All eight archived records satisfy the run-wise identity. Six paper rows have complementary MSE/Acc means, but TransformerConv reports `0.305 + 0.690 = 0.995`, outside the three-decimal rounding bound. The paper also reports unequal MSE/Acc SDs for GCN, GraphSAGE, and TransformerConv. These values cannot come from one common hard-label run set under the archived metric definition.",
        "",
        "## Required rerun contract",
        "",
        "Use paired seeds across all seven fixed architectures, persist the exact train/validation/test indices, select the checkpoint using validation data only, evaluate the test set once per seed, and compute arithmetic mean plus sample SD (`ddof=1`) only across completed independent seeds. Keep the full-PROTEINS experiment isolated from the deferred small-graph oversampling issue.",
        "",
        "## Generated evidence files",
        "",
        "- `paper_table2_values.csv`: values parsed from the current manuscript.",
        "- `archived_final_tests.csv`: one row per archived text log, including failed logs.",
        "- `paper_vs_archive_metrics.csv`: metric-level deltas and reported-interval checks.",
        "- `reproducibility_matrix.csv`: whether each manuscript row has sufficient independent evidence.",
        "- `table2_performance_comparison.png`: paper intervals versus individual archived terminal tests.",
        "- `table2_audit.json`: machine-readable audit summary with hashes and verdicts.",
    ]
    return "\n".join(lines) + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tex", type=Path, required=True, help="Current manuscript TeX source")
    parser.add_argument("--archive", type=Path, required=True, help="train_data/QWGCN directory")
    parser.add_argument("--output", type=Path, required=True, help="Local output directory")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir: Path = args.output
    output_dir.mkdir(parents=True, exist_ok=True)

    paper_values = parse_paper_table(args.tex)
    log_paths = sorted(args.archive.rglob("*.txt"))
    archived_runs = [parse_archived_log(path) for path in log_paths]
    evidence_rows = audit_rows(paper_values, archived_runs)
    reproducibility_rows = independent_evidence_rows(paper_values, archived_runs)
    winners = best_models_from_archived(archived_runs)
    metric_identity = mse_accuracy_identity_audit(paper_values, archived_runs)

    paper_rows = [asdict(row) for row in paper_values]
    archived_rows = [asdict(row) for row in archived_runs]
    write_csv(output_dir / "paper_table2_values.csv", paper_rows)
    write_csv(output_dir / "archived_final_tests.csv", archived_rows)
    write_csv(output_dir / "paper_vs_archive_metrics.csv", evidence_rows)
    write_csv(output_dir / "reproducibility_matrix.csv", reproducibility_rows)
    make_performance_chart(paper_values, archived_runs, output_dir / "table2_performance_comparison.png")

    audit = {
        "schema_version": 1,
        "manuscript": {"path": str(args.tex.resolve()), "sha256": sha256_file(args.tex)},
        "archive": {"path": str(args.archive.resolve()), "text_log_count": len(log_paths)},
        "counts": {
            "paper_rows": len(paper_values),
            "valid_final_tests": sum(run.has_final_test for run in archived_runs),
            "failed_logs": sum(not run.has_final_test for run in archived_runs),
            "documented_seed_logs": sum(run.seed_documented for run in archived_runs),
        },
        "verdicts": {
            "table2_mean_sd_reproducible": False,
            "qwgcn_best_f1_in_archived_snapshot": any(
                winner.startswith("QWGCN ") for winner in winners["F1"]["winners"]
            ),
            "qwgcn_best_mse_in_archived_snapshot": any(
                winner.startswith("QWGCN ") for winner in winners["MSE"]["winners"]
            ),
            "qwgcn_best_accuracy_in_archived_snapshot": any(
                winner.startswith("QWGCN ") for winner in winners["ACC"]["winners"]
            ),
            "safe_to_copy_archived_values_into_manuscript": False,
            "archived_mse_equals_one_minus_accuracy": metric_identity["archived_identity_holds"],
            "paper_mse_accuracy_means_are_complementary": metric_identity["paper_mean_identity_holds"],
            "paper_mse_accuracy_sds_are_consistent": metric_identity["paper_sd_identity_holds"],
        },
        "archived_leaders": winners,
        "paper_values": paper_rows,
        "archived_runs": archived_rows,
        "reproducibility_matrix": reproducibility_rows,
        "mse_accuracy_identity_audit": metric_identity,
    }
    (output_dir / "table2_audit.json").write_text(
        json.dumps(audit, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    (output_dir / "TABLE2_AUDIT.md").write_text(
        markdown_report(
            args.tex,
            args.archive,
            paper_values,
            archived_runs,
            evidence_rows,
            winners,
            metric_identity,
        ),
        encoding="utf-8",
    )

    print(json.dumps(audit["counts"], indent=2))
    print(json.dumps(audit["verdicts"], indent=2))


if __name__ == "__main__":
    main()
