"""Adversarial verification for the corrected Fig. 6 runtime benchmark."""

import argparse
import csv
import importlib.util
import json
import math
from pathlib import Path

import numpy as np


EXPECTED_NODES = {15, 30, 50, 100}
EXPECTED_ORDERS = set(range(1, 11))
EXPECTED_GRAPHS = {"Random", "Star", "Complete"}


def check(condition, label, details, results):
    results.append({"check": label, "passed": bool(condition), "details": str(details)})


def load_module(path):
    spec = importlib.util.spec_from_file_location("fig6_benchmark", str(path))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def read_csv(path):
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def pdf_page_count(path):
    try:
        from pypdf import PdfReader
    except ImportError:
        from PyPDF2 import PdfReader
    return len(PdfReader(str(path)).pages)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--script", type=Path, required=True)
    parser.add_argument("--tex", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    results = []
    module = load_module(args.script)

    rng = np.random.RandomState(20260907)
    matrix = rng.normal(size=(5, 5))
    matrix = (matrix + matrix.T) / 2.0
    generator = -1j * 0.43 * matrix
    for order in (1, 2, 6, 10):
        observed = module.dense_taylor_from_generator(generator, order)
        expected = sum(
            np.linalg.matrix_power(generator, k) / math.factorial(k)
            for k in range(order + 1)
        )
        error = float(np.max(np.abs(observed - expected)))
        check(error < 1e-11, "kernel_order_{}".format(order), "max abs error {:.3e}".format(error), results)

    source = args.script.read_text(encoding="utf-8")
    check(
        "-1j * t_value * alpha * a_hat" in source,
        "negative_time_generator",
        "benchmark generator implements exp(-i A_hat t)",
        results,
    )
    check(
        "fidelity" not in source.lower(),
        "scope_excludes_fidelity",
        "deferred fidelity metric is not recomputed by Fig. 6 script",
        results,
    )

    raw_rows = read_csv(args.output_dir / "fig6_runtime_raw.csv")
    summary_rows = read_csv(args.output_dir / "fig6_runtime_summary.csv")
    metrics = json.loads((args.output_dir / "fig6_runtime_metrics.json").read_text(encoding="utf-8"))

    coverage = {
        (int(row["n_nodes"]), row["graph_type"], int(row["order"]))
        for row in summary_rows
    }
    expected_coverage = {
        (n_nodes, graph_type, order)
        for n_nodes in EXPECTED_NODES
        for graph_type in EXPECTED_GRAPHS
        for order in EXPECTED_ORDERS
    }
    check(coverage == expected_coverage, "complete_task_coverage", "{} of 120 tasks".format(len(coverage)), results)

    trials = int(metrics["trials"])
    counts = {}
    for row in raw_rows:
        key = (int(row["n_nodes"]), row["graph_type"], int(row["order"]))
        counts[key] = counts.get(key, 0) + 1
    check(
        len(raw_rows) == 120 * trials and set(counts.values()) == {trials},
        "independent_trial_count",
        "{} rows; {} trials per task".format(len(raw_rows), trials),
        results,
    )

    valid_intervals = True
    positive_runtimes = True
    for row in summary_rows:
        q1 = float(row["q1_ms"])
        median = float(row["median_ms"])
        q3 = float(row["q3_ms"])
        valid_intervals &= q1 <= median <= q3
        positive_runtimes &= q1 > 0.0 and math.isfinite(q3)
        valid_intervals &= int(row["matrix_multiplications_per_call"]) == int(row["order"])
    check(valid_intervals, "summary_interval_and_work_count", "q1 <= median <= q3 and matmul count = K", results)
    check(positive_runtimes, "positive_finite_runtimes", "all summarized measurements are positive and finite", results)

    fit_summary = metrics["fit_summary"]
    check(
        bool(fit_summary["all_slopes_positive"]),
        "positive_order_scaling",
        "all 12 runtime-versus-order slopes are positive",
        results,
    )
    check(
        float(fit_summary["median_r_squared"]) >= 0.90,
        "median_linearity",
        "median R^2 = {:.4f}".format(float(fit_summary["median_r_squared"])),
        results,
    )

    selected = {
        (int(row["n_nodes"]), row["graph_type"], int(row["order"])): float(row["median_ms"])
        for row in summary_rows
    }
    cross_size_ratios = [
        selected[(100, graph_type, 10)] / selected[(15, graph_type, 10)]
        for graph_type in EXPECTED_GRAPHS
    ]
    check(
        min(cross_size_ratios) > 2.0,
        "node_size_effect",
        "N=100/N=15 runtime ratios at K=10: {}".format(
            ", ".join("{:.2f}".format(value) for value in sorted(cross_size_ratios))
        ),
        results,
    )

    max_topology_ratio = 1.0
    for n_nodes in EXPECTED_NODES:
        for order in EXPECTED_ORDERS:
            values = [selected[(n_nodes, graph_type, order)] for graph_type in EXPECTED_GRAPHS]
            max_topology_ratio = max(max_topology_ratio, max(values) / min(values))
    check(
        max_topology_ratio < 3.0,
        "topology_runtime_sanity",
        "maximum dense-topology median ratio = {:.3f}".format(max_topology_ratio),
        results,
    )

    tex = args.tex.read_text(encoding="utf-8")
    required_caption_terms = (
        "dense reference Taylor kernel",
        "interquartile ranges",
        "single BLAS thread",
        "graph construction and normalization are excluded",
    )
    check(
        all(term in tex for term in required_caption_terms),
        "caption_discloses_protocol",
        "caption states estimator, uncertainty, thread control, and exclusions",
        results,
    )
    check(
        "small timing differences must not be interpreted as topology-dependent complexity" in tex,
        "topology_claim_bounded",
        "manuscript no longer treats dense timing noise as topology complexity",
        results,
    )
    check(
        "supports only the dense global-operator cost statement" in tex,
        "scope_claim_bounded",
        "manuscript distinguishes reference kernel from sparse/end-to-end runtime",
        results,
    )

    figure_pdf = args.output_dir / "node_time_1.pdf"
    figure_png = args.output_dir / "node_time_1.png"
    check(figure_pdf.is_file() and pdf_page_count(figure_pdf) == 1, "pdf_artifact", "one-page PDF figure", results)
    check(figure_png.is_file() and figure_png.stat().st_size > 50_000, "png_artifact", "review PNG exists", results)

    passed = all(result["passed"] for result in results)
    report = {
        "status": "PASS" if passed else "FAIL",
        "checks_passed": sum(result["passed"] for result in results),
        "checks_total": len(results),
        "checks": results,
    }
    report_json = args.output_dir / "fig6_verification.json"
    report_md = args.output_dir / "fig6_verification.md"
    report_json.write_text(json.dumps(report, indent=2), encoding="utf-8")

    lines = ["# Fig. 6 adversarial verification", "", "Status: **{}**".format(report["status"]), ""]
    for result in results:
        mark = "PASS" if result["passed"] else "FAIL"
        lines.append("- {} — **{}**: {}".format(result["check"], mark, result["details"]))
    report_md.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print("{}: {}/{} checks passed".format(report["status"], report["checks_passed"], report["checks_total"]))
    for result in results:
        print("[{}] {}: {}".format("PASS" if result["passed"] else "FAIL", result["check"], result["details"]))
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
