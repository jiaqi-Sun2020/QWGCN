"""Adversarial verification for Project 6 (Fig. 1 edge-count consistency)."""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--script", type=Path, required=True)
    parser.add_argument("--tex", type=Path, required=True)
    parser.add_argument("--figure", type=Path, required=True)
    parser.add_argument("--before-figure", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    return parser.parse_args()


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def extract_adjacency(script_path: Path) -> np.ndarray:
    tree = ast.parse(script_path.read_text(encoding="utf-8-sig"), filename=str(script_path))
    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue
        if not any(isinstance(target, ast.Name) and target.id == "A" for target in node.targets):
            continue
        if not isinstance(node.value, ast.Call) or not node.value.args:
            continue
        try:
            matrix = np.asarray(ast.literal_eval(node.value.args[0]), dtype=float)
        except (ValueError, TypeError, SyntaxError) as exc:
            raise ValueError("Could not decode the literal adjacency matrix A.") from exc
        return matrix
    raise ValueError("No literal assignment to adjacency matrix A was found.")


def ctqw_probabilities(adjacency: np.ndarray, time: float) -> np.ndarray:
    eigenvalues, eigenvectors = np.linalg.eigh(adjacency)
    propagator = (eigenvectors * np.exp(-1j * eigenvalues * time)) @ eigenvectors.conj().T
    initial_state = np.ones(adjacency.shape[0], dtype=complex) / np.sqrt(adjacency.shape[0])
    return np.abs(propagator @ initial_state) ** 2


def verify(args: argparse.Namespace) -> dict:
    script = args.script.resolve()
    tex_path = args.tex.resolve()
    figure = args.figure.resolve()
    before_figure = args.before_figure.resolve()

    adjacency = extract_adjacency(script)
    degrees = adjacency.sum(axis=1)
    edge_count = int(round(float(adjacency.sum() / 2.0)))
    probabilities = ctqw_probabilities(adjacency, time=0.5)
    tex = tex_path.read_text(encoding="utf-8-sig")
    figure_size = Image.open(figure).size

    raw_checks = {
        "adjacency_is_10_by_10": adjacency.shape == (10, 10),
        "adjacency_is_symmetric": bool(np.allclose(adjacency, adjacency.T, atol=0.0)),
        "adjacency_has_zero_diagonal": bool(np.allclose(np.diag(adjacency), 0.0, atol=0.0)),
        "adjacency_is_binary": bool(np.all(np.isin(adjacency, [0.0, 1.0]))),
        "degree_sum_is_40": bool(np.isclose(degrees.sum(), 40.0, atol=0.0)),
        "undirected_edge_count_is_20": edge_count == 20,
        "nodes_0_and_2_have_degree_6": degrees[0] == 6 and degrees[2] == 6,
        "nodes_0_and_2_are_max_degree": degrees[0] == degrees.max() and degrees[2] == degrees.max(),
        "ctqw_probability_is_normalized": bool(np.isclose(probabilities.sum(), 1.0, atol=1e-12)),
        "node_0_probability_supports_0_23": abs(float(probabilities[0]) - 0.23) < 0.005,
        "node_2_probability_supports_0_25": abs(float(probabilities[2]) - 0.25) < 0.005,
        "caption_and_body_each_report_20_edges": tex.count("10 vertices and 20 edges") == 2,
        "no_19_edge_claim_remains": "10 vertices and 19 edges" not in tex,
        "paper_figure_is_unchanged": sha256(figure) == sha256(before_figure),
        "paper_figure_is_nonempty": figure_size[0] > 0 and figure_size[1] > 0,
    }
    checks = [
        {"name": name, "passed": bool(passed)} for name, passed in raw_checks.items()
    ]
    passed = sum(item["passed"] for item in checks)
    report = {
        "project": "project6_fig1_edge_count",
        "status": "PASS" if passed == len(checks) else "FAIL",
        "checks_passed": passed,
        "checks_total": len(checks),
        "checks": checks,
        "metrics": {
            "node_count": int(adjacency.shape[0]),
            "adjacency_sum": float(adjacency.sum()),
            "degree_sum": float(degrees.sum()),
            "edge_count": edge_count,
            "degrees": [int(value) for value in degrees],
            "max_degree": int(degrees.max()),
            "max_degree_nodes": [
                int(index) for index, value in enumerate(degrees) if value == degrees.max()
            ],
            "time": 0.5,
            "probability_node_0": float(probabilities[0]),
            "probability_node_2": float(probabilities[2]),
            "probability_sum": float(probabilities.sum()),
            "figure_width_px": figure_size[0],
            "figure_height_px": figure_size[1],
        },
        "hashes": {
            "script_sha256": sha256(script),
            "tex_sha256": sha256(tex_path),
            "figure_sha256": sha256(figure),
            "before_figure_sha256": sha256(before_figure),
        },
    }
    args.report.resolve().parent.mkdir(parents=True, exist_ok=True)
    args.report.resolve().write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return report


def main() -> None:
    report = verify(parse_args())
    print(
        f"{report['status']}: {report['checks_passed']}/{report['checks_total']} checks passed"
    )
    metrics = report["metrics"]
    print(
        "edge_count={edge_count}; degree_sum={degree_sum:.0f}; "
        "P0(t=0.5)={probability_node_0:.6f}; P2(t=0.5)={probability_node_2:.6f}".format(
            **metrics
        )
    )
    if report["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
