"""Adversarial verification for Project 7 (Fig. 2 and text/code consistency)."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.linalg import expm


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tex", type=Path, required=True)
    parser.add_argument("--qwgcn", type=Path, required=True)
    parser.add_argument("--qwgnn", type=Path, required=True)
    parser.add_argument("--figure", type=Path, required=True)
    parser.add_argument("--before-figure", type=Path, required=True)
    parser.add_argument("--before-qwgcn", type=Path, required=True)
    parser.add_argument("--before-qwgnn", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    return parser.parse_args()


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def latex_braces_balanced(text: str) -> bool:
    uncommented_lines = []
    for line in text.splitlines():
        cut = len(line)
        for index, character in enumerate(line):
            if character != "%":
                continue
            backslashes = 0
            cursor = index - 1
            while cursor >= 0 and line[cursor] == "\\":
                backslashes += 1
                cursor -= 1
            if backslashes % 2 == 0:
                cut = index
                break
        uncommented_lines.append(line[:cut])
    text = "\n".join(uncommented_lines)

    depth = 0
    for index, character in enumerate(text):
        if character not in "{}":
            continue
        backslashes = 0
        cursor = index - 1
        while cursor >= 0 and text[cursor] == "\\":
            backslashes += 1
            cursor -= 1
        if backslashes % 2:
            continue
        depth += 1 if character == "{" else -1
        if depth < 0:
            return False
    return depth == 0


def normalized_with_self_loops(adjacency: np.ndarray) -> np.ndarray:
    with_loops = adjacency + np.eye(adjacency.shape[0])
    degree = with_loops.sum(axis=1)
    scale = np.diag(degree ** -0.5)
    return scale @ with_loops @ scale


def verify(args: argparse.Namespace) -> dict:
    tex_path = args.tex.resolve()
    qwgcn_path = args.qwgcn.resolve()
    qwgnn_path = args.qwgnn.resolve()
    figure_path = args.figure.resolve()

    tex = tex_path.read_text(encoding="utf-8-sig")
    qwgcn = qwgcn_path.read_text(encoding="utf-8-sig")
    qwgnn = qwgnn_path.read_text(encoding="utf-8-sig")
    figure_size = Image.open(figure_path).size

    adjacency = np.array(
        [[0.0, 1.0, 0.0], [1.0, 0.0, 1.0], [0.0, 1.0, 0.0]],
        dtype=float,
    )
    operator = normalized_with_self_loops(adjacency)
    time = 1.0
    exact = expm(-1j * operator * time)
    second = np.eye(3) - 1j * time * operator - 0.5 * time**2 * operator @ operator
    unitarity_defect = float(np.linalg.norm(second.conj().T @ second - np.eye(3), 2))
    approximation_error = float(np.linalg.norm(exact - second, 2))

    qwgcn_pool = qwgcn.find("x = global_mean_pool(x, batch)")
    qwgcn_softmax = qwgcn.find("return F.log_softmax(x, dim=1)")
    qwgnn_pool = qwgnn.find("x = global_mean_pool(x, batch)")
    qwgnn_softmax = qwgnn.find("x = F.log_softmax(x, dim=1)")

    forbidden_tex = (
        "preserves the complex-valued modeling and unitarity constraints",
        "each edge is accessed only once",
        "we use a single CTQW",
        "k$-hop unitary evolution",
        "initialize the quantum walk time",
        "complex edge weight achieved by CTQW",
        "CTQW evolution edge weight",
        "global\\_mean\\_pool}\\!\\left(\n\t\t\\mathrm{log\\_softmax}",
        "global\\_mean\\_pool}\\Big(\\mathrm{log\\_softmax}",
    )

    raw_checks = {
        "figure2_is_unchanged": sha256(figure_path) == sha256(args.before_figure.resolve()),
        "figure2_is_nonempty": figure_size[0] > 0 and figure_size[1] > 0,
        "qwgcn_source_is_unchanged": sha256(qwgcn_path) == sha256(args.before_qwgcn.resolve()),
        "qwgnn_source_is_unchanged": sha256(qwgnn_path) == sha256(args.before_qwgnn.resolve()),
        "base_hamiltonian_is_A": (
            "H = A" in tex
            or (
                re.search(r"H\s*=\s*\\gamma\s*A", tex) is not None
                and re.search(r"\\gamma\s*=\s*1", tex) is not None
            )
        ),
        "exact_propagator_has_negative_sign": (
            re.search(
                r"G\(t\)\s*=\s*e\^\{-iHt\}\s*=\s*e\^\{-iAt\}", tex
            )
            is not None
        ),
        "no_positive_exact_propagator_remains": not re.search(r"e\^\{\+?i\s*(?:A|\\tilde\{A\})", tex),
        "second_order_polynomial_is_correct": (
            "I - i t \\tilde{A} - \\frac{t^2}{2} \\tilde{A}^2 = G_2(t)" in tex
        ),
        "ctqw_state_reference_is_defined": (
            tex.count("\\label{eq:CQW3}") == 1 and "\\ref{eq:CQW3}" in tex
        ),
        "runtime_qwgcn_uses_negative_first_order": (
            "evolved_x = x_complex - 1j * self.evolution_time" in qwgcn
        ),
        "runtime_qwgcn_uses_second_order": (
            "second_coeff = (-1j * self.evolution_time * self.diffusion_strength) ** 2 / 2.0"
            in qwgcn
        ),
        "runtime_defaults_are_K2_t1": (
            "evolution_order=2, evolution_time=1.0" in qwgcn
        ),
        "paper_defaults_are_K2_t1": (
            "same fixed step time $t=1$ and the same order $K=2$" in tex
        ),
        "paper_defines_runtime_self_loops": "$\\hat{A}=A+I$" in tex,
        "runtime_adds_self_loops": "add_self_loops(edge_index" in qwgcn,
        "paper_states_polynomial_is_nonunitary": "is not itself exactly unitary" in tex,
        "numerical_polynomial_is_nonunitary": unitarity_defect > 1e-6,
        "numerical_approximation_is_not_exact": approximation_error > 1e-6,
        "paper_evolves_current_feature_matrix": "Y^l = G_2(t)X^l" in tex,
        "runtime_evolves_current_feature_matrix_per_layer": (
            "evolution_weights, evolved_features = self.quantum_evolution(" in qwgcn
            and "for i, (quantum_layer, batch_norm)" in qwgcn
        ),
        "paper_normalizes_weights_per_graph": (
            "normalized independently within each graph" in tex
        ),
        "paper_calls_runtime_weights_nodewise": (
            "node-wise evolution weighting" in tex
        ),
        "runtime_normalizes_weights_per_graph": "graph_weight_sum.index_add_" in qwgcn,
        "paper_hop_weights_use_softmax": "\\operatorname{softmax}(\\theta)_h" in tex,
        "runtime_hop_weights_use_softmax": "F.softmax(self.hop_weights, dim=0)" in qwgcn,
        "paper_complexity_counts_K_and_k": "$O((K+k)E F_{\\text{in}})$" in tex,
        "paper_memory_keeps_E_and_features": (
            "$O(E+N(F_{\\text{in}}+F_{\\text{out}}))$" in tex
        ),
        "qwgcn_readout_order_is_pool_then_softmax": 0 <= qwgcn_pool < qwgcn_softmax,
        "qwgnn_readout_order_is_pool_then_softmax": 0 <= qwgnn_pool < qwgnn_softmax,
        "paper_readout_order_is_pool_then_softmax": (
            "\\mathrm{log\\_softmax}\\!\\left(\n\t\t\\mathrm{global\\_mean\\_pool}"
            in tex
            and "\\mathrm{log\\_softmax}\\Big(\\mathrm{global\\_mean\\_pool}"
            in tex
        ),
        "stale_contradictory_phrases_removed": not any(item in tex for item in forbidden_tex),
        "fidelity_definition_remains_deferred": (
            "\\bar{F}_{\\text{avg}} = \\frac{m + |\\text{Tr}(U^\\dagger M)|^2}{m(m+1)}"
            in tex
        ),
        "latex_braces_are_balanced": latex_braces_balanced(tex),
        "algorithm_blocks_are_balanced": (
            tex.count("\\begin{algorithm}") == tex.count("\\end{algorithm}")
            and tex.count("\\begin{algorithmic}") == tex.count("\\end{algorithmic}")
        ),
    }
    checks = [{"name": name, "passed": bool(value)} for name, value in raw_checks.items()]
    passed = sum(check["passed"] for check in checks)
    report = {
        "project": "project7_fig2_text_consistency",
        "status": "PASS" if passed == len(checks) else "FAIL",
        "checks_passed": passed,
        "checks_total": len(checks),
        "checks": checks,
        "metrics": {
            "figure_width_px": figure_size[0],
            "figure_height_px": figure_size[1],
            "unitarity_defect_spectral_norm": unitarity_defect,
            "second_order_error_spectral_norm_at_t1": approximation_error,
            "operator_spectral_norm": float(np.linalg.norm(operator, 2)),
        },
        "hashes": {
            "tex_sha256": sha256(tex_path),
            "qwgcn_sha256": sha256(qwgcn_path),
            "qwgnn_sha256": sha256(qwgnn_path),
            "figure_sha256": sha256(figure_path),
            "before_figure_sha256": sha256(args.before_figure.resolve()),
        },
        "scope": {
            "figure2": "manual visual audit: e^{-iAt}; no Taylor order or fixed-time claim; unchanged",
            "deferred": ["average fidelity redefinition", "small-graph data split"],
        },
    }
    args.report.resolve().parent.mkdir(parents=True, exist_ok=True)
    args.report.resolve().write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return report


def main() -> None:
    report = verify(parse_args())
    print(f"{report['status']}: {report['checks_passed']}/{report['checks_total']} checks passed")
    metrics = report["metrics"]
    print(
        "unitarity_defect={unitarity_defect_spectral_norm:.6f}; "
        "approximation_error(t=1)={second_order_error_spectral_norm_at_t1:.6f}".format(
            **metrics
        )
    )
    if report["status"] != "PASS":
        failed = [item["name"] for item in report["checks"] if not item["passed"]]
        print("failed=" + ", ".join(failed))
        raise SystemExit(1)


if __name__ == "__main__":
    main()
