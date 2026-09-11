"""Adversarial verification for the corrected manuscript Fig. 4.

Run from the corrected repository root inside the full QWGCN/paper workspace:
    python -B diagnostics/verify_fig4_operator_error.py \
        --script "D:/AI/Quantum/quantum/paper/paper_build/PAPER_PRA/img_code_ai/近似保真度计算/平均门保真度2.py" \
        --tex "D:/AI/Quantum/quantum/paper/paper_build/PAPER_PRA/20260907/refQWGCN0105.tex" \
        --figure "../_local_review/project3_fig4_operator_error/output/evolution_approx_1.png" \
        --report "../_local_review/project3_fig4_operator_error/output/verification_report.json"

This check deliberately does not call any fidelity function. It verifies the
operator convention, Taylor order, elementwise complex error, convergence,
caption values, and exported image contract independently of the plotting path.
"""

import argparse
import ast
import hashlib
import json
from pathlib import Path

import networkx as nx
import numpy as np
from PIL import Image
from scipy.linalg import expm


GRAPH_TYPES = ("random", "star", "complete")
EXPECTED_MAX_ERRORS = {
    "random": 0.027824479710259496,
    "star": 0.05425898734456596,
    "complete": 0.02044647146899899,
}


def load_functions(path):
    wanted = {
        "create_graph_matrices",
        "quantum_evolution_exact",
        "quantum_evolution_approx",
    }
    tree = ast.parse(path.read_text(encoding="utf-8-sig"))
    nodes = [
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name in wanted
    ]
    found = {node.name for node in nodes}
    if found != wanted:
        raise AssertionError(f"missing functions: {sorted(wanted - found)}")
    namespace = {"np": np, "nx": nx, "expm": expm}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), "exec"), namespace)
    return namespace


def independent_taylor(matrix, t, alpha, order):
    generator = -1j * alpha * matrix * t
    result = np.eye(matrix.shape[0], dtype=complex)
    term = np.eye(matrix.shape[0], dtype=complex)
    for k in range(1, order + 1):
        term = term @ generator / k
        result += term
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--script", type=Path, required=True)
    parser.add_argument("--tex", type=Path, required=True)
    parser.add_argument("--figure", type=Path, required=True)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()

    for path in (args.script, args.tex, args.figure):
        if not path.is_file():
            raise FileNotFoundError(path)

    functions = load_functions(args.script)
    create_graph_matrices = functions["create_graph_matrices"]
    exact_evolution = functions["quantum_evolution_exact"]
    approximate_evolution = functions["quantum_evolution_approx"]

    graph_results = []
    for graph_type in GRAPH_TYPES:
        _, normalized_adjacency, _ = create_graph_matrices(8, graph_type)
        assert np.allclose(normalized_adjacency, normalized_adjacency.conj().T)

        exact = exact_evolution(normalized_adjacency, 1.0, 1.0)
        assert np.allclose(exact.conj().T @ exact, np.eye(8), atol=1e-12)

        for order in (1, 2, 3):
            actual = approximate_evolution(
                normalized_adjacency, 1.0, 1.0, order=order
            )
            independent = independent_taylor(
                normalized_adjacency, 1.0, 1.0, order
            )
            assert np.allclose(actual, independent, atol=1e-13)

        second_order = approximate_evolution(
            normalized_adjacency, 1.0, 1.0, order=2
        )
        complex_error = np.abs(exact - second_order)
        modulus_only_error = np.abs(np.abs(exact) - np.abs(second_order))
        maximum = float(complex_error.max())
        assert np.isclose(maximum, EXPECTED_MAX_ERRORS[graph_type], atol=1e-12)
        assert np.all(modulus_only_error <= complex_error + 1e-14)

        convergence = []
        for t in (0.2, 0.1, 0.05):
            exact_t = exact_evolution(normalized_adjacency, t, 1.0)
            approx_t = approximate_evolution(
                normalized_adjacency, t, 1.0, order=2
            )
            convergence.append(float(np.linalg.norm(exact_t - approx_t, ord=2)))
        ratios = [convergence[0] / convergence[1], convergence[1] / convergence[2]]
        assert all(7.85 < ratio < 8.15 for ratio in ratios)

        graph_results.append(
            {
                "graph_type": graph_type,
                "max_abs_element_error": maximum,
                "modulus_only_max_error": float(modulus_only_error.max()),
                "spectral_error_t_0.2_0.1_0.05": convergence,
                "halving_time_error_ratios": ratios,
            }
        )

    tex = args.tex.read_text(encoding="utf-8-sig")
    for value in ("0.0278", "0.0543", "0.0204"):
        assert value in tex, f"caption value {value} is missing"
    assert "same color scale" in tex
    assert "\\Delta G = |G(t) - G_2(t)|" in tex

    with Image.open(args.figure) as image:
        width, height = image.size
        image_format = image.format
        assert image_format == "PNG"
        assert width >= 2500 and height >= 1400

    report = {
        "status": "PASS",
        "scope": "Fig. 4 operator-error definition and rendering contract",
        "fidelity_definition_tested": False,
        "inputs": {
            str(path.resolve()): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in (args.script, args.tex, args.figure)
        },
        "physics": {
            "exact": "exp(-i * A_hat * t)",
            "approximation": "I - i*A_hat*t - A_hat^2*t^2/2",
            "error": "elementwise |G(t) - G2(t)|",
            "convergence_expectation": "second-order truncation error is O(t^3)",
        },
        "graphs": graph_results,
        "caption_values_verified": [0.0278, 0.0543, 0.0204],
        "image": {"format": image_format, "width": width, "height": height},
    }

    rendered = json.dumps(report, indent=2, ensure_ascii=False)
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()
