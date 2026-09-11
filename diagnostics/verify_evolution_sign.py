"""Read-only, deterministic audit of the current evolution-sign convention.

Run from the corrected repository root inside the full QWGCN/paper workspace:
    python -B diagnostics/verify_evolution_sign.py

Requires NumPy, SciPy and NetworkX; no training, checkpoints, or plots are run.
The model's three arithmetic assignments are extracted verbatim via AST and
evaluated on NumPy arrays. This checks the polynomial, NOT a full PyTorch/PyG
forward pass. Named figure functions are extracted without executing plotting
entry points. Existing files are never modified.
"""

import ast
import hashlib
import json
import math
from pathlib import Path
from types import SimpleNamespace

import networkx as nx
import numpy as np
from scipy.linalg import expm


PROJECT = Path(__file__).resolve().parents[1]
MODEL = PROJECT / "model" / "QWGCN4.py"
PAPER = PROJECT.parents[1] / "paper" / "paper_build" / "PAPER_PRA"
FIGURES = PAPER / "img_code_ai" / "近似保真度计算"


def polynomial(a, t, sign):
    return np.eye(len(a)) + sign * 1j * t * a - t**2 * (a @ a) / 2


def spectral(a):
    return float(np.linalg.norm(a, 2))


def load_figure_functions(path):
    wanted = {
        "create_graph_matrices", "quantum_evolution_exact",
        "quantum_evolution_approx", "calculate_average_gate_fidelity",
    }
    tree = ast.parse(path.read_text(encoding="utf-8-sig"))
    nodes = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in wanted]
    env = {"np": np, "nx": nx, "expm": expm, "math": math}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), "exec"), env)
    return env


def model_arithmetic(a, x, t):
    tree = ast.parse(MODEL.read_text(encoding="utf-8-sig"))
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef)
               and n.name == "FastQuantumEvolution")
    forward = next(n for n in cls.body if isinstance(n, ast.FunctionDef)
                   and n.name == "forward")
    selected = []
    for n in ast.walk(forward):
        if isinstance(n, ast.Assign) and any(
            isinstance(k, ast.Name) and k.id in {"evolved_x", "second_coeff"}
            for k in n.targets
        ):
            selected.append(n)
    selected.sort(key=lambda n: n.lineno)
    assert len(selected) == 3, "Model arithmetic changed; re-review extraction."
    env = dict(x_complex=x, first_order=a @ x, second_order=a @ a @ x,
               self=SimpleNamespace(evolution_time=t, diffusion_strength=1.0))
    exec(compile(ast.Module(body=selected, type_ignores=[]), str(MODEL), "exec"), env)
    return env["evolved_x"], [n.lineno for n in selected]


def main():
    result = {"method": __doc__, "numpy": np.__version__, "networkx": nx.__version__}
    inputs = [MODEL, PAPER / "20260907" / "refQWGCN0105.tex",
              PAPER / "20260907" / "refQWGCN0105.pdf",
              FIGURES / "平均门保真度2.py", FIGURES / "node_e_time3.py"]
    result["input_sha256"] = {
        str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in inputs
    }

    # The initial derivative fixes the Hamiltonian; it cannot be inferred from
    # node probabilities alone. This counterexample is an undirected 2-node graph.
    a = np.array([[0., 1.], [1., 0.]])
    plus_generator = 1j * (1j * a)
    minus_generator = 1j * (-1j * a)
    assert np.allclose(plus_generator, -a)
    assert np.allclose(minus_generator, a)
    result["schrodinger_generator"] = {"positive_exponent": "H=-A", "negative_exponent": "H=+A"}

    rows = []
    for t in (0.1, 0.05, 0.025):
        u = expm(-1j * a * t)
        rows.append({"t": t, "wrong_sign_error": spectral(u-polynomial(a,t,1)),
                     "correct_sign_error": spectral(u-polynomial(a,t,-1))})
    wrong_ratio = rows[0]["wrong_sign_error"] / rows[1]["wrong_sign_error"]
    correct_ratio = rows[0]["correct_sign_error"] / rows[1]["correct_sign_error"]
    assert 1.99 < wrong_ratio < 2.01 and 7.99 < correct_ratio < 8.01
    result["convergence"] = rows
    result["halving_time_error_ratio"] = {"wrong": wrong_ratio, "correct": correct_ratio}

    t = math.pi / 4
    um, up = expm(-1j*t*a), expm(1j*t*a)
    real_state = np.array([1., 0.], dtype=complex)
    complex_state = np.array([1., 1j]) / math.sqrt(2)
    pr, qr = abs(um@real_state)**2, abs(up@real_state)**2
    pc, qc = abs(um@complex_state)**2, abs(up@complex_state)**2
    assert np.allclose(pr, qr) and np.allclose(pc, [1,0]) and np.allclose(qc, [0,1])
    result["two_node_counterexample"] = {
        "time": t, "real_state_prob_negative": pr.tolist(),
        "real_state_prob_positive": qr.tolist(),
        "complex_state": "[1,i]/sqrt(2)",
        "complex_state_prob_negative": pc.tolist(), "complex_state_prob_positive": qc.tolist(),
    }

    # Component-wise complex ReLU is not conjugation-equivariant.
    def crelu(z):
        return np.maximum(z.real, 0) + 1j*np.maximum(z.imag, 0)
    assert not np.allclose(abs(crelu(um@real_state)), abs(crelu(up@real_state)))
    result["complex_relu"] = {
        "negative_output_magnitudes": abs(crelu(um@real_state)).tolist(),
        "positive_output_magnitudes": abs(crelu(up@real_state)).tolist(),
    }

    # Code adds self loops then symmetrically normalizes. Test on an irregular
    # undirected graph with no pre-existing loops; B is the effective adjacency.
    raw = np.array([[0,1,0,0],[1,0,1,1],[0,1,0,1],[0,1,1,0]], dtype=float)
    looped = raw + np.eye(4)
    inv = np.diag(1/np.sqrt(looped.sum(axis=0)))
    b = inv @ looped @ inv
    rng = np.random.default_rng(20260906)
    x = rng.normal(size=(4,3)) + 1j*rng.normal(size=(4,3))
    y, lines = model_arithmetic(b,x,0.5)
    yp = polynomial(b,0.5,1)@x
    ym = polynomial(b,0.5,-1)@x
    assert np.allclose(y,ym) and not np.allclose(y,yp)
    flipped, _ = model_arithmetic(b,x,-0.5)
    assert np.allclose(flipped,yp)
    result["actual_model_arithmetic"] = {
        "source_lines": lines, "matches_negative_polynomial": True,
        "positive_residual": float(np.max(abs(y-yp))),
        "negative_residual": float(np.max(abs(y-ym))),
        "negative_time_equals_positive_polynomial": True,
        "limitation": "AST arithmetic only; full PyTorch/PyG model was not executed.",
    }

    p = polynomial(b,1.,-1)
    defect = p.conj().T @ p - np.eye(len(b))
    assert np.allclose(defect, np.linalg.matrix_power(b,4)/4)
    result["unitarity_defect_t1"] = spectral(defect)
    assert abs(result["unitarity_defect_t1"] - .25) < 1e-12

    env = load_figure_functions(FIGURES / "平均门保真度2.py")
    figure_rows = []
    for graph in ("random", "star", "complete"):
        _, bhat, _ = env["create_graph_matrices"](8,graph)
        u = env["quantum_evolution_exact"](bhat,1.,1.)
        approximation = env["quantum_evolution_approx"](bhat,1.,1.,order=2)
        corrected = polynomial(bhat,1.,-1)
        assert np.allclose(approximation, corrected)
        d = len(bhat)
        unbounded_corrected_score = (d+abs(np.trace(u.conj().T@corrected))**2)/(d*(d+1))
        figure_rows.append({
            "graph": graph,
            "approximation_matches_negative_sign": True,
            "current_clipped_score": float(env["calculate_average_gate_fidelity"](u,approximation)),
            "raw_score_not_valid_gate_fidelity": float(unbounded_corrected_score),
            "modulus_only_heatmap_max": float(np.max(abs(abs(u)-abs(approximation)))),
            "complex_difference_max": float(np.max(abs(u-approximation))),
        })
    result["corrected_figure4_functions_n8_t1"] = figure_rows

    # Figure 5 prints the nearest sampled time, not exactly the requested time.
    env5 = load_figure_functions(FIGURES / "node_e_time3.py")
    times = np.linspace(.01,1.5,50)
    figure5_rows = []
    for graph in ("random", "star", "complete"):
        bhat = env5["create_graph_matrices"](15,graph)
        scores = [env5["calculate_average_gate_fidelity"](
            env5["quantum_evolution_exact"](bhat,t,1.),
            env5["quantum_evolution_approx"](bhat,t,1.,2)) for t in times]
        i05, i1 = np.argmin(abs(times-.5)), np.argmin(abs(times-1.))
        figure5_rows.append({"graph":graph, "sample_for_t0.5":float(times[i05]),
                             "sample_for_t1":float(times[i1]),
                             "score_labeled_t0.5":float(scores[i05]),
                             "score_labeled_t1":float(scores[i1])})
    result["figure5_n15_existing_scores"] = figure5_rows
    assert [round(r["complex_difference_max"],4) for r in figure_rows] == [.0278,.0543,.0204]
    assert [round(r["score_labeled_t1"],3) for r in figure5_rows] == [.554,.796,.814]

    # Complete graph with loops: B=J/N. Increasing N hides the wrong sign in
    # the averaged trace score, while the worst-case operator error is fixed.
    result["dimension_counterexample"] = []
    qwrong = .5+1j
    for n in (8,300):
        tr = n-1 + np.exp(1j)*qwrong
        score = (n+abs(tr)**2)/(n*(n+1))
        result["dimension_counterexample"].append({"nodes":n, "wrong_trace_score":float(score),
                                                  "wrong_spectral_error":float(abs(np.exp(-1j)-qwrong))})
    result["checks"] = "PASS: convention, convergence, phase counterexamples, extracted arithmetic, nonunitarity"
    print(json.dumps(result, indent=2, ensure_ascii=True))


if __name__ == "__main__":
    main()
