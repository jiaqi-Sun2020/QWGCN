"""Render the Project 7 Fig. 2 and manuscript/runtime consistency review sheet."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle


INK = "#202a3b"
MUTED = "#606b7b"
ORANGE = "#c56522"
ORANGE_FILL = "#fff3e8"
BLUE = "#2067a8"
BLUE_FILL = "#eaf4fc"
NAVY = "#17345c"
PALE = "#f6f8fb"
WHITE = "#ffffff"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--figure", type=Path, required=True)
    parser.add_argument("--verification", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def rounded_panel(fig, bounds, edge, fill, width=2.0) -> None:
    x, y, panel_width, panel_height = bounds
    fig.add_artist(
        FancyBboxPatch(
            (x, y),
            panel_width,
            panel_height,
            transform=fig.transFigure,
            boxstyle="round,pad=0.010,rounding_size=0.012",
            linewidth=width,
            edgecolor=edge,
            facecolor=fill,
        )
    )


def comparison_card(fig, bounds, *, edge, fill, heading, rows) -> None:
    rounded_panel(fig, bounds, edge, fill, width=2.3)
    x, y, width, height = bounds
    fig.text(x + 0.022, y + height - 0.050, heading, fontsize=17, weight="bold", color=edge)
    start = y + height - 0.100
    for index, (label, detail) in enumerate(rows):
        row_y = start - index * 0.043
        fig.text(x + 0.024, row_y, label, fontsize=10.8, weight="bold", color=INK)
        fig.text(x + 0.154, row_y, detail, fontsize=10.5, color=INK)


def main() -> None:
    args = parse_args()
    verification = json.loads(args.verification.read_text(encoding="utf-8"))
    if verification.get("status") != "PASS":
        raise ValueError("Project 7 verification must pass before rendering.")

    metrics = verification["metrics"]
    architecture = plt.imread(args.figure)

    # Chart contract: comparison/scorecard, static PNG. The analytical question is
    # whether Fig. 2 and the manuscript now describe the checked runtime contract.
    plt.rcParams.update({"font.family": "DejaVu Sans", "figure.facecolor": WHITE})
    fig = plt.figure(figsize=(16.2, 10.2), dpi=140, facecolor=WHITE)
    fig.text(0.035, 0.960, "Project 7 - Fig. 2 & Manuscript/Runtime Consistency", fontsize=23, weight="bold", color=INK)
    fig.text(
        0.035,
        0.922,
        "Fig. 2 was already correct and remains unchanged; the manuscript method and complexity statements are aligned to the executable code.",
        fontsize=11.5,
        color=MUTED,
    )

    fig.add_artist(Rectangle((0.035, 0.595), 0.595, 0.285, transform=fig.transFigure, fill=False, linewidth=2.2, edgecolor=NAVY))
    ax = fig.add_axes((0.050, 0.612, 0.565, 0.250))
    ax.imshow(architecture)
    ax.axis("off")
    fig.text(0.050, 0.568, "FIG. 2 AUDIT", fontsize=12.3, weight="bold", color=NAVY)
    fig.text(0.158, 0.568, r"$H=A$  |  $G(t)=e^{-iAt}$  |  readout: Global_mean -> Log_softmax", fontsize=11.4, color=INK)
    fig.text(0.050, 0.541, "No Taylor order or fixed-time assertion is embedded in the image. Pixel hash is unchanged.", fontsize=10.6, color=MUTED)

    rounded_panel(fig, (0.655, 0.595, 0.310, 0.285), BLUE, BLUE_FILL, width=2.2)
    fig.text(0.678, 0.833, "VERIFIED CONTRACT", fontsize=17, weight="bold", color=BLUE)
    contract_rows = [
        ("Exact CTQW", r"$G(t)=e^{-iAt}$"),
        ("Fast QWGCN", r"$G_2=I-it\widetilde{A}-t^2\widetilde{A}^2/2$"),
        ("Runtime", r"$t=1$, $K=2$, per layer"),
        ("Graph operator", r"$\widehat{A}=A+I$; symmetric normalization"),
        ("Readout", "pool node magnitudes, then log-softmax"),
    ]
    for index, (label, detail) in enumerate(contract_rows):
        y = 0.790 - index * 0.045
        fig.text(0.680, y, label, fontsize=10.5, weight="bold", color=INK)
        fig.text(0.785, y, detail, fontsize=10.1, color=INK)

    comparison_card(
        fig,
        (0.045, 0.180, 0.440, 0.320),
        edge=ORANGE,
        fill=ORANGE_FILL,
        heading="BEFORE  ·  manuscript contradictions",
        rows=[
            ("Readout", "pool(log-softmax(|X|)) in both pseudocodes"),
            ("Operator", "self-loops omitted from normalized adjacency"),
            ("State", "separate walker state chained between QWGCN layers"),
            ("Weight", "described as a CTQW edge weight"),
            ("Physics", "finite Taylor polynomial claimed to preserve unitarity"),
            ("Cost", "one edge pass; O(EF) time and O(N) memory"),
        ],
    )
    comparison_card(
        fig,
        (0.515, 0.180, 0.440, 0.320),
        edge=BLUE,
        fill=BLUE_FILL,
        heading="AFTER  ·  code-aligned definitions",
        rows=[
            ("Readout", "log-softmax(global-mean-pool(|X|))"),
            ("Operator", "A_hat=A+I, then symmetric normalization"),
            ("State", "Y^l=G2(t)X^l on current complex features"),
            ("Weight", "per-graph normalized node-wise ||Y_i||^2"),
            ("Physics", "G2 explicitly identified as nonunitary"),
            ("Cost", "O((K+k)EF + N Fin Fout); sparse memory"),
        ],
    )

    rounded_panel(fig, (0.045, 0.030, 0.910, 0.100), NAVY, PALE, width=1.8)
    fig.text(0.066, 0.095, "ADVERSARIAL CHECK", fontsize=12.2, weight="bold", color=NAVY)
    fig.text(
        0.066,
        0.064,
        (
            f"{verification['checks_passed']}/{verification['checks_total']} checks passed  |  "
            f"example ||G2*G2-I||2 = {metrics['unitarity_defect_spectral_norm']:.3f}  |  "
            f"||exp(-iAt)-G2||2 at t=1 = {metrics['second_order_error_spectral_norm_at_t1']:.3f}"
        ),
        fontsize=11.2,
        family="DejaVu Sans Mono",
        color=INK,
    )
    fig.text(
        0.066,
        0.039,
        "Scope guard: the deferred fidelity definition/data and the deferred small-graph split were not changed.",
        fontsize=10.6,
        color=MUTED,
    )

    args.output.resolve().parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output.resolve(), bbox_inches="tight", facecolor=WHITE)
    plt.close(fig)


if __name__ == "__main__":
    main()
