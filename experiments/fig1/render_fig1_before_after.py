"""Render the Project 6 Fig. 1 before/after review sheet."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle


NAVY = "#1f2a40"
MUTED = "#5e6878"
BEFORE = "#c92f35"
BEFORE_FILL = "#fff1f1"
AFTER = "#168a4b"
AFTER_FILL = "#eaf9f0"
AMBER = "#b77700"
AMBER_FILL = "#fff6df"
WHITE = "#ffffff"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--figure", type=Path, required=True)
    parser.add_argument("--verification", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def card(fig, bounds, *, edge: str, fill: str, heading: str, lines: list[str]) -> None:
    x, y, width, height = bounds
    fig.add_artist(
        FancyBboxPatch(
            (x, y),
            width,
            height,
            transform=fig.transFigure,
            boxstyle="round,pad=0.010,rounding_size=0.012",
            linewidth=2.3,
            edgecolor=edge,
            facecolor=fill,
        )
    )
    fig.text(x + 0.022, y + height - 0.052, heading, color=edge, fontsize=18, weight="bold")
    for index, line in enumerate(lines):
        fig.text(x + 0.022, y + height - 0.095 - 0.035 * index, line, color=NAVY, fontsize=12.0)


def main() -> None:
    args = parse_args()
    verification = json.loads(args.verification.read_text(encoding="utf-8"))
    if verification.get("status") != "PASS":
        raise ValueError("The Fig. 1 verification report must pass before rendering.")
    metrics = verification["metrics"]
    image = plt.imread(args.figure)
    graph_crop = image[:, : int(image.shape[1] * 0.405)]

    plt.rcParams.update({"font.family": "DejaVu Sans", "figure.facecolor": WHITE})
    fig = plt.figure(figsize=(15.5, 9.2), dpi=140, facecolor=WHITE)
    fig.text(0.035, 0.955, "Project 6 - Fig. 1 Edge-count Consistency", fontsize=23, weight="bold", color=NAVY)
    fig.text(
        0.035,
        0.914,
        "Only the manuscript count changes; the graph topology, figure, and CTQW data remain unchanged.",
        fontsize=11.5,
        color=MUTED,
    )

    fig.text(0.265, 0.865, "BEFORE", fontsize=18, weight="bold", color=BEFORE, ha="center")
    fig.text(0.735, 0.865, "AFTER", fontsize=18, weight="bold", color=AFTER, ha="center")

    for x, color in ((0.035, BEFORE), (0.515, AFTER)):
        fig.add_artist(
            Rectangle(
                (x, 0.465),
                0.450,
                0.365,
                transform=fig.transFigure,
                fill=False,
                linewidth=2.8,
                edgecolor=color,
            )
        )
        ax = fig.add_axes((x + 0.018, 0.500, 0.414, 0.290))
        ax.imshow(graph_crop)
        ax.axis("off")

    fig.text(0.260, 0.485, "Same 20-edge topology", fontsize=11.0, color=MUTED, ha="center")
    fig.text(0.740, 0.485, "Same 20-edge topology", fontsize=11.0, color=MUTED, ha="center")

    card(
        fig,
        (0.045, 0.225, 0.425, 0.185),
        edge=BEFORE,
        fill=BEFORE_FILL,
        heading="BEFORE  ·  manuscript inconsistency",
        lines=[
            'Caption: “10 vertices and 19 edges”',
            'Body: “10 vertices and 19 edges”',
            "Conflict: adjacency matrix contains 20 edges.",
        ],
    )
    card(
        fig,
        (0.530, 0.225, 0.425, 0.185),
        edge=AFTER,
        fill=AFTER_FILL,
        heading="AFTER  ·  evidence-aligned wording",
        lines=[
            'Caption: “10 vertices and 20 edges”',
            'Body: “10 vertices and 20 edges”',
            "Figure and numerical experiment are unchanged.",
        ],
    )

    fig.add_artist(
        FancyBboxPatch(
            (0.045, 0.070),
            0.910,
            0.110,
            transform=fig.transFigure,
            boxstyle="round,pad=0.010,rounding_size=0.010",
            linewidth=1.9,
            edgecolor=AMBER,
            facecolor=AMBER_FILL,
        )
    )
    fig.text(0.065, 0.145, "FIRST-PRINCIPLES CHECK", fontsize=12.2, weight="bold", color=AMBER)
    fig.text(
        0.065,
        0.108,
        (
            f"sum(A) = {metrics['adjacency_sum']:.0f}  →  |E| = sum(A)/2 = {metrics['edge_count']}    |    "
            f"deg(0) = deg(2) = {metrics['max_degree']} (maximum)"
        ),
        fontsize=11.4,
        family="DejaVu Sans Mono",
        color=NAVY,
    )
    fig.text(
        0.065,
        0.079,
        (
            f"t = 0.5:  P(node 0) = {metrics['probability_node_0']:.3f},  "
            f"P(node 2) = {metrics['probability_node_2']:.3f}    |    Figure SHA-256 unchanged"
        ),
        fontsize=11.0,
        color=NAVY,
    )

    args.output.resolve().parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output.resolve(), bbox_inches="tight", facecolor=WHITE)
    plt.close(fig)


if __name__ == "__main__":
    main()
