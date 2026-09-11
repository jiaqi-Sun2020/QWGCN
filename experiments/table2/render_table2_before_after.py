"""Render the Project 5 Table II before/after review sheet.

The left table is read from the Project 5 audit artifact.  The right table
retains the manuscript baselines and replaces only QWGCN with the verified
five-seed rerun.  Legacy hard-label MSE/error and probability Brier score are
kept in separate columns so the visual does not imply a false comparison.
"""

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
GRID = "#d9dee7"
BEFORE = "#c92f35"
BEFORE_FILL = "#fff1f1"
AFTER = "#168a4b"
AFTER_FILL = "#eaf9f0"
AMBER = "#b77700"
AMBER_FILL = "#fff6df"
WHITE = "#ffffff"
ROW_ALT = "#f7f9fc"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render the Project 5 Table II before/after comparison PNG."
    )
    parser.add_argument("--audit-json", type=Path, required=True)
    parser.add_argument("--aggregate-json", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def load_inputs(audit_path: Path, aggregate_path: Path) -> tuple[list[dict], dict]:
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    aggregate = json.loads(aggregate_path.read_text(encoding="utf-8"))

    old_rows = audit.get("paper_values")
    new_models = aggregate.get("models")
    if not isinstance(old_rows, list) or len(old_rows) != 7:
        raise ValueError("Expected seven manuscript Table II rows in paper_values.")
    if not isinstance(new_models, list) or len(new_models) != 1:
        raise ValueError("Expected exactly one QWGCN aggregate row.")

    new_qwgcn = new_models[0]
    if new_qwgcn.get("model") != "QWGCN" or new_qwgcn.get("run_count") != 5:
        raise ValueError("Expected a complete five-seed QWGCN aggregate.")
    return old_rows, new_qwgcn


def mean_sd(mean: float, sd: float) -> str:
    return f"{mean:.3f} ± {sd:.3f}"


def style_table(
    table,
    *,
    highlight_last: bool,
    header_color: str,
    font_size: float = 9.1,
) -> None:
    table.auto_set_font_size(False)
    table.set_fontsize(font_size)
    table.scale(1.0, 1.58)
    cells = table.get_celld()
    max_row = max(row for row, _ in cells)
    for (row, col), cell in cells.items():
        cell.set_edgecolor(GRID)
        cell.set_linewidth(0.8)
        if row == 0:
            cell.set_facecolor(header_color)
            cell.get_text().set_color(WHITE)
            cell.get_text().set_weight("bold")
        else:
            cell.set_facecolor(ROW_ALT if row % 2 == 0 else WHITE)
            cell.get_text().set_color(NAVY)
        if highlight_last and row == max_row:
            cell.set_facecolor(AFTER_FILL if header_color == AFTER else BEFORE_FILL)
            cell.get_text().set_weight("bold")
            cell.get_text().set_color(AFTER if header_color == AFTER else BEFORE)
        if col == 0:
            cell.get_text().set_ha("left")


def draw_panel_border(fig, bounds: tuple[float, float, float, float], color: str) -> None:
    x, y, width, height = bounds
    fig.add_artist(
        Rectangle(
            (x, y),
            width,
            height,
            transform=fig.transFigure,
            fill=False,
            linewidth=2.8,
            edgecolor=color,
        )
    )


def add_summary_card(
    fig,
    bounds: tuple[float, float, float, float],
    *,
    edge: str,
    fill: str,
    heading: str,
    metric_lines: list[str],
    footer: str,
    metric_columns: int = 1,
) -> None:
    x, y, width, height = bounds
    fig.add_artist(
        FancyBboxPatch(
            (x, y),
            width,
            height,
            transform=fig.transFigure,
            boxstyle="round,pad=0.010,rounding_size=0.012",
            linewidth=2.2,
            edgecolor=edge,
            facecolor=fill,
        )
    )
    fig.text(x + 0.018, y + height - 0.038, heading, fontsize=16, weight="bold", color=edge)
    for index, line in enumerate(metric_lines):
        column = index % metric_columns
        row = index // metric_columns
        fig.text(
            x + 0.018 + column * (width / metric_columns),
            y + height - 0.080 - row * 0.043,
            line,
            fontsize=11.4,
            family="DejaVu Sans Mono",
            color=NAVY,
        )
    fig.text(x + 0.018, y + 0.012, footer, fontsize=8.8, color=MUTED)


def render(audit_path: Path, aggregate_path: Path, output_path: Path) -> None:
    old_rows, new = load_inputs(audit_path, aggregate_path)
    old_qwgcn = next(row for row in old_rows if row["model"] == "QWGCN")

    old_table = [
        [
            row["model"],
            mean_sd(row["f1_mean"], row["f1_sd"]),
            mean_sd(row["mse_mean"], row["mse_sd"]),
            mean_sd(row["acc_mean"], row["acc_sd"]),
        ]
        for row in old_rows
    ]

    after_table: list[list[str]] = []
    for row in old_rows:
        if row["model"] == "QWGCN":
            after_table.append(
                [
                    "QWGCN (corrected)",
                    mean_sd(new["f1_mean"], new["f1_sample_sd"]),
                    mean_sd(new["error_rate_mean"], new["error_rate_sample_sd"]),
                    mean_sd(new["mse_mean"], new["mse_sample_sd"]),
                    mean_sd(new["accuracy_mean"], new["accuracy_sample_sd"]),
                ]
            )
        else:
            after_table.append(
                [
                    row["model"],
                    mean_sd(row["f1_mean"], row["f1_sd"]),
                    mean_sd(row["mse_mean"], row["mse_sd"]),
                    "—",
                    mean_sd(row["acc_mean"], row["acc_sd"]),
                ]
            )

    f1_delta = new["f1_mean"] - old_qwgcn["f1_mean"]
    acc_delta = new["accuracy_mean"] - old_qwgcn["acc_mean"]
    f1_sd_change = new["f1_sample_sd"] / old_qwgcn["f1_sd"] - 1.0
    acc_sd_change = new["accuracy_sample_sd"] / old_qwgcn["acc_sd"] - 1.0

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "axes.unicode_minus": False,
            "figure.facecolor": WHITE,
        }
    )
    fig = plt.figure(figsize=(15.5, 9.2), dpi=140, facecolor=WHITE)

    fig.text(
        0.035,
        0.960,
        "Project 5 - Table II Performance Evidence",
        fontsize=22,
        weight="bold",
        color=NAVY,
    )
    fig.text(0.292, 0.908, "BEFORE", fontsize=17, weight="bold", color=BEFORE, ha="center")
    fig.text(0.755, 0.908, "AFTER", fontsize=17, weight="bold", color=AFTER, ha="center")

    left_bounds = (0.025, 0.535, 0.425, 0.350)
    right_bounds = (0.470, 0.535, 0.505, 0.350)
    draw_panel_border(fig, left_bounds, BEFORE)
    draw_panel_border(fig, right_bounds, AFTER)

    left_ax = fig.add_axes((0.040, 0.555, 0.395, 0.300))
    left_ax.axis("off")
    left_ax.text(
        0.0,
        1.050,
        "Manuscript Table II (reported mean ± SD)",
        transform=left_ax.transAxes,
        fontsize=11.0,
        weight="bold",
        color=NAVY,
    )
    left_ax.text(
        0.0,
        0.997,
        "Independent run membership cannot be reconstructed.",
        transform=left_ax.transAxes,
        fontsize=9.1,
        color=MUTED,
    )
    left_table_artist = left_ax.table(
        cellText=old_table,
        colLabels=["Model", "F1", "MSE", "Acc"],
        colWidths=[0.28, 0.24, 0.24, 0.24],
        cellLoc="center",
        loc="center",
    )
    style_table(left_table_artist, highlight_last=True, header_color=BEFORE)

    right_ax = fig.add_axes((0.485, 0.555, 0.475, 0.300))
    right_ax.axis("off")
    right_ax.text(
        0.0,
        1.050,
        "Evidence table (only QWGCN rerun; 5 seeds)",
        transform=right_ax.transAxes,
        fontsize=11.0,
        weight="bold",
        color=NAVY,
    )
    right_ax.text(
        0.0,
        0.997,
        "Legacy baselines retained; probability Brier is a separate metric.",
        transform=right_ax.transAxes,
        fontsize=9.1,
        color=MUTED,
    )
    right_table_artist = right_ax.table(
        cellText=after_table,
        colLabels=["Model", "F1", "Hard err.", "Brier", "Acc"],
        colWidths=[0.245, 0.19, 0.19, 0.18, 0.195],
        cellLoc="center",
        loc="center",
    )
    style_table(
        right_table_artist,
        highlight_last=True,
        header_color=AFTER,
        font_size=8.7,
    )

    fig.text(
        0.035,
        0.480,
        "QWGCN - direct comparison",
        fontsize=15.5,
        weight="bold",
        color=NAVY,
    )
    add_summary_card(
        fig,
        (0.045, 0.270, 0.405, 0.165),
        edge=BEFORE,
        fill=BEFORE_FILL,
        heading="BEFORE  ·  manuscript row",
        metric_lines=[
            f"F1       {mean_sd(old_qwgcn['f1_mean'], old_qwgcn['f1_sd'])}",
            f"MSE      {mean_sd(old_qwgcn['mse_mean'], old_qwgcn['mse_sd'])}",
            f"Accuracy {mean_sd(old_qwgcn['acc_mean'], old_qwgcn['acc_sd'])}",
        ],
        footer="Status: reported aggregates; original independent runs unavailable",
        metric_columns=2,
    )
    add_summary_card(
        fig,
        (0.515, 0.270, 0.440, 0.165),
        edge=AFTER,
        fill=AFTER_FILL,
        heading="AFTER  ·  corrected QWGCN",
        metric_lines=[
            f"F1       {mean_sd(new['f1_mean'], new['f1_sample_sd'])}",
            f"Hard err {mean_sd(new['error_rate_mean'], new['error_rate_sample_sd'])}",
            f"Brier    {mean_sd(new['mse_mean'], new['mse_sample_sd'])}",
            f"Accuracy {mean_sd(new['accuracy_mean'], new['accuracy_sample_sd'])}",
        ],
        footer="Status: seeds 42-46; per-sample test predictions verified",
        metric_columns=2,
    )

    fig.text(
        0.500,
        0.225,
        (
            f"F1 {f1_delta * 100:+.2f} pp    |    Accuracy {acc_delta * 100:+.2f} pp"
            f"    |    F1 SD {f1_sd_change * 100:+.0f}%    |    Acc SD {acc_sd_change * 100:+.0f}%"
        ),
        ha="center",
        fontsize=11.4,
        weight="bold",
        color=NAVY,
    )
    fig.text(
        0.500,
        0.190,
        "Directional legacy-baseline position: F1 2nd; Accuracy 3rd exact (tied 2nd at 3 decimals).",
        ha="center",
        fontsize=10.0,
        color=MUTED,
    )

    fig.add_artist(
        FancyBboxPatch(
            (0.055, 0.065),
            0.890,
            0.082,
            transform=fig.transFigure,
            boxstyle="round,pad=0.010,rounding_size=0.010",
            linewidth=1.8,
            edgecolor=AMBER,
            facecolor=AMBER_FILL,
        )
    )
    fig.text(0.075, 0.120, "COMPARABILITY LIMIT", fontsize=11.5, weight="bold", color=AMBER)
    fig.text(
        0.075,
        0.091,
        "Probability Brier (0.211) is not comparable to legacy hard-label MSE (0.299).",
        fontsize=10.2,
        color=NAVY,
    )
    fig.text(
        0.075,
        0.071,
        "Baselines were not retrained; AFTER is directional evidence, not a controlled new ranking.",
        fontsize=10.2,
        color=NAVY,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight", facecolor=WHITE)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    render(args.audit_json.resolve(), args.aggregate_json.resolve(), args.output.resolve())


if __name__ == "__main__":
    main()
