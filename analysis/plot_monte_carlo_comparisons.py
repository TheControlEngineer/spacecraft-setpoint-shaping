"""
Plot Monte Carlo comparison charts for all controller/method combinations.

Reads monte_carlo_runs.csv and generates summary comparison plots for:
  - Unshaped + Standard PD
  - Unshaped + Filtered PD
  - Fourth-order + Standard PD
  - Fourth-order + Filtered PD
"""

from __future__ import annotations

import argparse
import csv
import os
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np


COMBOS = [
    ("unshaped_standard_pd", "Unshaped + Standard PD", "#d62728"),  # red
    ("unshaped_filtered_pd", "Unshaped + Filtered PD", "#ff7f0e"),  # orange
    ("fourth_standard_pd", "Fourth-order + Standard PD", "#1f77b4"),  # blue
    ("fourth_filtered_pd", "Fourth-order + Filtered PD", "#9467bd"),  # violet
]

METRICS = [
    ("rms_pointing_error_deg", "RMS Pointing Error (deg)"),
    ("peak_pointing_error_deg", "Peak Pointing Error (deg)"),
    ("rms_vibration_mm", "RMS Vibration (mm)"),
    ("peak_torque_nm", "Peak Torque (N·m)"),
    ("rms_torque_nm", "RMS Torque (N·m)"),
    ("rw_saturation_percent", "RW Saturation (%)"),
]


def _load_rows(path: str) -> List[Dict[str, str]]:
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        return [row for row in reader]


def _get_metric_values(rows: List[Dict[str, str]], column: str) -> np.ndarray:
    values: List[float] = []
    for row in rows:
        raw = row.get(column, "")
        try:
            val = float(raw)
        except (TypeError, ValueError):
            continue
        if np.isfinite(val):
            values.append(val)
    return np.array(values, dtype=float)


def _plot_box_comparisons(rows: List[Dict[str, str]], out_path: str) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    axes = axes.flatten()

    legend_handles = []
    legend_labels = []

    for idx, (metric, title) in enumerate(METRICS):
        ax = axes[idx]
        data = []
        labels = []
        colors = []
        for combo_key, combo_label, color in COMBOS:
            col = f"{combo_key}_{metric}"
            vals = _get_metric_values(rows, col)
            if len(vals) == 0:
                continue
            data.append(vals)
            labels.append(combo_label)
            colors.append(color)

        if not data:
            ax.set_title(f"{title}\n(no data)")
            ax.axis("off")
            continue

        bp = ax.boxplot(
            data,
            labels=[lbl.replace(" + ", "\n+ ") for lbl in labels],
            patch_artist=True,
            showfliers=True,
        )
        for patch, color in zip(bp["boxes"], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        for median in bp["medians"]:
            median.set_color("black")
        ax.set_title(title, fontweight="bold")
        ax.grid(True, alpha=0.3)

        if not legend_handles:
            for combo_key, combo_label, color in COMBOS:
                legend_handles.append(plt.Line2D([0], [0], color=color, linewidth=6))
                legend_labels.append(combo_label)

    for idx in range(len(METRICS), len(axes)):
        axes[idx].axis("off")

    fig.suptitle("Monte Carlo Comparisons (Combined Feedforward + Feedback)", fontweight="bold")
    fig.legend(legend_handles, legend_labels, loc="lower center", ncol=2, fontsize=9)
    fig.tight_layout(rect=[0, 0.05, 1, 0.96])
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description="Plot Monte Carlo comparison charts.")
    parser.add_argument(
        "--csv",
        default=os.path.join(os.path.dirname(__file__), "monte_carlo_runs.csv"),
        help="Path to monte_carlo_runs.csv",
    )
    parser.add_argument(
        "--out-dir",
        default=os.path.dirname(__file__),
        help="Directory to save plots",
    )
    args = parser.parse_args()

    if not os.path.isfile(args.csv):
        print(f"CSV not found: {args.csv}")
        return 1

    rows = _load_rows(args.csv)
    if not rows:
        print(f"No data in CSV: {args.csv}")
        return 1

    os.makedirs(args.out_dir, exist_ok=True)
    out_path = os.path.join(args.out_dir, "monte_carlo_comparisons_box.png")
    _plot_box_comparisons(rows, out_path)
    print(f"Saved {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
