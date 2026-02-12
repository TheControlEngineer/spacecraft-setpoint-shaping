"""
Replot factor sweep CSVs (no simulation required) with semilogy scales.

Reads the existing mc_sweep_*.csv files and produces plots using
log y axes for non dB metrics.
"""

from __future__ import annotations

import argparse
import csv
import os
from collections import defaultdict
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np


COMBOS = [
    ("s_curve_standard_pd", "S-curve + Standard PD", "#ff7f0e"),
    ("s_curve_filtered_pd", "S-curve + Filtered PD", "#ffbb78"),
    ("fourth_standard_pd", "Fourth-order + Standard PD", "#1f77b4"),
    ("fourth_filtered_pd", "Fourth-order + Filtered PD", "#aec7e8"),
]

METRICS = [
    ("rms_vibration_mm", "RMS Vibration (mm)"),
    ("rms_pointing_error_deg", "RMS Pointing Error (deg)"),
    ("peak_torque_nm", "Peak Torque (N·m)"),
    ("rms_torque_nm", "RMS Torque (N·m)"),
]

FREQ_METRICS = [
    ("pointing_band_rms_deg", "Pointing RMS near f0 (deg)"),
    ("vibration_band_rms_mm", "Vibration RMS near f0 (mm)"),
    ("pointing_psd_at_f0_db", "Pointing PSD at f0 (dB)"),
    ("vibration_psd_at_f0_db", "Vibration PSD at f0 (dB)"),
]


def _load_csv(path: str) -> Dict[str, Dict[str, List[Tuple[float, float]]]]:
    """Parse a factor sweep CSV into {combo: {metric: [(x, y), ...]}} for plotting."""
    data: Dict[str, Dict[str, List[Tuple[float, float]]]] = defaultdict(lambda: defaultdict(list))
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            combo = row.get("combo")
            factor_val = row.get("factor_value")
            if not combo or factor_val is None:
                continue
            try:
                x = float(factor_val)
            except ValueError:
                continue
            for metric, _ in METRICS + FREQ_METRICS:
                val = row.get(metric)
                if val is None:
                    continue
                try:
                    y = float(val)
                except ValueError:
                    continue
                if np.isfinite(y):
                    data[combo][metric].append((x, y))
    return data


def _plot_from_data(
    data: Dict[str, Dict[str, List[Tuple[float, float]]]],
    title: str,
    x_label: str,
    out_path: str,
    log_x: bool,
    metrics: List[Tuple[str, str]],
    bins: int = 10,
) -> None:
    """Render a 2x2 scatter + binned median figure from pre loaded CSV data."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    axes = axes.flatten()

    for idx, (metric_key, metric_label) in enumerate(metrics):
        ax = axes[idx]
        for combo_key, combo_label, color in COMBOS:
            points = data.get(combo_key, {}).get(metric_key, [])
            if not points:
                continue
            arr = np.array(points, dtype=float)
            x = arr[:, 0]
            y = arr[:, 1]
            use_logy = not metric_key.endswith("_db")
            if use_logy:
                y = np.where(y > 0, y, np.nan)
            if log_x and use_logy:
                ax.loglog(x, y, linestyle="none", marker="o", markersize=3, alpha=0.25, color=color)
            elif log_x:
                ax.semilogx(x, y, linestyle="none", marker="o", markersize=3, alpha=0.25, color=color)
            elif use_logy:
                ax.semilogy(x, y, linestyle="none", marker="o", markersize=3, alpha=0.25, color=color)
            else:
                ax.plot(x, y, linestyle="none", marker="o", markersize=3, alpha=0.25, color=color)

            # Bin stats
            if bins > 1:
                if log_x:
                    edges = np.logspace(np.log10(np.min(x)), np.log10(np.max(x)), bins + 1)
                    centers = np.sqrt(edges[:-1] * edges[1:])
                else:
                    edges = np.linspace(np.min(x), np.max(x), bins + 1)
                    centers = 0.5 * (edges[:-1] + edges[1:])
                medians = []
                p10 = []
                p90 = []
                for i in range(bins):
                    mask = (x >= edges[i]) & (x < edges[i + 1])
                    vals = y[mask]
                    vals = vals[np.isfinite(vals)]
                    if vals.size == 0:
                        medians.append(np.nan)
                        p10.append(np.nan)
                        p90.append(np.nan)
                        continue
                    medians.append(np.percentile(vals, 50))
                    p10.append(np.percentile(vals, 10))
                    p90.append(np.percentile(vals, 90))
                medians = np.array(medians)
                p10 = np.array(p10)
                p90 = np.array(p90)
                if use_logy:
                    medians = np.where(medians > 0, medians, np.nan)
                    p10 = np.where(p10 > 0, p10, np.nan)
                    p90 = np.where(p90 > 0, p90, np.nan)
                if log_x and use_logy:
                    ax.loglog(centers, medians, color=color, linewidth=2, label=combo_label)
                elif log_x:
                    ax.semilogx(centers, medians, color=color, linewidth=2, label=combo_label)
                elif use_logy:
                    ax.semilogy(centers, medians, color=color, linewidth=2, label=combo_label)
                else:
                    ax.plot(centers, medians, color=color, linewidth=2, label=combo_label)
                ax.fill_between(centers, p10, p90, color=color, alpha=0.15)

        ax.set_title(metric_label, fontweight="bold")
        ax.set_xlabel(x_label)
        ax.grid(True, alpha=0.3)

    handles = [plt.Line2D([0], [0], color=c, linewidth=3) for _, _, c in COMBOS]
    labels = [label for _, label, _ in COMBOS]
    fig.suptitle(title, fontweight="bold")
    fig.legend(handles, labels, loc="lower center", ncol=2, fontsize=9)
    fig.tight_layout(rect=[0, 0.05, 1, 0.95])
    fig.savefig(out_path, dpi=170)
    plt.close(fig)


def main() -> int:
    """CLI entry point: load CSVs from disk and regenerate semilogy sweep plots."""
    parser = argparse.ArgumentParser(description="Replot factor sweep CSVs.")
    parser.add_argument("--dir", default=os.path.dirname(__file__), help="Directory containing mc_sweep_*.csv")
    parser.add_argument("--bins", type=int, default=10, help="Bins for percentile trend lines")
    args = parser.parse_args()

    base_dir = os.path.abspath(args.dir)
    csvs = {
        "mc_sweep_inertia.csv": ("Sweep: Inertia Error (±20%)", "Inertia scale factor", False, METRICS),
        "mc_sweep_modal_frequency.csv": ("Sweep: Modal Frequency Error", "Modal frequency scale factor", False, METRICS),
        "mc_sweep_modal_damping.csv": ("Sweep: Modal Damping Variation", "Modal damping scale factor", False, METRICS),
        "mc_sweep_disturbance_frequency.csv": ("Sweep: Disturbance Frequency (sinusoidal torque)",
                                              "Disturbance frequency (Hz)", True, METRICS),
        "mc_sweep_noise_level.csv": ("Sweep: Sensor Noise Frequency (sinusoidal)",
                                     "Sensor noise frequency (Hz)", True, METRICS),
    }

    for name, (title, xlabel, log_x, metrics) in csvs.items():
        path = os.path.join(base_dir, name)
        if not os.path.isfile(path):
            print(f"Missing CSV: {path}")
            continue
        data = _load_csv(path)
        out_name = os.path.splitext(name)[0] + "_semilogy.png"
        out_path = os.path.join(base_dir, out_name)
        _plot_from_data(data, title, xlabel, out_path, log_x, metrics, bins=args.bins)
        print(f"Saved {out_path}")

    # Targeted metrics plots if present
    targeted_csvs = {
        "mc_sweep_disturbance_frequency.csv": ("Sweep: Disturbance Frequency (frequency-targeted metrics)",
                                              "Disturbance frequency (Hz)", True, FREQ_METRICS,
                                              "mc_sweep_disturbance_frequency_targeted_semilogy.png"),
        "mc_sweep_noise_level.csv": ("Sweep: Sensor Noise Frequency (frequency-targeted metrics)",
                                     "Sensor noise frequency (Hz)", True, FREQ_METRICS,
                                     "mc_sweep_noise_level_targeted_semilogy.png"),
    }
    for name, (title, xlabel, log_x, metrics, out_name) in targeted_csvs.items():
        path = os.path.join(base_dir, name)
        if not os.path.isfile(path):
            continue
        data = _load_csv(path)
        if not data:
            continue
        out_path = os.path.join(base_dir, out_name)
        _plot_from_data(data, title, xlabel, out_path, log_x, metrics, bins=args.bins)
        print(f"Saved {out_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
