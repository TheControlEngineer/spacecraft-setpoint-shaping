"""
Sweep reaction wheel cant angle for the 3-wheel pyramid used in spacecraft_model.py.

Generates:
1) Condition number of Gs (allocation matrix)
2) Worst-case wheel torque amplification for unit body torque
3) Axis-wise max body torque (for unit wheel torque limit)

This uses the same wheel geometry as spacecraft_model.py:
    w1 = [ sin(a),  cos(a), 0 ]
    w2 = [-sin(a),  cos(a), 0 ]
    w3 = [ 0,       0,      1 ]

At a = 45 deg, this matches the model's gsHat_B definitions.
"""

from __future__ import annotations

import argparse
import math
import os
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt


def _gs_matrix(alpha_rad: float) -> np.ndarray:
    """Return 3x3 Gs matrix with wheel axes as columns."""
    s = math.sin(alpha_rad)
    c = math.cos(alpha_rad)
    w1 = np.array([s, c, 0.0])
    w2 = np.array([-s, c, 0.0])
    w3 = np.array([0.0, 0.0, 1.0])
    return np.column_stack([w1, w2, w3])


def _sphere_samples(n: int) -> np.ndarray:
    """Fibonacci sphere sampling for approximately uniform directions."""
    i = np.arange(n)
    phi = (1 + 5 ** 0.5) / 2  # golden ratio
    theta = 2 * np.pi * i / phi
    z = 1 - 2 * (i + 0.5) / n
    r = np.sqrt(1 - z ** 2)
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    return np.column_stack([x, y, z])


def sweep(alpha_deg_min: float, alpha_deg_max: float, alpha_deg_step: float, n_dirs: int) -> dict[str, np.ndarray]:
    alphas_deg = np.arange(alpha_deg_min, alpha_deg_max + 1e-9, alpha_deg_step)
    alphas_rad = np.deg2rad(alphas_deg)

    cond_vals = np.zeros_like(alphas_deg, dtype=float)
    worst_amp = np.zeros_like(alphas_deg, dtype=float)
    max_tau_x = np.zeros_like(alphas_deg, dtype=float)
    max_tau_y = np.zeros_like(alphas_deg, dtype=float)
    max_tau_z = np.zeros_like(alphas_deg, dtype=float)

    directions = _sphere_samples(n_dirs)

    for idx, a in enumerate(alphas_rad):
        gs = _gs_matrix(a)
        # 1) Condition number of allocation matrix
        cond_vals[idx] = np.linalg.cond(gs)

        # 2) Worst-case wheel torque amplification for unit body torque
        gs_pinv = np.linalg.pinv(gs)
        # u = -Gs^+ * tau ; amplification = ||u|| / ||tau||, but ||tau|| = 1
        u = -(gs_pinv @ directions.T).T
        amp = np.linalg.norm(u, axis=1)
        worst_amp[idx] = np.max(amp)

        # 3) Axis-wise max body torque with |u_i| <= 1
        # tau = -Gs u; for a single axis, max is sum of abs of row entries
        row_x = gs[0, :]
        row_y = gs[1, :]
        row_z = gs[2, :]
        max_tau_x[idx] = np.sum(np.abs(row_x))
        max_tau_y[idx] = np.sum(np.abs(row_y))
        max_tau_z[idx] = np.sum(np.abs(row_z))

    return {
        "alpha_deg": alphas_deg,
        "cond": cond_vals,
        "worst_amp": worst_amp,
        "max_tau_x": max_tau_x,
        "max_tau_y": max_tau_y,
        "max_tau_z": max_tau_z,
    }


def _style_axes(ax, xlabel: str, ylabel: str, title: str) -> None:
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontweight="bold")
    ax.grid(True, alpha=0.3)


def main() -> None:
    parser = argparse.ArgumentParser(description="Sweep reaction wheel cant angle.")
    parser.add_argument("--alpha-min", type=float, default=10.0, help="Min cant angle (deg)")
    parser.add_argument("--alpha-max", type=float, default=80.0, help="Max cant angle (deg)")
    parser.add_argument("--alpha-step", type=float, default=1.0, help="Cant angle step (deg)")
    parser.add_argument("--n-dirs", type=int, default=2000, help="Number of unit torque directions")
    parser.add_argument("--out-dir", type=str, default="analysis/plots", help="Output directory")
    args = parser.parse_args()

    results = sweep(args.alpha_min, args.alpha_max, args.alpha_step, args.n_dirs)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    alpha = results["alpha_deg"]

    # Plot 1: Condition number
    fig1, ax1 = plt.subplots(figsize=(7.2, 4.5))
    ax1.plot(alpha, results["cond"], color="black", linewidth=2)
    ax1.axvline(45.0, color="tab:blue", linestyle="--", linewidth=1)
    _style_axes(ax1, "Cant angle alpha (deg)", "cond(Gs)", "Allocation Conditioning vs Cant Angle")
    fig1.tight_layout()
    fig1.savefig(out_dir / "rw_cant_conditioning.png", dpi=200)

    # Plot 2: Worst-case torque amplification
    fig2, ax2 = plt.subplots(figsize=(7.2, 4.5))
    ax2.plot(alpha, results["worst_amp"], color="tab:red", linewidth=2)
    ax2.axvline(45.0, color="tab:blue", linestyle="--", linewidth=1)
    _style_axes(ax2, "Cant angle alpha (deg)", "max ||u|| (||tau||=1)", "Worst-Case Wheel Torque Amplification")
    fig2.tight_layout()
    fig2.savefig(out_dir / "rw_cant_worst_case_amplification.png", dpi=200)

    # Plot 3: Axis-wise max body torque (unit wheel torque limit)
    fig3, ax3 = plt.subplots(figsize=(7.2, 4.5))
    ax3.plot(alpha, results["max_tau_x"], label="Max |tau_x|", linewidth=2)
    ax3.plot(alpha, results["max_tau_y"], label="Max |tau_y|", linewidth=2)
    ax3.plot(alpha, results["max_tau_z"], label="Max |tau_z|", linewidth=2)
    ax3.axvline(45.0, color="tab:blue", linestyle="--", linewidth=1)
    _style_axes(ax3, "Cant angle alpha (deg)", "Max body torque (u_max=1)", "Axis Authority vs Cant Angle")
    ax3.legend()
    fig3.tight_layout()
    fig3.savefig(out_dir / "rw_cant_axis_authority.png", dpi=200)

    # Plot 4: Axis-wise authority normalized to max possible (u_max=1)
    # X and Y each use two wheels -> max possible is 2; Z uses one wheel -> max is 1.
    fig4, ax4 = plt.subplots(figsize=(7.2, 4.5))
    ax4.plot(alpha, results["max_tau_x"] / 2.0, label="normalised tau_x", linewidth=2)
    ax4.plot(alpha, results["max_tau_y"] / 2.0, label="normalised tau_y", linewidth=2)
    ax4.plot(alpha, results["max_tau_z"], label="normalised tau_z", linewidth=2)
    ax4.axvline(45.0, color="tab:blue", linestyle="--", linewidth=1)
    _style_axes(ax4, "Cant angle alpha (deg)", "Normalized axis authority", "Axis Authority vs Cant Angle (Normalized)")
    ax4.set_xlim(0.0, 90.0)
    ax4.legend()
    fig4.tight_layout()
    fig4.savefig(out_dir / "rw_cant_axis_authority_normalized.png", dpi=200)

    print(f"Saved plots to: {out_dir}")


if __name__ == "__main__":
    main()
