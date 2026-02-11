"""
Temporary motion profile comparison plotter.

Generates position, velocity, acceleration, jerk, and snap comparisons
for S-curve and fourth-order feedforward profiles using the project
configuration and trajectory loading path.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, Tuple

import matplotlib.pyplot as plt
import numpy as np
from scipy import signal


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
SRC_DIR = PROJECT_ROOT / "src"

if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import run_mission as ms  # noqa: E402
from basilisk_sim.design_shaper import design_s_curve_trajectory  # noqa: E402


def _compute_jerk_snap(time: np.ndarray, alpha: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Compute jerk and snap with finite differences."""
    if len(time) < 3 or len(alpha) < 3:
        zeros = np.zeros_like(alpha, dtype=float)
        return zeros, zeros
    jerk = np.gradient(alpha, time)
    snap = np.gradient(jerk, time)
    return jerk, snap


def _compute_psd(time: np.ndarray, values: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Compute one-sided PSD with a rectangular window in physical units.

    Uses scipy periodogram with:
    - detrend='constant'
    - window='boxcar'
    - scaling='density'
    This returns PSD in units of (signal_unit^2 / Hz).
    """
    if len(time) < 4 or len(values) < 4:
        return np.array([]), np.array([])
    dt = float(np.median(np.diff(time)))
    if not np.isfinite(dt) or dt <= 0.0:
        return np.array([]), np.array([])
    fs = 1.0 / dt
    x = np.array(values, dtype=float)
    freqs, psd = signal.periodogram(
        x,
        fs=fs,
        window="boxcar",
        detrend="constant",
        scaling="density",
        return_onesided=True,
    )
    return freqs, psd


def _build_profile(method: str) -> Dict[str, np.ndarray]:
    """Get profile arrays from mission feedforward design path."""
    config = ms.default_config()
    data = ms._compute_torque_profile(config, method=method, settling_time=0.0)
    time = np.array(data["time"], dtype=float)
    theta = np.array(data["theta"], dtype=float)
    omega = np.array(data["omega"], dtype=float)
    alpha = np.array(data["alpha"], dtype=float)

    # Align arrays defensively.
    time, aligned = ms._align_series(time, theta, omega, alpha)
    theta = aligned[0]
    omega = aligned[1]
    alpha = aligned[2]
    jerk, snap = _compute_jerk_snap(time, alpha)

    # For S-curve, use the analytical jerk/snap from the trajectory designer.
    if method == "s_curve":
        axis = ms._normalize_axis(config.rotation_axis)
        ff_inertia = ms._get_feedforward_inertia(config)
        I_axis = float(axis @ ff_inertia @ axis)
        t_sc, _, _, _, jerk_sc, snap_sc, _ = design_s_curve_trajectory(
            target_duration=config.slew_duration_s,
            theta_final=np.radians(config.slew_angle_deg),
            I_axis=I_axis,
            max_torque=config.rw_max_torque_nm,
            dt=ms.UNIFIED_SAMPLE_DT,
            settling_time=0.0,
        )
        t_sc = np.array(t_sc, dtype=float)
        jerk_sc = np.array(jerk_sc, dtype=float)
        snap_sc = np.array(snap_sc, dtype=float)
        t_aligned, aligned = ms._align_series(time, jerk, snap, t_sc, jerk_sc, snap_sc)
        time = t_aligned
        jerk = aligned[3]
        snap = aligned[4]
        theta = theta[: len(time)]
        omega = omega[: len(time)]
        alpha = alpha[: len(time)]

    return {
        "time": time,
        "theta": theta,
        "omega": omega,
        "alpha": alpha,
        "jerk": jerk,
        "snap": snap,
    }


def main() -> None:
    config = ms.default_config()
    fourth = _build_profile("fourth")
    s_curve = _build_profile("s_curve")

    colors = {"fourth": "#1f77b4", "s_curve": "#ff7f0e"}
    names = {"fourth": "Fourth-order", "s_curve": "S-curve"}

    fig, axes = plt.subplots(3, 2, figsize=(14, 11))
    axes = axes.flatten()

    series = [
        ("theta", "Position", "deg"),
        ("omega", "Velocity", "deg/s"),
        ("alpha", "Acceleration", "deg/s^2"),
        ("jerk", "Jerk", "deg/s^3"),
        ("snap", "Snap", "deg/s^4"),
    ]

    for i, (key, title, unit) in enumerate(series):
        ax = axes[i]
        t4 = fourth["time"]
        ts = s_curve["time"]
        y4 = np.degrees(fourth[key])
        ys = np.degrees(s_curve[key])

        ax.plot(t4, y4, color=colors["fourth"], linewidth=2.2, label=names["fourth"])
        ax.plot(ts, ys, color=colors["s_curve"], linewidth=2.2, label=names["s_curve"])
        ax.set_title(title)
        ax.set_ylabel(unit)
        ax.grid(True, alpha=0.3)
        if i >= 3:
            ax.set_xlabel("Time (s)")

    # Sixth subplot: acceleration PSD with mode markers.
    ax_psd = axes[5]
    f4, p4 = _compute_psd(fourth["time"], fourth["alpha"])
    fs, ps = _compute_psd(s_curve["time"], s_curve["alpha"])
    eps = 1e-18
    if len(f4) > 0:
        ax_psd.plot(f4, 10.0 * np.log10(p4 + eps), color=colors["fourth"], linewidth=2.0, label=names["fourth"])
    if len(fs) > 0:
        ax_psd.plot(fs, 10.0 * np.log10(ps + eps), color=colors["s_curve"], linewidth=2.0, label=names["s_curve"])

    mode_freqs = list(config.modal_freqs_hz)[:2]
    mode_colors = ["#d62728", "#9467bd"]
    for idx, freq_hz in enumerate(mode_freqs):
        ax_psd.axvline(
            x=float(freq_hz),
            color=mode_colors[idx],
            linestyle="--",
            linewidth=1.6,
            label=f"Mode {idx + 1}: {freq_hz:.1f} Hz",
        )
        if len(f4) > 0:
            i4 = int(np.argmin(np.abs(f4 - freq_hz)))
            y4_db = 10.0 * np.log10(p4[i4] + eps)
            ax_psd.plot([f4[i4]], [y4_db], marker="o", color=colors["fourth"], markersize=4)
        if len(fs) > 0:
            is_ = int(np.argmin(np.abs(fs - freq_hz)))
            ys_db = 10.0 * np.log10(ps[is_] + eps)
            ax_psd.plot([fs[is_]], [ys_db], marker="o", color=colors["s_curve"], markersize=4)
    ax_psd.set_xlim(0.0, 2.5)
    ax_psd.set_title("Acceleration PSD")
    ax_psd.set_xlabel("Frequency (Hz)")
    ax_psd.set_ylabel("PSD (dB)")
    ax_psd.grid(True, alpha=0.3)
    ax_psd.legend(loc="upper right", fontsize=8)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2, frameon=True, bbox_to_anchor=(0.5, 0.01))
    fig.suptitle("Motion Profile Comparison: Fourth-order vs S-curve", fontsize=15, y=0.98)
    fig.tight_layout(rect=[0, 0.06, 1, 0.95])

    out_dir = PROJECT_ROOT / "output" / "plots"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "motion_profile_comparison.png"
    fig.savefig(out_path, dpi=220)
    plt.close(fig)

    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
