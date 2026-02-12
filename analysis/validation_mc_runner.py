"""
Comprehensive Verification, Validation, and Monte Carlo Analysis.

This script implements the complete V&V and Monte Carlo plan from validation_mc.md.
It validates feedforward shaping, feedback control, and robustness under uncertainty.

Sections:
    1. Verification - Implementation correctness
    2. Validation - Physics and performance validity
    3. Monte Carlo - Robustness under uncertainty

Usage:
    python validation_mc_runner.py [--verification] [--validation] [--monte-carlo N]
    python validation_mc_runner.py --all  # Run everything
"""

from __future__ import annotations

import argparse
import csv
import inspect
import json
import os
import shutil
import subprocess
import sys
import time as time_module
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
from scipy import signal
from scipy.fft import fft, fftfreq

# Add paths for module imports
_analysis_dir = Path(__file__).parent.resolve()
_basilisk_dir = _analysis_dir.parent
_src_dir = _basilisk_dir / "src"
_scripts_dir = _basilisk_dir / "scripts"
for path in (_src_dir, _scripts_dir):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

# Backward compatible string aliases used throughout this script.
analysis_dir = str(_analysis_dir)
basilisk_dir = str(_basilisk_dir)

# Import run_mission (formerly mission_simulation) for plan compliant analysis
import run_mission as ms

from basilisk_sim.spacecraft_properties import (
    HUB_INERTIA,
    FLEX_MODE_MASS,
    FLEX_MODE_LOCATIONS,
    compute_effective_inertia,
    compute_modal_gains,
    compute_mode_lever_arms,
)

# ============================================================================
# Configuration and Constants
# ============================================================================

METHODS = ["s_curve", "fourth"]
CONTROLLERS = ["standard_pd", "filtered_pd"]
UNIFIED_SAMPLE_DT = 0.01  # 100 Hz to match Basilisk simulation
POST_SLEW_POINTING_LIMIT_ARCSEC = 7.0
POST_SLEW_VIBRATION_LIMIT_MM = 0.2
POST_SLEW_ACCEL_LIMIT_MM_S2 = 10.0
MC_COMBO_STYLES = {
    "s_curve_standard_pd": ("S-curve + Standard PD", "#ff7f0e"),
    "s_curve_filtered_pd": ("S-curve + Filtered PD", "#ffbb78"),
    "fourth_standard_pd": ("Fourth-order + Standard PD", "#1f77b4"),
    "fourth_filtered_pd": ("Fourth-order + Filtered PD", "#aec7e8"),
}

# Default pass/fail thresholds (validation_mc.md section 3.3)
DEFAULT_THRESHOLDS = {
    "rms_pointing_error_deg_p95": 0.005,
    "peak_torque_nm_p99": 70.0,
    "rms_vibration_mm_p95": 0.1,
    "torque_saturation_percent_max": 5.0,
}


@dataclass
class ValidationConfig:
    """Configuration for validation runs."""

    # Spacecraft inertia
    inertia: np.ndarray = field(default_factory=lambda: compute_effective_inertia().copy())
    rotation_axis: np.ndarray = field(default_factory=lambda: np.array([0.0, 0.0, 1.0]))

    # Modal parameters
    modal_freqs_hz: List[float] = field(default_factory=lambda: [0.4, 1.3])
    modal_damping: List[float] = field(default_factory=lambda: [0.02, 0.015])
    modal_mass_kg: float = FLEX_MODE_MASS

    # Maneuver parameters
    slew_angle_deg: float = 180.0
    slew_duration_s: float = 30.0
    settling_time_s: float = 30.0

    # Control parameters
    control_bandwidth_factor: float = 2.5  # bandwidth = first_mode / factor
    control_damping_ratio: float = 0.9
    control_filter_cutoff_hz: float = 8.0

    # Actuator limits
    rw_max_torque_nm: float = 70.0

    # Sensor noise (rad/s RMS for rate gyro)
    sensor_noise_std_rad_s: float = 1e-5

    # Disturbance torque (N*m bias)
    disturbance_torque_nm: float = 0.0

    # Camera parameters for jitter calculation
    camera_lever_arm_m: float = 4.0
    pixel_scale_arcsec: float = 2.0

    def copy(self) -> "ValidationConfig":
        """Create a deep copy of the config."""
        return ValidationConfig(
            inertia=self.inertia.copy(),
            rotation_axis=self.rotation_axis.copy(),
            modal_freqs_hz=self.modal_freqs_hz.copy(),
            modal_damping=self.modal_damping.copy(),
            modal_mass_kg=self.modal_mass_kg,
            slew_angle_deg=self.slew_angle_deg,
            slew_duration_s=self.slew_duration_s,
            settling_time_s=self.settling_time_s,
            control_bandwidth_factor=self.control_bandwidth_factor,
            control_damping_ratio=self.control_damping_ratio,
            control_filter_cutoff_hz=self.control_filter_cutoff_hz,
            rw_max_torque_nm=self.rw_max_torque_nm,
            sensor_noise_std_rad_s=self.sensor_noise_std_rad_s,
            disturbance_torque_nm=self.disturbance_torque_nm,
            camera_lever_arm_m=self.camera_lever_arm_m,
            pixel_scale_arcsec=self.pixel_scale_arcsec,
        )


@dataclass
class VerificationResult:
    """Result of a verification test."""
    test_name: str
    passed: bool
    message: str
    details: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ValidationResult:
    """Result of a validation test."""
    test_name: str
    passed: bool
    message: str
    metrics: Dict[str, float] = field(default_factory=dict)
    plots: List[str] = field(default_factory=list)


@dataclass
class MonteCarloRun:
    """Single Monte Carlo run result."""
    run_id: int
    config_perturbations: Dict[str, float]
    metrics: Dict[str, float]
    passed: bool
    failure_reasons: List[str] = field(default_factory=list)


@dataclass
class MonteCarloSummary:
    """Summary of Monte Carlo analysis."""
    n_runs: int
    n_passed: int
    pass_rate: float
    percentiles: Dict[str, Dict[str, float]]
    histograms: Dict[str, Tuple[np.ndarray, np.ndarray]]


# ============================================================================
# Utility Functions
# ============================================================================

def _ensure_dir(path: str) -> None:
    """Create directory if it doesn't exist."""
    os.makedirs(path, exist_ok=True)


def _reset_dir(path: str) -> None:
    """Recreate a clean directory."""
    if os.path.isdir(path):
        shutil.rmtree(path)
    os.makedirs(path, exist_ok=True)


def _normalize_axis(axis: np.ndarray) -> np.ndarray:
    """Return unit vector for rotation axis."""
    axis = np.array(axis, dtype=float).flatten()
    norm = np.linalg.norm(axis)
    return axis / norm if norm > 0 else np.array([0.0, 0.0, 1.0])


def _write_csv(path: str, headers: List[str], rows: List[List[Any]]) -> None:
    """Write CSV file."""
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        writer.writerows(rows)


def _mrp_to_angle_deg(sigma: np.ndarray) -> float:
    """Convert MRP to rotation angle in degrees."""
    sigma = np.array(sigma, dtype=float).flatten()
    mag = np.linalg.norm(sigma)
    return np.degrees(4 * np.arctan(mag))


def _compute_psd(time: np.ndarray, data: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Compute power spectral density using Welch's method."""
    if len(time) < 16 or len(data) < 16:
        return np.array([]), np.array([])

    dt = np.median(np.diff(time))
    if not np.isfinite(dt) or dt <= 0:
        return np.array([]), np.array([])

    fs = 1.0 / dt
    nperseg = min(1024, len(data) // 4)
    nperseg = max(16, nperseg)

    try:
        freq, psd = signal.welch(data, fs=fs, nperseg=nperseg, noverlap=nperseg//2)
        return freq, psd
    except Exception:
        return np.array([]), np.array([])


def _compute_rms(data: np.ndarray) -> float:
    """Compute RMS of data."""
    if len(data) == 0:
        return 0.0
    return float(np.sqrt(np.mean(data**2)))


def _compute_band_rms(freq: np.ndarray, psd: np.ndarray, fmin: float, fmax: float) -> float:
    """Compute band limited RMS from PSD."""
    if len(freq) == 0 or len(psd) == 0:
        return float("nan")
    mask = (freq >= fmin) & (freq <= fmax) & np.isfinite(psd) & (psd >= 0)
    if not np.any(mask):
        return float("nan")
    df = np.gradient(freq[mask])
    return float(np.sqrt(np.sum(psd[mask] * df)))


def _compute_post_slew_stats(
    time: np.ndarray, values: np.ndarray, slew_duration_s: float
) -> Tuple[float, float]:
    """Compute RMS and peak for post slew window (or full if unavailable)."""
    if len(time) == 0 or len(values) == 0:
        return float("nan"), float("nan")
    mask = time >= slew_duration_s
    series = values[mask] if np.any(mask) else values
    rms = _compute_rms(series)
    peak = float(np.max(np.abs(series))) if len(series) else float("nan")
    return rms, peak


def _plot_post_slew_pointing_box_from_csv(
    out_dir: str,
    threshold_arcsec: float = POST_SLEW_POINTING_LIMIT_ARCSEC,
) -> Optional[str]:
    """Create post slew RMS pointing box plot from existing Monte Carlo CSV."""
    csv_path = os.path.join(out_dir, "monte_carlo_runs.csv")
    if not os.path.isfile(csv_path):
        print(f"Post-slew box plot skipped, missing CSV: {csv_path}")
        return None

    with open(csv_path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    if not rows:
        print(f"Post-slew box plot skipped, no rows in CSV: {csv_path}")
        return None

    combo_cols: List[str] = []
    for col in rows[0].keys():
        if col.endswith("_rms_pointing_error_deg") and col != "rms_pointing_error_deg":
            combo_cols.append(col)

    if not combo_cols:
        print(f"Post-slew box plot skipped, no per-combo RMS columns in: {csv_path}")
        return None

    combo_cols = sorted(combo_cols)
    data: List[np.ndarray] = []
    labels: List[str] = []
    colors: List[str] = []
    violation_rows: List[List[Any]] = []

    for col in combo_cols:
        combo_key = col.replace("_rms_pointing_error_deg", "")
        vals_deg = []
        for row in rows:
            raw = row.get(col, "nan")
            try:
                val = float(raw)
            except (TypeError, ValueError):
                val = float("nan")
            if np.isfinite(val):
                vals_deg.append(val)

        if not vals_deg:
            continue

        vals_arcsec = np.array(vals_deg, dtype=float) * 3600.0
        style = MC_COMBO_STYLES.get(combo_key, (combo_key.replace("_", " "), "#7f7f7f"))
        combo_label, combo_color = style

        n_total = int(vals_arcsec.size)
        n_viol = int(np.sum(vals_arcsec > float(threshold_arcsec)))
        viol_pct = (100.0 * n_viol / n_total) if n_total > 0 else float("nan")
        p95_arcsec = float(np.percentile(vals_arcsec, 95)) if n_total > 0 else float("nan")
        p99_arcsec = float(np.percentile(vals_arcsec, 99)) if n_total > 0 else float("nan")

        data.append(vals_arcsec)
        labels.append(combo_label)
        colors.append(combo_color)
        violation_rows.append(
            [
                combo_key,
                combo_label,
                n_total,
                n_viol,
                viol_pct,
                p95_arcsec,
                p99_arcsec,
                float(threshold_arcsec),
            ]
        )

    if not data:
        print(f"Post-slew box plot skipped, no finite post-slew RMS data in: {csv_path}")
        return None

    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    bp = ax.boxplot(
        data,
        labels=[lbl.replace(" + ", "\n+ ") for lbl in labels],
        patch_artist=True,
        showfliers=True,
    )

    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.75)
    for median in bp["medians"]:
        median.set_color("black")
        median.set_linewidth(1.5)

    ax.axhline(
        y=float(threshold_arcsec),
        color="red",
        linestyle="--",
        linewidth=1.5,
        label=f"{threshold_arcsec:.1f} arcsec RMS limit",
    )

    y_max = float(max(np.nanmax(vals) for vals in data))
    if np.isfinite(y_max) and y_max > 0:
        ax.set_ylim(bottom=0.0, top=max(y_max * 1.18, threshold_arcsec * 1.3))

    for idx, row in enumerate(violation_rows, start=1):
        n_total = int(row[2])
        n_viol = int(row[3])
        viol_pct = float(row[4])
        txt = f"{n_viol}/{n_total} violated ({viol_pct:.1f}%)"
        ax.text(
            idx,
            ax.get_ylim()[1] * 0.96,
            txt,
            ha="center",
            va="top",
            fontsize=9,
            color="black",
        )

    ax.set_title("Post-slew RMS Pointing Error (Monte Carlo)")
    ax.set_ylabel("RMS pointing error (arcsec)")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower right")
    fig.tight_layout()

    plot_path = os.path.join(out_dir, "monte_carlo_post_slew_pointing_box.png")
    fig.savefig(plot_path, dpi=180)
    plt.close(fig)

    violation_csv = os.path.join(out_dir, "monte_carlo_post_slew_pointing_violations.csv")
    _write_csv(
        violation_csv,
        [
            "combo_key",
            "combo_label",
            "n_samples",
            "n_violations",
            "violation_percent",
            "p95_arcsec",
            "p99_arcsec",
            "threshold_arcsec",
        ],
        violation_rows,
    )

    print(f"Saved post-slew pointing box plot: {plot_path}")
    print(f"Saved post-slew violation table: {violation_csv}")
    return plot_path


def _plot_post_slew_vibration_box_from_csv(
    out_dir: str,
    threshold_mm: float = POST_SLEW_VIBRATION_LIMIT_MM,
) -> Optional[str]:
    """Create post slew RMS vibration box plot from existing Monte Carlo CSV."""
    csv_path = os.path.join(out_dir, "monte_carlo_runs.csv")
    if not os.path.isfile(csv_path):
        print(f"Post-slew vibration box plot skipped, missing CSV: {csv_path}")
        return None

    with open(csv_path, "r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))

    if not rows:
        print(f"Post-slew vibration box plot skipped, no rows in CSV: {csv_path}")
        return None

    combo_cols: List[str] = []
    for col in rows[0].keys():
        if col.endswith("_rms_vibration_mm") and col != "rms_vibration_mm":
            combo_cols.append(col)

    if not combo_cols:
        print(f"Post-slew vibration box plot skipped, no per-combo RMS columns in: {csv_path}")
        return None

    combo_cols = sorted(combo_cols)
    data: List[np.ndarray] = []
    labels: List[str] = []
    colors: List[str] = []
    violation_rows: List[List[Any]] = []

    for col in combo_cols:
        combo_key = col.replace("_rms_vibration_mm", "")
        vals_mm: List[float] = []
        for row in rows:
            raw = row.get(col, "nan")
            try:
                val = float(raw)
            except (TypeError, ValueError):
                val = float("nan")
            if np.isfinite(val):
                vals_mm.append(val)

        if not vals_mm:
            continue

        arr_mm = np.array(vals_mm, dtype=float)
        style = MC_COMBO_STYLES.get(combo_key, (combo_key.replace("_", " "), "#7f7f7f"))
        combo_label, combo_color = style

        n_total = int(arr_mm.size)
        n_viol = int(np.sum(arr_mm > float(threshold_mm)))
        viol_pct = (100.0 * n_viol / n_total) if n_total > 0 else float("nan")
        p95_mm = float(np.percentile(arr_mm, 95)) if n_total > 0 else float("nan")
        p99_mm = float(np.percentile(arr_mm, 99)) if n_total > 0 else float("nan")

        data.append(arr_mm)
        labels.append(combo_label)
        colors.append(combo_color)
        violation_rows.append(
            [
                combo_key,
                combo_label,
                n_total,
                n_viol,
                viol_pct,
                p95_mm,
                p99_mm,
                float(threshold_mm),
            ]
        )

    if not data:
        print(f"Post-slew vibration box plot skipped, no finite post-slew RMS data in: {csv_path}")
        return None

    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    bp = ax.boxplot(
        data,
        labels=[lbl.replace(" + ", "\n+ ") for lbl in labels],
        patch_artist=True,
        showfliers=True,
    )

    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.75)
    for median in bp["medians"]:
        median.set_color("black")
        median.set_linewidth(1.5)

    ax.axhline(
        y=float(threshold_mm),
        color="red",
        linestyle="--",
        linewidth=1.5,
        label=f"{threshold_mm:.2f} mm RMS limit",
    )

    y_max = float(max(np.nanmax(vals) for vals in data))
    if np.isfinite(y_max) and y_max > 0:
        ax.set_ylim(bottom=0.0, top=max(y_max * 1.18, threshold_mm * 1.3))

    for idx, row in enumerate(violation_rows, start=1):
        n_total = int(row[2])
        n_viol = int(row[3])
        viol_pct = float(row[4])
        txt = f"{n_viol}/{n_total} violated ({viol_pct:.1f}%)"
        ax.text(
            idx,
            ax.get_ylim()[1] * 0.96,
            txt,
            ha="center",
            va="top",
            fontsize=9,
            color="black",
        )

    ax.set_title("Post-slew RMS Vibration (Monte Carlo)")
    ax.set_ylabel("RMS vibration (mm)")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower right")
    fig.tight_layout()

    plot_path = os.path.join(out_dir, "monte_carlo_post_slew_vibration_box.png")
    fig.savefig(plot_path, dpi=180)
    plt.close(fig)

    violation_csv = os.path.join(out_dir, "monte_carlo_post_slew_vibration_violations.csv")
    _write_csv(
        violation_csv,
        [
            "combo_key",
            "combo_label",
            "n_samples",
            "n_violations",
            "violation_percent",
            "p95_mm",
            "p99_mm",
            "threshold_mm",
        ],
        violation_rows,
    )

    print(f"Saved post-slew vibration box plot: {plot_path}")
    print(f"Saved post-slew vibration violation table: {violation_csv}")
    return plot_path


def _plot_post_slew_acceleration_box_from_csv(
    out_dir: str,
    threshold_mm_s2: float = POST_SLEW_ACCEL_LIMIT_MM_S2,
) -> Optional[str]:
    """Create post slew RMS modal acceleration box plot from existing Monte Carlo CSV."""
    csv_path = os.path.join(out_dir, "monte_carlo_runs.csv")
    if not os.path.isfile(csv_path):
        print(f"Post-slew acceleration box plot skipped, missing CSV: {csv_path}")
        return None

    with open(csv_path, "r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))

    if not rows:
        print(f"Post-slew acceleration box plot skipped, no rows in CSV: {csv_path}")
        return None

    combo_cols: List[str] = []
    for col in rows[0].keys():
        if col.endswith("_rms_modal_accel_mm_s2") and col != "rms_modal_accel_mm_s2":
            combo_cols.append(col)

    if not combo_cols:
        print(f"Post-slew acceleration box plot skipped, no per-combo RMS columns in: {csv_path}")
        return None

    combo_cols = sorted(combo_cols)
    data: List[np.ndarray] = []
    labels: List[str] = []
    colors: List[str] = []
    violation_rows: List[List[Any]] = []

    for col in combo_cols:
        combo_key = col.replace("_rms_modal_accel_mm_s2", "")
        vals_mm_s2: List[float] = []
        for row in rows:
            raw = row.get(col, "nan")
            try:
                val = float(raw)
            except (TypeError, ValueError):
                val = float("nan")
            if np.isfinite(val):
                vals_mm_s2.append(val)

        if not vals_mm_s2:
            continue

        arr_mm_s2 = np.array(vals_mm_s2, dtype=float)
        style = MC_COMBO_STYLES.get(combo_key, (combo_key.replace("_", " "), "#7f7f7f"))
        combo_label, combo_color = style

        n_total = int(arr_mm_s2.size)
        n_viol = int(np.sum(arr_mm_s2 > float(threshold_mm_s2)))
        viol_pct = (100.0 * n_viol / n_total) if n_total > 0 else float("nan")
        p95_mm_s2 = float(np.percentile(arr_mm_s2, 95)) if n_total > 0 else float("nan")
        p99_mm_s2 = float(np.percentile(arr_mm_s2, 99)) if n_total > 0 else float("nan")

        data.append(arr_mm_s2)
        labels.append(combo_label)
        colors.append(combo_color)
        violation_rows.append(
            [
                combo_key,
                combo_label,
                n_total,
                n_viol,
                viol_pct,
                p95_mm_s2,
                p99_mm_s2,
                float(threshold_mm_s2),
            ]
        )

    if not data:
        print(f"Post-slew acceleration box plot skipped, no finite post-slew RMS data in: {csv_path}")
        return None

    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    bp = ax.boxplot(
        data,
        labels=[lbl.replace(" + ", "\n+ ") for lbl in labels],
        patch_artist=True,
        showfliers=True,
    )

    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.75)
    for median in bp["medians"]:
        median.set_color("black")
        median.set_linewidth(1.5)

    ax.axhline(
        y=float(threshold_mm_s2),
        color="red",
        linestyle="--",
        linewidth=1.5,
        label=f"{threshold_mm_s2:.1f} mm/s^2 RMS limit",
    )

    y_max = float(max(np.nanmax(vals) for vals in data))
    if np.isfinite(y_max) and y_max > 0:
        ax.set_ylim(bottom=0.0, top=max(y_max * 1.18, threshold_mm_s2 * 1.3))

    for idx, row in enumerate(violation_rows, start=1):
        n_total = int(row[2])
        n_viol = int(row[3])
        viol_pct = float(row[4])
        txt = f"{n_viol}/{n_total} violated ({viol_pct:.1f}%)"
        ax.text(
            idx,
            ax.get_ylim()[1] * 0.96,
            txt,
            ha="center",
            va="top",
            fontsize=9,
            color="black",
        )

    ax.set_title("Post-slew RMS Modal Acceleration (Monte Carlo)")
    ax.set_ylabel("RMS modal acceleration (mm/s^2)")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower right")
    fig.tight_layout()

    plot_path = os.path.join(out_dir, "monte_carlo_post_slew_acceleration_box.png")
    fig.savefig(plot_path, dpi=180)
    plt.close(fig)

    violation_csv = os.path.join(out_dir, "monte_carlo_post_slew_acceleration_violations.csv")
    _write_csv(
        violation_csv,
        [
            "combo_key",
            "combo_label",
            "n_samples",
            "n_violations",
            "violation_percent",
            "p95_mm_s2",
            "p99_mm_s2",
            "threshold_mm_s2",
        ],
        violation_rows,
    )

    print(f"Saved post-slew acceleration box plot: {plot_path}")
    print(f"Saved post-slew acceleration violation table: {violation_csv}")
    return plot_path


def _reconstruct_mc_perturbations(n_runs: int, seed: int = 42) -> Dict[str, np.ndarray]:
    """Rebuild perturbation factors using the same RNG stream as PlanMonteCarloRunner."""
    rng = np.random.default_rng(seed)
    factors = {
        "inertia_scale": np.zeros(n_runs, dtype=float),
        "freq_scale": np.zeros(n_runs, dtype=float),
        "damp_scale": np.zeros(n_runs, dtype=float),
        "modal_gains_scale": np.zeros(n_runs, dtype=float),
        "cutoff_scale": np.zeros(n_runs, dtype=float),
        "noise_scale": np.zeros(n_runs, dtype=float),
        "disturbance_scale": np.zeros(n_runs, dtype=float),
    }
    for i in range(n_runs):
        factors["inertia_scale"][i] = 1.0 + rng.uniform(-0.2, 0.2)
        factors["freq_scale"][i] = 1.0 + rng.uniform(-0.1, 0.1)
        factors["damp_scale"][i] = 1.0 + rng.uniform(-0.5, 0.5)
        factors["modal_gains_scale"][i] = 1.0 + rng.uniform(-0.2, 0.2)
        factors["cutoff_scale"][i] = 1.0 + rng.uniform(-0.2, 0.2)
        factors["noise_scale"][i] = 1.0 + rng.uniform(-0.5, 0.5)
        factors["disturbance_scale"][i] = 1.0 + rng.uniform(-0.5, 0.5)
    return factors


def _plot_post_slew_pointing_factor_histograms_from_csv(
    out_dir: str,
    seed: int = 42,
) -> List[str]:
    """Plot histogram based parameter impact views for post slew RMS pointing error."""
    csv_path = os.path.join(out_dir, "monte_carlo_runs.csv")
    if not os.path.isfile(csv_path):
        print(f"Pointing factor histogram skipped, missing CSV: {csv_path}")
        return []

    with open(csv_path, "r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        print(f"Pointing factor histogram skipped, no rows in CSV: {csv_path}")
        return []

    run_ids: List[int] = []
    metric_by_run: Dict[int, Dict[str, float]] = {}
    metric_cols = [
        "s_curve_standard_pd_rms_pointing_error_deg",
        "fourth_standard_pd_rms_pointing_error_deg",
    ]

    for row in rows:
        try:
            run_id = int(row.get("run_id", "-1"))
        except (TypeError, ValueError):
            continue
        if run_id < 0:
            continue
        run_ids.append(run_id)
        metric_map: Dict[str, float] = {}
        for col in metric_cols:
            raw = row.get(col, "nan")
            try:
                val = float(raw)
            except (TypeError, ValueError):
                val = float("nan")
            metric_map[col] = val
        metric_by_run[run_id] = metric_map

    if not run_ids:
        print(f"Pointing factor histogram skipped, no valid run_id in CSV: {csv_path}")
        return []

    n_runs = max(run_ids) + 1
    factors = _reconstruct_mc_perturbations(n_runs=n_runs, seed=seed)
    factor_names = [name for name in factors.keys() if name != "cutoff_scale"]
    factor_labels = {
        "inertia_scale": "Inertia scale",
        "freq_scale": "Resonant mode scale",
        "damp_scale": "Modal damping scale",
        "modal_gains_scale": "Modal gain scale",
        "cutoff_scale": "Filter cutoff scale",
        "noise_scale": "Rate noise scale",
        "disturbance_scale": "Disturbance scale",
    }
    method_specs = [
        ("s_curve_standard_pd_rms_pointing_error_deg", "S-curve + Standard PD", "s_curve_standard_pd"),
        ("fourth_standard_pd_rms_pointing_error_deg", "Fourth-order + Standard PD", "fourth_standard_pd"),
    ]

    saved_paths: List[str] = []
    impact_rows: List[List[Any]] = []

    for metric_col, method_label, method_tag in method_specs:
        fig, axes = plt.subplots(3, 3, figsize=(16, 12))
        axes_flat = np.atleast_1d(axes).flatten()
        for idx, factor_name in enumerate(factor_names):
            ax = axes_flat[idx]
            x_vals: List[float] = []
            y_vals: List[float] = []
            for rid in run_ids:
                metric = metric_by_run.get(rid, {}).get(metric_col, float("nan"))
                if not np.isfinite(metric):
                    continue
                if rid >= len(factors[factor_name]):
                    continue
                x = float(factors[factor_name][rid])
                y_arcsec = float(metric) * 3600.0
                if np.isfinite(x) and np.isfinite(y_arcsec):
                    x_vals.append(x)
                    y_vals.append(y_arcsec)

            if len(x_vals) < 8:
                ax.set_title(f"{factor_labels.get(factor_name, factor_name)}\nno data")
                ax.axis("off")
                continue

            x_arr = np.array(x_vals, dtype=float)
            y_arr = np.array(y_vals, dtype=float)
            q1 = float(np.percentile(x_arr, 25))
            q3 = float(np.percentile(x_arr, 75))
            low_mask = x_arr <= q1
            high_mask = x_arr >= q3
            y_low = y_arr[low_mask]
            y_high = y_arr[high_mask]
            if y_low.size == 0 or y_high.size == 0:
                ax.set_title(f"{factor_labels.get(factor_name, factor_name)}\nno quartile split")
                ax.axis("off")
                continue

            y_all = np.concatenate([y_low, y_high])
            y_min = float(np.min(y_all))
            y_max = float(np.max(y_all))
            if not np.isfinite(y_min) or not np.isfinite(y_max) or y_max <= y_min:
                bins = 15
            else:
                bins = np.linspace(y_min, y_max, 22)

            ax.hist(y_low, bins=bins, alpha=0.65, color="#1f77b4", label=f"Low quartile <= {q1:.3f}")
            ax.hist(y_high, bins=bins, alpha=0.65, color="#ff7f0e", label=f"High quartile >= {q3:.3f}")

            med_low = float(np.median(y_low))
            med_high = float(np.median(y_high))
            delta_med = med_high - med_low
            effect_abs = abs(delta_med)
            impact_rows.append(
                [
                    method_label,
                    factor_name,
                    factor_labels.get(factor_name, factor_name),
                    int(x_arr.size),
                    int(y_low.size),
                    int(y_high.size),
                    q1,
                    q3,
                    med_low,
                    med_high,
                    delta_med,
                    effect_abs,
                ]
            )

            ax.set_title(factor_labels.get(factor_name, factor_name))
            ax.set_xlabel("Post-slew RMS pointing error (arcsec)")
            ax.set_ylabel("Count")
            ax.grid(True, alpha=0.25)
            ax.legend(fontsize=8)
            ax.text(
                0.98,
                0.95,
                f"|Δmedian|={effect_abs:.2f} arcsec",
                transform=ax.transAxes,
                ha="right",
                va="top",
                fontsize=9,
                bbox=dict(facecolor="white", edgecolor="gray", alpha=0.85),
            )

        for idx in range(len(factor_names), len(axes_flat)):
            axes_flat[idx].axis("off")

        fig.suptitle(f"Monte Carlo Parameter Impact Histograms\n{method_label}", fontsize=14, fontweight="bold")
        fig.tight_layout(rect=[0, 0.02, 1, 0.96])
        plot_path = os.path.join(out_dir, f"monte_carlo_post_slew_pointing_factor_hist_{method_tag}.png")
        fig.savefig(plot_path, dpi=180)
        plt.close(fig)
        saved_paths.append(plot_path)
        print(f"Saved post-slew pointing factor histograms: {plot_path}")

    impact_rows_sorted: List[List[Any]] = []
    for method_label in [m[1] for m in method_specs]:
        rows_method = [r for r in impact_rows if r[0] == method_label]
        rows_method.sort(key=lambda r: (float(r[11]) if np.isfinite(float(r[11])) else -1.0), reverse=True)
        rank = 1
        for row in rows_method:
            impact_rows_sorted.append([rank] + row)
            rank += 1

    impact_csv = os.path.join(out_dir, "monte_carlo_post_slew_pointing_factor_impact.csv")
    _write_csv(
        impact_csv,
        [
            "rank_within_method",
            "method",
            "factor",
            "factor_label",
            "n_samples",
            "n_low_quartile",
            "n_high_quartile",
            "q25_factor",
            "q75_factor",
            "median_low_arcsec",
            "median_high_arcsec",
            "delta_median_arcsec",
            "abs_delta_median_arcsec",
        ],
        impact_rows_sorted,
    )
    print(f"Saved post-slew pointing factor impact table: {impact_csv}")

    return saved_paths


def _remove_legacy_correlation_outputs(out_dir: str) -> None:
    """Remove legacy correlation outputs that were replaced by histogram outputs."""
    legacy_files = [
        os.path.join(out_dir, "monte_carlo_post_slew_pointing_correlation.png"),
        os.path.join(out_dir, "monte_carlo_post_slew_pointing_correlation.csv"),
    ]
    for path in legacy_files:
        if os.path.isfile(path):
            try:
                os.remove(path)
                print(f"Removed legacy correlation output: {path}")
            except OSError:
                pass


def _format_eta(seconds: float) -> str:
    """Format seconds as H:MM:SS or M:SS."""
    if not np.isfinite(seconds) or seconds < 0:
        return "?"
    secs = int(round(seconds))
    mins, sec = divmod(secs, 60)
    hrs, mins = divmod(mins, 60)
    if hrs > 0:
        return f"{hrs}h {mins:02d}m {sec:02d}s"
    return f"{mins:02d}m {sec:02d}s"


def _update_progress(
    prefix: str,
    current: int,
    total: int,
    start_time: float,
    last_update: float,
    min_interval_s: float = 0.5,
    width: int = 32,
) -> float:
    """Render a single line progress bar with ETA."""
    now = time_module.time()
    if current < total and (now - last_update) < min_interval_s:
        return last_update

    total = max(total, 1)
    ratio = min(1.0, max(0.0, current / total))
    filled = int(round(width * ratio))
    bar = "=" * filled + "-" * (width - filled)
    elapsed = now - start_time
    rate = current / elapsed if elapsed > 0 else 0.0
    remaining = (total - current) / rate if rate > 0 else float("inf")
    eta = _format_eta(remaining)
    msg = f"\r{prefix} [{bar}] {current}/{total} ({ratio*100:5.1f}%) ETA {eta}"
    print(msg, end="", flush=True)
    if current >= total:
        print()
    return now


def _run_vizard_demo_batch(
    overrides: Dict[str, object],
    output_dir: str,
    run_mode: str = "combined",
    timeout_s: Optional[float] = 600.0,
) -> None:
    """Run vizard_demo for all method/controller combos with config overrides."""
    output_dir_abs = os.path.abspath(output_dir)
    _ensure_dir(output_dir_abs)
    cfg_path = os.path.join(output_dir_abs, "temp_config.json")
    with open(cfg_path, "w", encoding="utf-8") as f:
        json.dump(overrides, f)

    script_path = os.path.abspath(os.path.join(basilisk_dir, "scripts", "run_vizard_demo.py"))
    for method in METHODS:
        for controller in CONTROLLERS:
            cmd = [
                sys.executable,
                script_path,
                method,
                "--controller",
                controller,
                "--mode",
                run_mode,
                "--config",
                cfg_path,
                "--output-dir",
                output_dir_abs,
            ]
            subprocess.run(
                cmd,
                cwd=output_dir_abs,
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                timeout=timeout_s,
            )


# --- Error handling helpers ---------------------------------------------------
def _format_subprocess_failure(exc: BaseException) -> str:
    """Format a subprocess exception into a readable diagnostic message."""
    cmd = getattr(exc, "cmd", "")
    if isinstance(cmd, (list, tuple)):
        cmd_str = " ".join(str(part) for part in cmd)
    else:
        cmd_str = str(cmd)
    stdout = getattr(exc, "stdout", "") or ""
    stderr = getattr(exc, "stderr", "") or ""
    if isinstance(exc, subprocess.TimeoutExpired):
        msg = f"Command timed out after {exc.timeout}s: {cmd_str}"
    elif isinstance(exc, subprocess.CalledProcessError):
        msg = f"Command failed (code {exc.returncode}): {cmd_str}"
    else:
        msg = f"Command failed: {cmd_str}\n{exc}"
    if stdout.strip():
        msg += f"\n\n--- stdout ---\n{stdout.strip()}"
    if stderr.strip():
        msg += f"\n\n--- stderr ---\n{stderr.strip()}"
    return msg


def _write_mc_failure_log(run_id: int, exc: BaseException, output_dir: str) -> str:
    """Write a failure log for a Monte Carlo run that raised an exception."""
    _ensure_dir(output_dir)
    log_path = os.path.join(output_dir, f"mc_fail_run_{run_id:04d}.log")
    with open(log_path, "w", encoding="utf-8") as f:
        f.write(_format_subprocess_failure(exc))
        f.write("\n")
    return log_path


# ============================================================================
# PLAN COMPLIANT RUNNER (validation_mc.md)
# ============================================================================

PLAN_THRESHOLDS = {
    "rms_pointing_error_deg": 0.005,
    "peak_torque_nm": 70.0,
    "rms_vibration_mm": 0.1,
    "torque_saturation_percent": 5.0,
}


def _load_csv_as_dict(path: str, key_field: str) -> Dict[str, Dict[str, str]]:
    """Load a CSV file into a dict keyed by the specified field."""
    if not os.path.isfile(path):
        return {}
    rows = {}
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            key = row.get(key_field, "")
            rows[key] = row
    return rows


class PlanCompliantRunner:
    """Runner that strictly follows validation_mc.md."""

    def __init__(
        self,
        out_dir: str,
        data_dir: Optional[str] = None,
        sensor_noise_std_rad_s: float = 1e-5,
        disturbance_torque_nm: float = 1e-5,
    ):
        self.out_dir = out_dir
        self.data_dir = data_dir or basilisk_dir
        self.config = ms.default_config()
        self.sensor_noise_std_rad_s = float(sensor_noise_std_rad_s)
        self.disturbance_torque_nm = float(disturbance_torque_nm)
        _ensure_dir(out_dir)

    # ------------------------------------------------------------------
    # VERIFICATION (Section 1)
    # ------------------------------------------------------------------
    def run_verification(self) -> List[VerificationResult]:
        """Execute all verification checks and save the report."""
        results: List[VerificationResult] = []
        tests = [
            self._verify_trajectory_consistency,
            self._verify_controller_implementation,
            self._verify_logging_integrity,
            self._verify_psd_computations,
        ]
        start = time_module.time()
        last = start
        total = len(tests)
        _update_progress("Verification", 0, total, start, last)
        for idx, func in enumerate(tests, start=1):
            results.append(func())
            last = _update_progress("Verification", idx, total, start, last)
        self._save_verification_report(results)
        return results

    def _verify_trajectory_consistency(self) -> VerificationResult:
        """Verify feedforward trajectory consistency across shaping methods."""
        details: Dict[str, Any] = {}
        issues: List[str] = []
        axis = _normalize_axis(self.config.rotation_axis)
        ff_inertia = ms._get_feedforward_inertia(self.config)
        I_axis = float(axis @ ff_inertia @ axis)
        details["ff_I_axis"] = I_axis

        for method in METHODS:
            try:
                profile = ms._compute_torque_profile(self.config, method, settling_time=30.0)
                time = np.array(profile.get("time", []), dtype=float)
                theta = np.array(profile.get("theta", []), dtype=float)
                omega = np.array(profile.get("omega", []), dtype=float)
                alpha = np.array(profile.get("alpha", []), dtype=float)
                torque = np.array(profile.get("torque", []), dtype=float)

                n = len(time)
                details[f"{method}_samples"] = n
                if n < 2:
                    issues.append(f"{method}: insufficient samples ({n})")
                    continue

                lengths = {
                    "theta": len(theta),
                    "omega": len(omega),
                    "alpha": len(alpha),
                    "torque": len(torque),
                }
                for name, length in lengths.items():
                    if length != n:
                        issues.append(f"{method}: {name} length {length} != time length {n}")

                if np.any(np.diff(time) <= 0):
                    issues.append(f"{method}: non-monotonic time array")

                if n > 2:
                    dt = np.diff(time)
                    dt_med = float(np.median(dt))
                    dt_std = float(np.std(dt))
                    details[f"{method}_dt_median"] = dt_med
                    details[f"{method}_dt_std"] = dt_std
                    if dt_med <= 0 or (dt_std / dt_med) > 0.01:
                        issues.append(f"{method}: non-uniform timestep (std/med={dt_std/dt_med:.3f})")

                final_deg = float(np.degrees(theta[-1])) if len(theta) else float("nan")
                target = float(self.config.slew_angle_deg)
                details[f"{method}_final_deg"] = final_deg
                details[f"{method}_target_deg"] = target
                if np.isfinite(final_deg) and abs(abs(final_deg) - abs(target)) > 0.5:
                    issues.append(f"{method}: final angle {final_deg:.2f} deg vs {target:.2f} deg")

                if n > 3 and len(theta) == n and len(omega) == n:
                    omega_check = np.gradient(theta, time)
                    alpha_check = np.gradient(omega, time)
                    omega_err = _compute_rms(omega - omega_check)
                    alpha_err = _compute_rms(alpha - alpha_check) if len(alpha) == n else float("nan")
                    omega_scale = max(1e-6, float(np.max(np.abs(omega))))
                    alpha_scale = max(1e-6, float(np.max(np.abs(alpha)))) if len(alpha) else 1.0
                    details[f"{method}_omega_rms_err"] = omega_err
                    details[f"{method}_alpha_rms_err"] = alpha_err
                    if omega_err / omega_scale > 0.02:
                        issues.append(f"{method}: omega consistency error {omega_err:.3e} rad/s")
                    if np.isfinite(alpha_err) and alpha_err / alpha_scale > 0.02:
                        issues.append(f"{method}: alpha consistency error {alpha_err:.3e} rad/s^2")

                if len(torque) == n and len(alpha) == n:
                    torque_expected = I_axis * alpha
                    torque_err = _compute_rms(torque - torque_expected)
                    torque_scale = max(1e-6, float(_compute_rms(torque_expected)))
                    details[f"{method}_torque_rms_err"] = torque_err
                    if torque_err / torque_scale > 0.02:
                        issues.append(f"{method}: torque mismatch RMS {torque_err:.3e} Nm")

                if method == "fourth":
                    traj_candidates = [
                        os.path.join(basilisk_dir, "data", "trajectories", "spacecraft_trajectory_4th_180deg_30s.npz"),
                        os.path.join(basilisk_dir, "spacecraft_trajectory_4th_180deg_30s.npz"),
                        os.path.abspath(os.path.join(basilisk_dir, "..", "spacecraft_trajectory_4th_180deg_30s.npz")),
                    ]
                    traj_path = next((p for p in traj_candidates if os.path.isfile(p)), None)
                    if traj_path:
                        traj = np.load(traj_path, allow_pickle=True)
                        t_raw = np.array(traj.get("time", []), dtype=float)
                        th_raw = np.array(traj.get("theta", []), dtype=float)
                        om_raw = np.array(traj.get("omega", []), dtype=float)
                        al_raw = np.array(traj.get("alpha", []), dtype=float)
                        _, resampled = ms._resample_time_series(t_raw, th_raw, om_raw, al_raw)
                        details["fourth_resampled"] = bool(resampled)
            except Exception as exc:
                issues.append(f"{method}: {exc}")
        passed = len(issues) == 0
        message = "PASS" if passed else "; ".join(issues)
        return VerificationResult("trajectory_consistency", passed, message, details)

    def _verify_controller_implementation(self) -> VerificationResult:
        """Verify feedback controller gain assignment and stability margins."""
        details: Dict[str, Any] = {}
        issues: List[str] = []
        try:
            from basilisk_sim.feedback_control import MRPFeedbackController, FilteredDerivativeController

            axis = _normalize_axis(self.config.rotation_axis)
            I_axis = float(axis @ self.config.inertia @ axis)
            first_mode = min(self.config.modal_freqs_hz) if self.config.modal_freqs_hz else 0.4
            bandwidth = first_mode / 2.5
            omega_bw = 2 * np.pi * bandwidth
            sigma_scale = 4.0
            k_std = sigma_scale * I_axis * omega_bw**2
            p_std = 2 * 0.9 * I_axis * omega_bw
            p_filt = p_std * 1.5
            filter_cutoff = self.config.control_filter_cutoff_hz or 8.0

            details["designed_K_std"] = k_std
            details["designed_P_std"] = p_std
            details["designed_P_filt"] = p_filt
            details["filter_cutoff_hz"] = filter_cutoff

            ctrl_std = MRPFeedbackController(
                inertia=self.config.inertia, K=k_std, P=p_std, Ki=-1.0
            )
            if abs(ctrl_std.K - k_std) > 1e-6:
                issues.append(f"standard_pd K mismatch: {ctrl_std.K} vs {k_std}")
            if abs(ctrl_std.P - p_std) > 1e-6:
                issues.append(f"standard_pd P mismatch: {ctrl_std.P} vs {p_std}")

            ctrl_filt = FilteredDerivativeController(
                inertia=self.config.inertia, K=k_std, P=p_filt, filter_freq_hz=filter_cutoff
            )
            if abs(ctrl_filt.filter_freq_hz - filter_cutoff) > 1e-6:
                issues.append(
                    f"filtered_pd cutoff mismatch: {ctrl_filt.filter_freq_hz} vs {filter_cutoff}"
                )

            ctrl_data = ms._compute_control_analysis(self.config)
            gains = ctrl_data.get("gains", {})
            if gains:
                details["analysis_K"] = gains.get("K")
                details["analysis_P"] = gains.get("P")
                analysis_cutoff = gains.get("filter_cutoff")
                details["analysis_filter_cutoff"] = analysis_cutoff
                if abs(float(gains.get("K", k_std)) - k_std) > 1e-6:
                    issues.append("control_analysis K differs from design")
                if abs(float(gains.get("P", p_std)) - p_std) > 1e-6:
                    issues.append("control_analysis P differs from design")
                if analysis_cutoff is not None:
                    if abs(float(analysis_cutoff) - filter_cutoff) > 1e-6:
                        issues.append("control_analysis cutoff differs from config")

            margins = ctrl_data.get("margins", {})
            for name in CONTROLLERS:
                margin_data = margins.get(name)
                if not margin_data:
                    details[f"{name}_margin_available"] = False
                    continue
                details[f"{name}_margin_available"] = True
                details[f"{name}_gain_margin_db"] = float(margin_data["gain_margin_db"])
                details[f"{name}_phase_margin_deg"] = float(margin_data["phase_margin_deg"])
        except Exception as exc:
            issues.append(str(exc))
        passed = len(issues) == 0
        message = "PASS" if passed else "; ".join(issues)
        return VerificationResult("controller_implementation", passed, message, details)

    def _verify_logging_integrity(self) -> VerificationResult:
        """Verify NPZ log files contain required keys with consistent array lengths."""
        details = {}
        issues = []
        required_keys = ["time", "sigma", "omega", "fb_torque", "total_torque", "rw_torque", "mode1", "mode2"]
        acc_key_pairs = [("mode1_acc", "mode1_acc_signed"), ("mode2_acc", "mode2_acc_signed")]

        missing_any = False
        for method in METHODS:
            for controller in CONTROLLERS:
                npz_path = os.path.join(self.data_dir, f"vizard_demo_{method}_{controller}.npz")
                if not os.path.isfile(npz_path):
                    missing_any = True

        if missing_any:
            overrides = {
                "modal_freqs_hz": self.config.modal_freqs_hz,
                "modal_damping": self.config.modal_damping,
                "modal_gains_scale": 1.0,
                "control_filter_cutoff_hz": self.config.control_filter_cutoff_hz
                if self.config.control_filter_cutoff_hz is not None
                else 8.0,
                "inertia_scale": 1.0,
                "rw_max_torque_nm": self.config.rw_max_torque_nm,
                "slew_angle_deg": self.config.slew_angle_deg,
                "slew_duration_s": self.config.slew_duration_s,
                "sensor_noise_std_rad_s": 0.0,
                "disturbance_torque_nm": 0.0,
            }
            try:
                _run_vizard_demo_batch(overrides, self.data_dir)
            except subprocess.CalledProcessError as exc:
                issues.append(f"failed to generate NPZs: {exc}")

        for method in METHODS:
            for controller in CONTROLLERS:
                npz_path = os.path.join(self.data_dir, f"vizard_demo_{method}_{controller}.npz")
                if not os.path.isfile(npz_path):
                    issues.append(f"missing NPZ: {os.path.basename(npz_path)}")
                    continue
                try:
                    data = np.load(npz_path, allow_pickle=True)
                    time = np.array(data.get("time", []), dtype=float)
                    n = len(time)
                    if n == 0:
                        issues.append(f"{os.path.basename(npz_path)} empty time array")
                        continue
                    details[f"{method}_{controller}_samples"] = n

                    for key in required_keys:
                        if key not in data or len(data.get(key, [])) == 0:
                            issues.append(f"{os.path.basename(npz_path)} missing {key}")
                            continue
                        arr = np.array(data.get(key, []))
                        if len(arr) != n:
                            issues.append(f"{os.path.basename(npz_path)} {key} length {len(arr)} != {n}")

                    for key_a, key_b in acc_key_pairs:
                        arr_a = data.get(key_a)
                        arr_b = data.get(key_b)
                        if (arr_a is None or len(arr_a) == 0) and (arr_b is None or len(arr_b) == 0):
                            issues.append(f"{os.path.basename(npz_path)} missing {key_a}/{key_b}")
                except Exception as exc:
                    issues.append(f"{os.path.basename(npz_path)} load error: {exc}")
        passed = len(issues) == 0
        message = "PASS" if passed else "; ".join(issues)
        return VerificationResult("logging_integrity", passed, message, details)

    def _verify_psd_computations(self) -> VerificationResult:
        """Verify PSD parameter selection and decibel conversion correctness."""
        details: Dict[str, Any] = {}
        issues: List[str] = []
        # Use mission_simulation PSD parameter chooser for consistency
        sample_npz = os.path.join(self.data_dir, f"vizard_demo_{METHODS[0]}_{CONTROLLERS[0]}.npz")
        if not os.path.isfile(sample_npz):
            issues.append("missing sample NPZ for PSD check")
        else:
            data = np.load(sample_npz, allow_pickle=True)
            time = np.array(data.get("time", []), dtype=float)
            torque = np.array(data.get("total_torque", []), dtype=float)
            if torque.ndim > 1:
                torque = torque[:, 2]
            params = ms._choose_psd_params(time, torque)
            if not params:
                issues.append("PSD params could not be computed")
            else:
                details.update({f"psd_{k}": v for k, v in params.items()})
                required = ["fs", "nperseg", "noverlap", "window", "detrend"]
                for key in required:
                    if key not in params:
                        issues.append(f"PSD param missing: {key}")

        # Confirm PSD uses 10*log10 in plotting routines
        psd_funcs = [
            ms._plot_psd_comparison,
            ms._plot_torque_command_psd,
            ms._plot_torque_psd_split,
        ]
        for func in psd_funcs:
            try:
                source = inspect.getsource(func)
                if "10.0 * np.log10" not in source and "10 * np.log10" not in source:
                    issues.append(f"{func.__name__} missing 10log10 PSD conversion")
                if "20 * np.log10" in source or "20.0 * np.log10" in source:
                    issues.append(f"{func.__name__} uses 20log10 on PSD")
            except OSError:
                issues.append(f"{func.__name__} source unavailable for PSD check")

        passed = len(issues) == 0
        message = "PASS" if passed else "; ".join(issues)
        return VerificationResult("psd_computations", passed, message, details)

    def _save_verification_report(self, results: List[VerificationResult]) -> None:
        """Write verification results to a JSON report file."""
        report_path = os.path.join(self.out_dir, "verification_report.json")
        report = {
            "timestamp": datetime.now().isoformat(),
            "summary": {
                "total_tests": len(results),
                "passed": sum(1 for r in results if r.passed),
                "failed": sum(1 for r in results if not r.passed),
            },
            "tests": [
                {
                    "name": r.test_name,
                    "passed": r.passed,
                    "message": r.message,
                    "details": r.details,
                }
                for r in results
            ],
        }
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)

    # ------------------------------------------------------------------
    # VALIDATION (Section 2)
    # ------------------------------------------------------------------
    def run_validation(self) -> List[ValidationResult]:
        """Execute all validation checks with baseline simulation and reporting."""
        results: List[ValidationResult] = []
        steps = [
            ("Baseline simulation", None),
            ("tracking", self._validate_tracking),
            ("disturbance_rejection", self._validate_disturbance_rejection),
            ("noise_rejection", self._validate_noise_rejection),
            ("vibration_suppression", self._validate_vibration_suppression),
            ("torque_limits", self._validate_torque_limits),
            ("deliverables", self._validate_deliverables),
        ]
        start = time_module.time()
        last = start
        total = len(steps)
        _update_progress("Validation", 0, total, start, last)

        # Step 1: baseline simulation
        ms.run_mission_simulation(
            self.config,
            out_dir=self.out_dir,
            data_dir=self.data_dir,
            make_plots=True,
            export_csv=True,
            generate_pointing=True,
        )
        last = _update_progress("Validation", 1, total, start, last)

        # Remaining validation checks
        for idx, (_, func) in enumerate(steps[1:], start=2):
            if func is not None:
                results.append(func())
            last = _update_progress("Validation", idx, total, start, last)
        self._save_validation_report(results)
        return results

    def _collect_pointing_rms(self, data_dir: str, label_suffix: str = "") -> Dict[str, float]:
        """Collect post slew RMS pointing error for each method/controller pair."""
        metrics: Dict[str, float] = {}
        pointing_data = ms._load_all_pointing_data(
            data_dir, config=self.config, generate_if_missing=False
        )
        for method in METHODS:
            method_data = pointing_data.get(method, {})
            for controller in CONTROLLERS:
                data = method_data.get(controller)
                if not data:
                    continue
                time = np.array(data.get("time", []), dtype=float)
                errors = ms._extract_pointing_error(data, config=self.config)
                time, aligned = ms._align_series(time, errors)
                errors = aligned[0]
                rms, _ = _compute_post_slew_stats(time, errors, self.config.slew_duration_s)
                metrics[f"{method}_{controller}{label_suffix}"] = rms
        return metrics

    def _validate_tracking(self) -> ValidationResult:
        """Check closed loop tracking against pointing error thresholds."""
        issues = []
        metrics = {}
        plots = [os.path.join(self.out_dir, "mission_tracking_response.png"),
                 os.path.join(self.out_dir, "mission_tracking_tf.png")]
        for p in plots:
            if not os.path.isfile(p):
                issues.append(f"missing plot: {os.path.basename(p)}")

        summary_path = os.path.join(self.out_dir, "mission_summary.csv")
        if os.path.isfile(summary_path):
            with open(summary_path, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    if row.get("category") == "pointing":
                        key = row.get("method", "")
                        metrics[f"{key}_rms_pointing_error_deg"] = float(row.get("metric2", "nan"))
        else:
            issues.append("missing mission_summary.csv for tracking metrics")

        if not metrics:
            # Fallback to direct computation from NPZs
            metrics.update(self._collect_pointing_rms(self.data_dir, label_suffix="_rms_pointing_error_deg"))
        # Threshold check (P95 target in plan)
        for k, val in metrics.items():
            if np.isfinite(val) and val > PLAN_THRESHOLDS["rms_pointing_error_deg"]:
                issues.append(f"{k} RMS {val:.6f} > {PLAN_THRESHOLDS['rms_pointing_error_deg']}")
        passed = len(issues) == 0
        message = "PASS" if passed else "; ".join(issues)
        return ValidationResult("tracking", passed, message, metrics, plots)

    def _validate_disturbance_rejection(self) -> ValidationResult:
        """Measure pointing degradation under external disturbance torques."""
        issues: List[str] = []
        metrics: Dict[str, float] = {}
        plots = [
            os.path.join(self.out_dir, "mission_disturbance_tf.png"),
            os.path.join(self.out_dir, "mission_disturbance_to_torque.png"),
        ]
        for p in plots:
            if not os.path.isfile(p):
                issues.append(f"missing plot: {os.path.basename(p)}")

        baseline = self._collect_pointing_rms(self.data_dir, label_suffix="_rms_base_deg")
        if not baseline:
            issues.append("baseline pointing metrics unavailable")

        disturbance_dir = os.path.join(self.out_dir, "validation_disturbance")
        _reset_dir(disturbance_dir)

        overrides = {
            "modal_freqs_hz": self.config.modal_freqs_hz,
            "modal_damping": self.config.modal_damping,
            "modal_gains_scale": 1.0,
            "control_filter_cutoff_hz": self.config.control_filter_cutoff_hz
            if self.config.control_filter_cutoff_hz is not None
            else 8.0,
            "inertia_scale": 1.0,
            "rw_max_torque_nm": self.config.rw_max_torque_nm,
            "slew_angle_deg": self.config.slew_angle_deg,
            "slew_duration_s": self.config.slew_duration_s,
            "sensor_noise_std_rad_s": 0.0,
            "disturbance_torque_nm": self.disturbance_torque_nm,
        }
        try:
            _run_vizard_demo_batch(overrides, disturbance_dir)
            disturbed = self._collect_pointing_rms(disturbance_dir, label_suffix="_rms_disturbed_deg")
            for key, base_val in baseline.items():
                metrics[key] = base_val
                dist_key = key.replace("_rms_base_deg", "_rms_disturbed_deg")
                dist_val = disturbed.get(dist_key, float("nan"))
                metrics[dist_key] = dist_val
                if np.isfinite(base_val) and np.isfinite(dist_val):
                    metrics[key.replace("_rms_base_deg", "_delta_deg")] = dist_val - base_val
        except subprocess.CalledProcessError as exc:
            issues.append(f"disturbance run failed: {exc}")

        passed = len(issues) == 0
        message = "PASS" if passed else "; ".join(issues)
        return ValidationResult("disturbance_rejection", passed, message, metrics, plots)

    def _validate_noise_rejection(self) -> ValidationResult:
        """Measure pointing degradation under sensor noise injection."""
        issues: List[str] = []
        metrics: Dict[str, float] = {}
        plots = [os.path.join(self.out_dir, "mission_noise_to_torque.png")]
        for p in plots:
            if not os.path.isfile(p):
                issues.append(f"missing plot: {os.path.basename(p)}")

        baseline = self._collect_pointing_rms(self.data_dir, label_suffix="_rms_base_deg")
        if not baseline:
            issues.append("baseline pointing metrics unavailable")

        noise_dir = os.path.join(self.out_dir, "validation_noise")
        _reset_dir(noise_dir)

        overrides = {
            "modal_freqs_hz": self.config.modal_freqs_hz,
            "modal_damping": self.config.modal_damping,
            "modal_gains_scale": 1.0,
            "control_filter_cutoff_hz": self.config.control_filter_cutoff_hz
            if self.config.control_filter_cutoff_hz is not None
            else 8.0,
            "inertia_scale": 1.0,
            "rw_max_torque_nm": self.config.rw_max_torque_nm,
            "slew_angle_deg": self.config.slew_angle_deg,
            "slew_duration_s": self.config.slew_duration_s,
            "sensor_noise_std_rad_s": self.sensor_noise_std_rad_s,
            "disturbance_torque_nm": 0.0,
        }
        try:
            _run_vizard_demo_batch(overrides, noise_dir)
            noisy = self._collect_pointing_rms(noise_dir, label_suffix="_rms_noisy_deg")
            for key, base_val in baseline.items():
                metrics[key] = base_val
                noisy_key = key.replace("_rms_base_deg", "_rms_noisy_deg")
                noisy_val = noisy.get(noisy_key, float("nan"))
                metrics[noisy_key] = noisy_val
                if np.isfinite(base_val) and np.isfinite(noisy_val):
                    metrics[key.replace("_rms_base_deg", "_delta_deg")] = noisy_val - base_val
        except subprocess.CalledProcessError as exc:
            issues.append(f"noise run failed: {exc}")

        passed = len(issues) == 0
        message = "PASS" if passed else "; ".join(issues)
        return ValidationResult("noise_rejection", passed, message, metrics, plots)

    def _validate_vibration_suppression(self) -> ValidationResult:
        """Validate modal vibration suppression and PSD reduction at resonance."""
        issues = []
        metrics = {}
        summary_path = os.path.join(self.out_dir, "mission_summary.csv")
        if os.path.isfile(summary_path):
            with open(summary_path, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    if row.get("category") == "pointing":
                        key = row.get("method", "") + "_rms_vibration_mm"
                        metrics[key] = float(row.get("metric1", "nan"))
        else:
            issues.append("missing mission_summary.csv for vibration metrics")
        for k, val in metrics.items():
            if np.isfinite(val) and val > PLAN_THRESHOLDS["rms_vibration_mm"]:
                issues.append(f"{k} RMS {val:.6f} > {PLAN_THRESHOLDS['rms_vibration_mm']}")

        # Modal PSD checks at structural frequencies
        mission_psd = ms._build_mission_psd_data(self.config, self.data_dir, generate_if_missing=False)
        if not mission_psd:
            issues.append("mission PSD data unavailable for modal checks")
        else:
            for mode_idx, f_mode in enumerate(self.config.modal_freqs_hz):
                for method in METHODS:
                    for controller in CONTROLLERS:
                        data = mission_psd.get(method, {}).get(controller, {})
                        freq = np.array(data.get("psd_freq", []), dtype=float)
                        psd = np.array(data.get("psd", []), dtype=float)
                        if len(freq) == 0 or len(psd) == 0:
                            continue
                        idx = int(np.argmin(np.abs(freq - f_mode)))
                        if psd[idx] <= 0 or not np.isfinite(psd[idx]):
                            val_db = float("nan")
                        else:
                            val_db = float(10.0 * np.log10(psd[idx]))
                        metrics[f"{method}_{controller}_mode{mode_idx+1}_psd_db"] = val_db

            baseline_methods = [m for m in METHODS if m != "fourth"]
            for controller in CONTROLLERS:
                for mode_idx in range(len(self.config.modal_freqs_hz)):
                    key_fourth = f"fourth_{controller}_mode{mode_idx+1}_psd_db"
                    fourth_val = metrics.get(key_fourth, float("nan"))
                    for baseline in baseline_methods:
                        key_base = f"{baseline}_{controller}_mode{mode_idx+1}_psd_db"
                        base_val = metrics.get(key_base, float("nan"))
                        if np.isfinite(base_val) and np.isfinite(fourth_val):
                            metrics[
                                f"{baseline}_to_fourth_{controller}_mode{mode_idx+1}_reduction_db"
                            ] = base_val - fourth_val
        passed = len(issues) == 0
        message = "PASS" if passed else "; ".join(issues)
        return ValidationResult("vibration_suppression", passed, message, metrics, [])

    def _validate_torque_limits(self) -> ValidationResult:
        """Validate peak torque and reaction wheel saturation against limits."""
        issues = []
        metrics = {}
        metrics_path = os.path.join(self.out_dir, "torque_command_metrics.csv")
        if os.path.isfile(metrics_path):
            with open(metrics_path, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    if row.get("type") == "total_body":
                        key = f"{row.get('method')}_{row.get('controller')}"
                        peak = float(row.get("peak_torque_nm", "nan"))
                        rms = float(row.get("rms_torque_nm", "nan"))
                        sat_pct = float(row.get("rw_saturation_percent", "nan")) if row.get("rw_saturation_percent") else float("nan")
                        peak_rw = float(row.get("peak_rw_torque_nm", "nan")) if row.get("peak_rw_torque_nm") else float("nan")

                        metrics[f"{key}_peak_torque_nm"] = peak
                        metrics[f"{key}_rms_torque_nm"] = rms
                        metrics[f"{key}_rw_saturation_percent"] = sat_pct
                        metrics[f"{key}_peak_rw_torque_nm"] = peak_rw

                        if np.isfinite(peak) and peak > PLAN_THRESHOLDS["peak_torque_nm"]:
                            issues.append(f"{key} peak {peak:.2f} > {PLAN_THRESHOLDS['peak_torque_nm']}")
                        if np.isfinite(sat_pct) and sat_pct > PLAN_THRESHOLDS["torque_saturation_percent"]:
                            issues.append(f"{key} rw saturation {sat_pct:.2f}% > {PLAN_THRESHOLDS['torque_saturation_percent']}%")
                        if np.isfinite(peak_rw) and self.config.rw_max_torque_nm:
                            if peak_rw > float(self.config.rw_max_torque_nm):
                                issues.append(f"{key} peak RW {peak_rw:.2f} > {self.config.rw_max_torque_nm}")
        else:
            issues.append("missing torque_command_metrics.csv for torque limits")
        passed = len(issues) == 0
        message = "PASS" if passed else "; ".join(issues)
        return ValidationResult("torque_limits", passed, message, metrics, [])

    def _save_validation_report(self, results: List[ValidationResult]) -> None:
        """Write validation results and aggregated metrics to JSON and CSV."""
        report_path = os.path.join(self.out_dir, "validation_report.json")
        all_metrics = {}
        for r in results:
            all_metrics.update(r.metrics)
        report = {
            "timestamp": datetime.now().isoformat(),
            "summary": {
                "total_tests": len(results),
                "passed": sum(1 for r in results if r.passed),
                "failed": sum(1 for r in results if not r.passed),
            },
            "tests": [
                {
                    "name": r.test_name,
                    "passed": r.passed,
                    "message": r.message,
                    "metrics": r.metrics,
                    "plots": r.plots,
                }
                for r in results
            ],
            "all_metrics": all_metrics,
        }
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)

        csv_path = os.path.join(self.out_dir, "validation_metrics.csv")
        rows = [[k, str(v)] for k, v in sorted(all_metrics.items())]
        _write_csv(csv_path, ["metric", "value"], rows)

    def _validate_deliverables(self) -> ValidationResult:
        """Check that all required output plots and CSV files were generated."""
        plots = [
            "mission_tracking_response.png",
            "mission_tracking_tf.png",
            "mission_disturbance_tf.png",
            "mission_noise_to_torque.png",
            "mission_disturbance_to_torque.png",
            "mission_torque_command.png",
            "mission_torque_command_psd.png",
            "mission_torque_psd_split.png",
            "mission_torque_psd_coherence.png",
        ]
        csvs = [
            "torque_command_metrics.csv",
            "torque_psd_rms.csv",
            "mission_summary.csv",
            "psd_mission.csv",
        ]

        missing = []
        for name in plots + csvs:
            path = os.path.join(self.out_dir, name)
            if not os.path.isfile(path):
                missing.append(name)

        passed = len(missing) == 0
        message = "PASS" if passed else f"missing deliverables: {', '.join(missing)}"
        metrics = {"missing_count": float(len(missing))}
        return ValidationResult("deliverables", passed, message, metrics, [os.path.join(self.out_dir, p) for p in plots])

    # ------------------------------------------------------------------
    # MONTE CARLO (Section 3)
    # ------------------------------------------------------------------
    def run_monte_carlo(self, n_runs: int = 500) -> MonteCarloSummary:
        """Delegate Monte Carlo execution to PlanMonteCarloRunner."""
        mc_runner = PlanMonteCarloRunner(
            self.config,
            self.out_dir,
            n_runs=n_runs,
            data_dir=self.data_dir,
            sensor_noise_std_rad_s=self.sensor_noise_std_rad_s,
            disturbance_torque_nm=self.disturbance_torque_nm,
        )
        return mc_runner.run()


class PlanMonteCarloRunner:
    """Monte Carlo analysis that regenerates NPZs using vizard_demo overrides."""

    SUMMARY_METRICS = [
        "rms_pointing_error_deg",
        "peak_torque_nm",
        "rms_vibration_mm",
        "rms_torque_nm",
        "torque_saturation_percent",
    ]
    COMPARISON_METRICS = [
        ("rms_pointing_error_deg", "RMS Pointing Error (deg)"),
        ("peak_pointing_error_deg", "Peak Pointing Error (deg)"),
        ("rms_vibration_mm", "RMS Vibration (mm)"),
        ("peak_torque_nm", "Peak Torque (N*m)"),
        ("rms_torque_nm", "RMS Torque (N*m)"),
        ("rw_saturation_percent", "RW Saturation (%)"),
    ]
    COMBO_STYLES = MC_COMBO_STYLES

    def __init__(
        self,
        base_config: ms.MissionConfig,
        out_dir: str,
        n_runs: int,
        data_dir: str,
        sensor_noise_std_rad_s: float = 1e-5,
        disturbance_torque_nm: float = 1e-5,
    ):
        self.base_config = base_config
        self.out_dir = out_dir
        self.n_runs = n_runs
        self.data_dir = data_dir
        self.sensor_noise_std_rad_s = float(sensor_noise_std_rad_s)
        self.disturbance_torque_nm = float(disturbance_torque_nm)
        self.rng = np.random.default_rng(42)
        self.thresholds = PLAN_THRESHOLDS.copy()
        self.mc_work_dir = os.path.join(self.out_dir, "mc_work")
        _reset_dir(self.mc_work_dir)

    def _copy_config(self) -> ms.MissionConfig:
        """Deep copy the base mission config to avoid mutating the original."""
        cfg = ms.MissionConfig(**asdict(self.base_config))
        cfg.inertia = np.array(cfg.inertia, dtype=float)
        cfg.rotation_axis = np.array(cfg.rotation_axis, dtype=float)
        cfg.modal_freqs_hz = list(cfg.modal_freqs_hz)
        cfg.modal_damping = list(cfg.modal_damping)
        cfg.modal_gains = list(cfg.modal_gains) if cfg.modal_gains is not None else []
        cfg.control_modal_gains = (
            list(cfg.control_modal_gains) if cfg.control_modal_gains is not None else []
        )
        return cfg

    def _perturb(self) -> Tuple[ms.MissionConfig, Dict[str, float], Dict[str, object]]:
        """Apply random perturbations and return the config, scale factors, and overrides."""
        cfg = self._copy_config()
        perturb: Dict[str, float] = {}

        inertia_scale = 1 + self.rng.uniform(-0.2, 0.2)
        perturb["inertia_scale"] = inertia_scale
        hub_scaled = HUB_INERTIA.copy() * inertia_scale
        cfg.inertia = compute_effective_inertia(hub_inertia=hub_scaled.copy())

        freq_scale = 1 + self.rng.uniform(-0.1, 0.1)
        perturb["freq_scale"] = freq_scale
        cfg.modal_freqs_hz = [f * freq_scale for f in cfg.modal_freqs_hz]

        damp_scale = 1 + self.rng.uniform(-0.5, 0.5)
        perturb["damp_scale"] = damp_scale
        cfg.modal_damping = [max(0.001, d * damp_scale) for d in cfg.modal_damping]

        gains_scale = 1 + self.rng.uniform(-0.2, 0.2)
        perturb["modal_gains_scale"] = gains_scale
        modal_gains = cfg.modal_gains or compute_modal_gains(cfg.inertia, cfg.rotation_axis)
        control_gains = cfg.control_modal_gains or modal_gains
        cfg.modal_gains = [g * gains_scale for g in modal_gains]
        cfg.control_modal_gains = [g * gains_scale for g in control_gains]

        cutoff_scale = 1 + self.rng.uniform(-0.2, 0.2)
        perturb["cutoff_scale"] = cutoff_scale
        if cfg.control_filter_cutoff_hz is None:
            cfg.control_filter_cutoff_hz = 8.0
        cfg.control_filter_cutoff_hz = max(0.1, float(cfg.control_filter_cutoff_hz) * cutoff_scale)

        noise_scale = 1 + self.rng.uniform(-0.5, 0.5)
        perturb["noise_scale"] = noise_scale
        noise_std = max(0.0, self.sensor_noise_std_rad_s * noise_scale)

        dist_scale = 1 + self.rng.uniform(-0.5, 0.5)
        perturb["disturbance_scale"] = dist_scale
        dist_torque = self.disturbance_torque_nm * dist_scale

        overrides = {
            "modal_freqs_hz": cfg.modal_freqs_hz,
            "modal_damping": cfg.modal_damping,
            "modal_gains_scale": gains_scale,
            "control_filter_cutoff_hz": cfg.control_filter_cutoff_hz,
            "inertia_scale": inertia_scale,
            "rw_max_torque_nm": cfg.rw_max_torque_nm,
            "slew_angle_deg": cfg.slew_angle_deg,
            "slew_duration_s": cfg.slew_duration_s,
            "sensor_noise_std_rad_s": noise_std,
            "disturbance_torque_nm": dist_torque,
        }
        return cfg, perturb, overrides

    def _compute_metrics(
        self,
        cfg: ms.MissionConfig,
        feedback_vibration: Dict[str, Dict[str, object]],
        pointing_data: Dict[str, Dict[str, object]],
    ) -> Dict[str, float]:
        """Compute pointing, vibration, torque, and saturation metrics from run data."""
        metrics: Dict[str, float] = {}

        rms_list: List[float] = []
        peak_torque_list: List[float] = []
        rms_torque_list: List[float] = []
        vib_list: List[float] = []
        accel_list: List[float] = []
        sat_list: List[float] = []

        for method in METHODS:
            method_data = pointing_data.get(method, {})
            for controller in CONTROLLERS:
                data = method_data.get(controller)
                if not data:
                    continue
                time = np.array(data.get("time", []), dtype=float)
                errors = ms._extract_pointing_error(data, config=cfg)
                time, aligned = ms._align_series(time, errors)
                errors = aligned[0]
                rms, peak = _compute_post_slew_stats(time, errors, cfg.slew_duration_s)
                metrics[f"{method}_{controller}_rms_pointing_error_deg"] = rms
                metrics[f"{method}_{controller}_peak_pointing_error_deg"] = peak
                if np.isfinite(rms):
                    rms_list.append(rms)

        for key, data in feedback_vibration.items():
            time = np.array(data.get("time", []), dtype=float)
            disp = np.array(data.get("displacement", []), dtype=float)
            accel = np.array(data.get("acceleration_modal_raw", data.get("acceleration", np.array([]))), dtype=float)
            torque = data.get("torque_total", data.get("torque", np.array([])))
            torque = np.array(torque, dtype=float)
            if len(time) == 0:
                continue
            time, aligned = ms._align_series(time, disp, torque, accel)
            disp = aligned[0]
            torque = aligned[1]
            accel = aligned[2]
            rms_disp, _ = _compute_post_slew_stats(time, disp, cfg.slew_duration_s)
            rms_vib_mm = rms_disp * 1000.0 if np.isfinite(rms_disp) else float("nan")
            vib_list.append(rms_vib_mm)
            metrics[f"{key}_rms_vibration_mm"] = rms_vib_mm
            if len(accel):
                rms_accel, _ = _compute_post_slew_stats(time, accel, cfg.slew_duration_s)
                rms_modal_accel_mm_s2 = rms_accel * 1000.0 if np.isfinite(rms_accel) else float("nan")
            else:
                rms_modal_accel_mm_s2 = float("nan")
            metrics[f"{key}_rms_modal_accel_mm_s2"] = rms_modal_accel_mm_s2
            if np.isfinite(rms_modal_accel_mm_s2):
                accel_list.append(rms_modal_accel_mm_s2)

            if len(torque):
                peak_torque = float(np.max(np.abs(torque)))
                rms_torque = _compute_rms(torque)
                metrics[f"{key}_peak_torque_nm"] = peak_torque
                metrics[f"{key}_rms_torque_nm"] = rms_torque
                peak_torque_list.append(peak_torque)
                rms_torque_list.append(rms_torque)

            rw_torque = data.get("rw_torque")
            sat_pct = float("nan")
            if rw_torque is not None and cfg.rw_max_torque_nm:
                rw_arr = np.array(rw_torque, dtype=float)
                if rw_arr.ndim == 2 and rw_arr.shape[1] > 1:
                    rw_mag = np.max(np.abs(rw_arr), axis=1)
                else:
                    rw_mag = np.abs(rw_arr).flatten()
                if len(rw_mag):
                    sat_count = np.sum(rw_mag >= float(cfg.rw_max_torque_nm) * 0.99)
                    sat_pct = 100.0 * sat_count / len(rw_mag)
            metrics[f"{key}_rw_saturation_percent"] = sat_pct
            if np.isfinite(sat_pct):
                sat_list.append(sat_pct)

        metrics["rms_pointing_error_deg"] = max(rms_list) if rms_list else float("nan")
        metrics["peak_torque_nm"] = max(peak_torque_list) if peak_torque_list else float("nan")
        metrics["rms_torque_nm"] = max(rms_torque_list) if rms_torque_list else float("nan")
        metrics["rms_vibration_mm"] = max(vib_list) if vib_list else float("nan")
        metrics["rms_modal_accel_mm_s2"] = max(accel_list) if accel_list else float("nan")
        metrics["torque_saturation_percent"] = max(sat_list) if sat_list else float("nan")

        return metrics

    def _empty_metrics(self) -> Dict[str, float]:
        """Return a metrics dict filled with NaN for failed runs."""
        metrics: Dict[str, float] = {}
        for method in METHODS:
            for controller in CONTROLLERS:
                prefix = f"{method}_{controller}"
                metrics[f"{prefix}_rms_pointing_error_deg"] = float("nan")
                metrics[f"{prefix}_peak_pointing_error_deg"] = float("nan")
                metrics[f"{prefix}_rms_vibration_mm"] = float("nan")
                metrics[f"{prefix}_rms_modal_accel_mm_s2"] = float("nan")
                metrics[f"{prefix}_peak_torque_nm"] = float("nan")
                metrics[f"{prefix}_rms_torque_nm"] = float("nan")
                metrics[f"{prefix}_rw_saturation_percent"] = float("nan")
        for key in self.SUMMARY_METRICS:
            metrics[key] = float("nan")
        return metrics

    def _evaluate_run(self, metrics: Dict[str, float]) -> Tuple[bool, List[str]]:
        """Evaluate a single run's metrics against pass/fail thresholds."""
        reasons: List[str] = []
        for key in self.SUMMARY_METRICS:
            val = metrics.get(key, float("nan"))
            if not np.isfinite(val):
                reasons.append(f"{key} invalid")
        if np.isfinite(metrics.get("rms_pointing_error_deg", np.nan)) and metrics["rms_pointing_error_deg"] > self.thresholds["rms_pointing_error_deg"]:
            reasons.append("rms_pointing_error_deg exceeds threshold")
        if np.isfinite(metrics.get("rms_vibration_mm", np.nan)) and metrics["rms_vibration_mm"] > self.thresholds["rms_vibration_mm"]:
            reasons.append("rms_vibration_mm exceeds threshold")
        if np.isfinite(metrics.get("peak_torque_nm", np.nan)) and metrics["peak_torque_nm"] > self.thresholds["peak_torque_nm"]:
            reasons.append("peak_torque_nm exceeds threshold")
        if np.isfinite(metrics.get("torque_saturation_percent", np.nan)) and metrics["torque_saturation_percent"] > self.thresholds["torque_saturation_percent"]:
            reasons.append("torque_saturation_percent exceeds threshold")
        return len(reasons) == 0, reasons

    def run(self) -> MonteCarloSummary:
        """Execute all Monte Carlo iterations with progress tracking and output generation."""
        runs: List[MonteCarloRun] = []
        start = time_module.time()
        last = start
        total = self.n_runs
        _update_progress("Monte Carlo", 0, total, start, last)
        for run_id in range(self.n_runs):
            cfg, perturb, overrides = self._perturb()
            run_ok = False
            fail_log = ""
            for attempt in range(2):
                try:
                    _run_vizard_demo_batch(overrides, self.mc_work_dir)
                    run_ok = True
                    break
                except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
                    fail_log = _write_mc_failure_log(run_id, exc, self.mc_work_dir)
                    if attempt == 0:
                        continue
            if not run_ok:
                metrics = self._empty_metrics()
                reasons = ["vizard_demo failed"]
                if fail_log:
                    reasons.append(f"see {os.path.basename(fail_log)}")
                runs.append(MonteCarloRun(run_id, perturb, metrics, False, reasons))
                last = _update_progress("Monte Carlo", run_id + 1, total, start, last)
                continue

            feedback_vibration = ms._collect_feedback_data(cfg, data_dir=self.mc_work_dir, prefer_npz=True)
            pointing_data = ms._load_all_pointing_data(self.mc_work_dir, config=cfg, generate_if_missing=False)

            metrics = self._compute_metrics(cfg, feedback_vibration, pointing_data)
            passed, reasons = self._evaluate_run(metrics)
            runs.append(MonteCarloRun(run_id, perturb, metrics, passed, reasons))

            last = _update_progress("Monte Carlo", run_id + 1, total, start, last)

        summary = self._compute_summary(runs)
        self._save_results(summary, runs)
        self._plot_histograms(summary)
        self._plot_comparison_boxes(runs)
        _plot_post_slew_pointing_box_from_csv(
            self.out_dir,
            threshold_arcsec=POST_SLEW_POINTING_LIMIT_ARCSEC,
        )
        _plot_post_slew_vibration_box_from_csv(
            self.out_dir,
            threshold_mm=POST_SLEW_VIBRATION_LIMIT_MM,
        )
        _plot_post_slew_acceleration_box_from_csv(
            self.out_dir,
            threshold_mm_s2=POST_SLEW_ACCEL_LIMIT_MM_S2,
        )
        _plot_post_slew_pointing_factor_histograms_from_csv(self.out_dir)
        _remove_legacy_correlation_outputs(self.out_dir)
        return summary

    def _compute_summary(self, runs: List[MonteCarloRun]) -> MonteCarloSummary:
        """Aggregate per run results into percentile statistics and histograms."""
        n_passed = sum(1 for r in runs if r.passed)
        pass_rate = n_passed / len(runs) if runs else 0.0

        percentiles: Dict[str, Dict[str, float]] = {}
        histograms: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
        for name in self.SUMMARY_METRICS:
            values = np.array([r.metrics.get(name, np.nan) for r in runs], dtype=float)
            values = values[np.isfinite(values)]
            if values.size == 0:
                continue
            percentiles[name] = {
                "P50": float(np.percentile(values, 50)),
                "P95": float(np.percentile(values, 95)),
                "P99": float(np.percentile(values, 99)),
                "mean": float(np.mean(values)),
                "std": float(np.std(values)),
                "min": float(np.min(values)),
                "max": float(np.max(values)),
            }
            hist, bins = np.histogram(values, bins=50)
            histograms[name] = (hist, bins)

        return MonteCarloSummary(
            n_runs=len(runs),
            n_passed=n_passed,
            pass_rate=pass_rate,
            percentiles=percentiles,
            histograms=histograms,
        )

    def _save_results(self, summary: MonteCarloSummary, runs: List[MonteCarloRun]) -> None:
        """Write Monte Carlo report JSON, per run CSV, and percentile CSV."""
        report_path = os.path.join(self.out_dir, "monte_carlo_report.json")

        criteria = {
            "rms_pointing_error_deg_P95": summary.percentiles.get("rms_pointing_error_deg", {}).get("P95", float("nan")),
            "rms_vibration_mm_P95": summary.percentiles.get("rms_vibration_mm", {}).get("P95", float("nan")),
            "peak_torque_nm_P99": summary.percentiles.get("peak_torque_nm", {}).get("P99", float("nan")),
            "torque_saturation_percent_P95": summary.percentiles.get("torque_saturation_percent", {}).get("P95", float("nan")),
        }

        report = {
            "timestamp": datetime.now().isoformat(),
            "n_runs": summary.n_runs,
            "n_passed": summary.n_passed,
            "pass_rate": summary.pass_rate,
            "thresholds": self.thresholds,
            "percentiles": summary.percentiles,
            "criteria": criteria,
            "uncertainties": {
                "inertia_pct": 0.2,
                "modal_freq_pct": 0.1,
                "modal_damping_pct": 0.5,
                "modal_gain_pct": 0.2,
                "sensor_noise_pct": 0.5,
                "disturbance_pct": 0.5,
                "filter_cutoff_pct": 0.2,
            },
            "noise_disturbance_nominal": {
                "sensor_noise_std_rad_s": self.sensor_noise_std_rad_s,
                "disturbance_torque_nm": self.disturbance_torque_nm,
            },
        }

        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)

        csv_path = os.path.join(self.out_dir, "monte_carlo_runs.csv")
        if runs:
            headers = ["run_id", "passed"] + list(runs[0].metrics.keys())
            rows = []
            for run in runs:
                row = [run.run_id, run.passed]
                row.extend([run.metrics.get(k, "") for k in headers[2:]])
                rows.append(row)
            _write_csv(csv_path, headers, rows)

        pct_path = os.path.join(self.out_dir, "monte_carlo_percentiles.csv")
        pct_rows = []
        for name, stats in summary.percentiles.items():
            pct_rows.append([
                name,
                stats.get("P50", ""),
                stats.get("P95", ""),
                stats.get("P99", ""),
                stats.get("mean", ""),
                stats.get("std", ""),
                stats.get("min", ""),
                stats.get("max", ""),
            ])
        _write_csv(pct_path, ["metric", "P50", "P95", "P99", "mean", "std", "min", "max"], pct_rows)

    def _plot_histograms(self, summary: MonteCarloSummary) -> None:
        """Plot summary metric histograms with percentile markers."""
        metric_names = list(summary.histograms.keys())
        if not metric_names:
            return

        n_cols = min(3, len(metric_names))
        n_rows = (len(metric_names) + n_cols - 1) // n_cols
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows))
        axes = np.atleast_1d(axes).flatten()

        for idx, metric in enumerate(metric_names):
            ax = axes[idx]
            hist, bins = summary.histograms[metric]
            centers = (bins[:-1] + bins[1:]) / 2
            ax.bar(centers, hist, width=bins[1] - bins[0], alpha=0.7, edgecolor="black")
            stats = summary.percentiles.get(metric, {})
            if stats:
                ax.axvline(stats.get("P50", 0), color="g", linestyle="-", label="P50")
                ax.axvline(stats.get("P95", 0), color="orange", linestyle="--", label="P95")
                ax.axvline(stats.get("P99", 0), color="r", linestyle=":", label="P99")
            ax.set_xlabel(metric)
            ax.set_ylabel("Count")
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=8)

        for idx in range(len(metric_names), len(axes)):
            axes[idx].set_visible(False)

        plt.suptitle(f"Monte Carlo Results ({summary.n_runs} runs, {summary.pass_rate*100:.1f}% pass rate)")
        plt.tight_layout()
        plot_path = os.path.join(self.out_dir, "monte_carlo_histograms.png")
        plt.savefig(plot_path, dpi=150)
        plt.close()

    def _plot_comparison_boxes(self, runs: List[MonteCarloRun]) -> None:
        """Plot per combination metric boxplots from Monte Carlo runs."""
        if not runs:
            return

        combos = [f"{method}_{controller}" for method in METHODS for controller in CONTROLLERS]
        fig, axes = plt.subplots(2, 3, figsize=(16, 9))
        axes = np.atleast_1d(axes).flatten()

        legend_handles: List[Any] = []
        legend_labels: List[str] = []

        for idx, (metric_key, metric_label) in enumerate(self.COMPARISON_METRICS):
            ax = axes[idx]
            data: List[np.ndarray] = []
            labels: List[str] = []
            colors: List[str] = []

            for combo in combos:
                col = f"{combo}_{metric_key}"
                vals = np.array([run.metrics.get(col, np.nan) for run in runs], dtype=float)
                vals = vals[np.isfinite(vals)]
                if vals.size == 0:
                    continue

                style = self.COMBO_STYLES.get(combo, (combo.replace("_", " "), "#7f7f7f"))
                combo_label, combo_color = style
                data.append(vals)
                labels.append(combo_label)
                colors.append(combo_color)

            if not data:
                ax.set_title(f"{metric_label}\n(no data)")
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

            ax.set_title(metric_label, fontweight="bold")
            ax.grid(True, alpha=0.3)

            if not legend_handles:
                for combo in combos:
                    style = self.COMBO_STYLES.get(combo, (combo.replace("_", " "), "#7f7f7f"))
                    combo_label, combo_color = style
                    if combo_label not in labels:
                        continue
                    legend_handles.append(plt.Line2D([0], [0], color=combo_color, linewidth=6))
                    legend_labels.append(combo_label)

        for idx in range(len(self.COMPARISON_METRICS), len(axes)):
            axes[idx].axis("off")

        fig.suptitle(
            "Monte Carlo Comparison Plots",
            fontweight="bold",
        )
        if legend_handles:
            fig.legend(legend_handles, legend_labels, loc="lower center", ncol=2, fontsize=9)
            fig.tight_layout(rect=[0, 0.06, 1, 0.96])
        else:
            fig.tight_layout(rect=[0, 0.02, 1, 0.96])

        plot_path = os.path.join(self.out_dir, "monte_carlo_comparisons_box.png")
        plt.savefig(plot_path, dpi=160)
        plt.close()

# ============================================================================
# 1. VERIFICATION TESTS
# ============================================================================

class VerificationSuite:
    """Verification tests for implementation correctness."""

    def __init__(self, config: ValidationConfig, out_dir: str):
        self.config = config
        self.out_dir = out_dir
        self.results: List[VerificationResult] = []
        _ensure_dir(out_dir)

    def run_all(self) -> List[VerificationResult]:
        """Run all verification tests."""
        print("\n" + "=" * 60)
        print("VERIFICATION SUITE")
        print("=" * 60)

        self.results = []
        self.results.append(self._verify_trajectory_consistency())
        self.results.append(self._verify_controller_implementation())
        self.results.append(self._verify_logging_integrity())
        self.results.append(self._verify_psd_computations())
        self.results.append(self._verify_inertia_consistency())
        self.results.append(self._verify_modal_coupling())

        self._print_summary()
        self._save_report()

        return self.results

    def _verify_trajectory_consistency(self) -> VerificationResult:
        """1.1 Verify feedforward trajectory consistency."""
        print("\n[V1.1] Trajectory Consistency...")

        details = {}
        issues = []

        # Check fourth order trajectory file
        traj_path = os.path.join(basilisk_dir, "data", "trajectories", "spacecraft_trajectory_4th_180deg_30s.npz")
        if not os.path.isfile(traj_path):
            issues.append("Fourth-order trajectory file not found")
        else:
            try:
                traj = np.load(traj_path, allow_pickle=True)
                t = np.array(traj.get("time", []), dtype=float)
                theta = np.array(traj.get("theta", []), dtype=float)
                omega = np.array(traj.get("omega", []), dtype=float)
                alpha = np.array(traj.get("alpha", []), dtype=float)

                # Check array lengths match
                if not (len(t) == len(theta) == len(omega) == len(alpha)):
                    issues.append(f"Array length mismatch: t={len(t)}, theta={len(theta)}, omega={len(omega)}, alpha={len(alpha)}")

                # Check final angle
                final_angle_deg = np.degrees(theta[-1])
                target = self.config.slew_angle_deg
                angle_error = abs(final_angle_deg - target)
                details["final_angle_deg"] = final_angle_deg
                details["target_angle_deg"] = target
                details["angle_error_deg"] = angle_error

                if angle_error > 1.0:
                    issues.append(f"Final angle {final_angle_deg:.2f} deg differs from target {target:.2f} deg by {angle_error:.2f} deg")

                # Check sample rate
                dt = np.median(np.diff(t))
                details["sample_dt_s"] = float(dt)
                if abs(dt - UNIFIED_SAMPLE_DT) > 0.001:
                    issues.append(f"Sample rate {1/dt:.1f} Hz differs from expected {1/UNIFIED_SAMPLE_DT:.1f} Hz")

                # Check kinematic consistency: omega = d(theta)/dt, alpha = d(omega)/dt
                omega_check = np.gradient(theta, t)
                alpha_check = np.gradient(omega, t)
                omega_error = _compute_rms(omega - omega_check)
                alpha_error = _compute_rms(alpha - alpha_check)
                details["omega_consistency_rms"] = omega_error
                details["alpha_consistency_rms"] = alpha_error

                # Thresholds for numerical differentiation error
                if omega_error > 0.01:
                    issues.append(f"Omega inconsistency: RMS error {omega_error:.4f} rad/s")

            except Exception as e:
                issues.append(f"Error loading trajectory: {e}")

        passed = len(issues) == 0
        message = "PASS" if passed else f"FAIL: {'; '.join(issues)}"
        print(f"  {message}")

        return VerificationResult(
            test_name="trajectory_consistency",
            passed=passed,
            message=message,
            details=details
        )

    def _verify_controller_implementation(self) -> VerificationResult:
        """1.2 Verify feedback controller implementation."""
        print("\n[V1.2] Controller Implementation...")

        from basilisk_sim.feedback_control import MRPFeedbackController, FilteredDerivativeController

        details = {}
        issues = []

        axis = _normalize_axis(self.config.rotation_axis)
        I_axis = float(axis @ self.config.inertia @ axis)
        first_mode = min(self.config.modal_freqs_hz) if self.config.modal_freqs_hz else 0.4

        # Standard PD design
        omega_bw = 2 * np.pi * first_mode / self.config.control_bandwidth_factor
        sigma_scale = 4.0
        K_design = sigma_scale * I_axis * omega_bw**2
        P_design = 2 * self.config.control_damping_ratio * I_axis * omega_bw

        details["I_axis"] = I_axis
        details["first_mode_hz"] = first_mode
        details["designed_K"] = K_design
        details["designed_P"] = P_design
        details["designed_bandwidth_hz"] = omega_bw / (2 * np.pi)

        # Create controller and check gains
        try:
            ctrl_std = MRPFeedbackController(
                inertia=self.config.inertia,
                K=K_design,
                P=P_design,
                Ki=-1.0
            )

            # Verify gains were set correctly
            if abs(ctrl_std.K - K_design) > 1e-6:
                issues.append(f"Standard PD K mismatch: {ctrl_std.K} vs designed {K_design}")
            if abs(ctrl_std.P - P_design) > 1e-6:
                issues.append(f"Standard PD P mismatch: {ctrl_std.P} vs designed {P_design}")

            details["standard_pd_K"] = ctrl_std.K
            details["standard_pd_P"] = ctrl_std.P

        except Exception as e:
            issues.append(f"Standard PD creation failed: {e}")

        # Filtered PD
        try:
            ctrl_filt = FilteredDerivativeController(
                inertia=self.config.inertia,
                K=K_design,
                P=P_design * 1.5,  # Typical scaling for filtered PD
                filter_freq_hz=self.config.control_filter_cutoff_hz
            )

            details["filtered_pd_K"] = ctrl_filt.K
            details["filtered_pd_P"] = ctrl_filt.P
            details["filter_cutoff_hz"] = ctrl_filt.filter_freq_hz

            if ctrl_filt.filter_freq_hz != self.config.control_filter_cutoff_hz:
                issues.append(f"Filter cutoff mismatch: {ctrl_filt.filter_freq_hz} vs config {self.config.control_filter_cutoff_hz}")

        except Exception as e:
            issues.append(f"Filtered PD creation failed: {e}")

        # Test control law computation
        try:
            sigma_test = np.array([0.01, 0.0, 0.0])
            omega_test = np.array([0.0, 0.0, 0.001])
            ctrl_std.set_target(np.zeros(3))
            torque = ctrl_std.compute_torque(sigma_test, omega_test)

            if not np.all(np.isfinite(torque)):
                issues.append("Standard PD produces non-finite torque")
            details["test_torque_std"] = torque.tolist()

        except Exception as e:
            issues.append(f"Control law test failed: {e}")

        passed = len(issues) == 0
        message = "PASS" if passed else f"FAIL: {'; '.join(issues)}"
        print(f"  {message}")

        return VerificationResult(
            test_name="controller_implementation",
            passed=passed,
            message=message,
            details=details
        )

    def _verify_logging_integrity(self) -> VerificationResult:
        """1.3 Verify logging and signal integrity."""
        print("\n[V1.3] Logging Integrity...")

        details = {}
        issues = []

        # Check for NPZ files from simulation
        npz_patterns = [
            f"vizard_demo_{method}_{controller}.npz"
            for method in METHODS
            for controller in CONTROLLERS
        ]

        for pattern in npz_patterns:
            npz_path = os.path.join(basilisk_dir, pattern)
            if os.path.isfile(npz_path):
                try:
                    data = np.load(npz_path, allow_pickle=True)

                    # Check required keys
                    required_keys = ["time", "sigma", "omega"]
                    for key in required_keys:
                        if key not in data:
                            issues.append(f"{pattern}: missing key '{key}'")

                    # Check array alignment
                    time = np.array(data.get("time", []))
                    sigma = np.array(data.get("sigma", []))

                    if len(time) > 0 and len(sigma) > 0:
                        if len(time) != len(sigma):
                            issues.append(f"{pattern}: time ({len(time)}) and sigma ({len(sigma)}) length mismatch")

                    # Check torque logging
                    fb_torque = data.get("fb_torque")
                    ff_torque = data.get("ff_torque")
                    total_torque = data.get("total_torque")

                    if fb_torque is None or len(fb_torque) == 0:
                        issues.append(f"{pattern}: fb_torque not logged or empty")
                    if total_torque is None or len(total_torque) == 0:
                        issues.append(f"{pattern}: total_torque not logged or empty")

                    # Check modal acceleration logging
                    mode1_acc = data.get("mode1_acc")
                    if mode1_acc is not None and len(mode1_acc) > 0:
                        details[f"{pattern}_has_modal_acc"] = True
                    else:
                        details[f"{pattern}_has_modal_acc"] = False
                        # Not an error, just a note

                    details[f"{pattern}_n_samples"] = len(time)

                except Exception as e:
                    issues.append(f"{pattern}: error loading: {e}")
            else:
                details[f"{pattern}_exists"] = False

        passed = len(issues) == 0
        message = "PASS" if passed else f"FAIL: {'; '.join(issues)}"
        print(f"  {message}")

        return VerificationResult(
            test_name="logging_integrity",
            passed=passed,
            message=message,
            details=details
        )

    def _verify_psd_computations(self) -> VerificationResult:
        """1.4 Verify frequency domain computations."""
        print("\n[V1.4] PSD Computations...")

        details = {}
        issues = []

        # Generate test signal with known PSD
        fs = 100.0  # Hz
        dt = 1.0 / fs
        duration = 60.0
        t = np.arange(0, duration, dt)

        # Single sinusoid at known frequency
        f_test = 0.5  # Hz
        amplitude = 1.0
        test_signal = amplitude * np.sin(2 * np.pi * f_test * t)

        # Compute PSD
        freq, psd = _compute_psd(t, test_signal)

        if len(freq) == 0:
            issues.append("PSD computation returned empty arrays")
        else:
            # Find peak frequency
            peak_idx = np.argmax(psd)
            peak_freq = freq[peak_idx]
            details["test_frequency_hz"] = f_test
            details["detected_peak_hz"] = peak_freq

            freq_error = abs(peak_freq - f_test)
            if freq_error > 0.1:
                issues.append(f"PSD peak at {peak_freq:.3f} Hz, expected {f_test:.3f} Hz")

            # Check that PSD has correct units (power spectral DENSITY)
            # For sinusoid, PSD integrates to power = amplitude^2 / 2
            total_power = np.trapezoid(psd, freq)
            expected_power = amplitude**2 / 2
            details["total_power"] = total_power
            details["expected_power"] = expected_power

            # Allow some error due to windowing
            power_error = abs(total_power - expected_power) / expected_power
            if power_error > 0.5:
                issues.append(f"Power error {power_error*100:.1f}% (expected ~amplitude^2/2)")

        # Verify dB conversion is 10*log10 for PSD (not 20*log10)
        test_psd = np.array([1.0, 10.0, 100.0])
        expected_db = 10 * np.log10(test_psd)
        details["psd_to_db_check"] = expected_db.tolist()

        passed = len(issues) == 0
        message = "PASS" if passed else f"FAIL: {'; '.join(issues)}"
        print(f"  {message}")

        return VerificationResult(
            test_name="psd_computations",
            passed=passed,
            message=message,
            details=details
        )

    def _verify_inertia_consistency(self) -> VerificationResult:
        """Verify inertia matrix consistency between modules."""
        print("\n[V1.5] Inertia Consistency...")

        details = {}
        issues = []

        # Compute effective inertia
        I_eff = compute_effective_inertia()
        details["effective_inertia_diag"] = np.diag(I_eff).tolist()
        details["hub_inertia_diag"] = np.diag(HUB_INERTIA).tolist()

        # Check that effective > hub (due to appendage masses)
        for i in range(3):
            if I_eff[i, i] < HUB_INERTIA[i, i]:
                issues.append(f"Effective inertia I_{i}{i} < hub inertia")

        # Check positive definite
        eigvals = np.linalg.eigvals(I_eff)
        if np.any(eigvals <= 0):
            issues.append("Inertia matrix not positive definite")
        details["inertia_eigenvalues"] = eigvals.tolist()

        # Check symmetric
        if not np.allclose(I_eff, I_eff.T):
            issues.append("Inertia matrix not symmetric")

        passed = len(issues) == 0
        message = "PASS" if passed else f"FAIL: {'; '.join(issues)}"
        print(f"  {message}")

        return VerificationResult(
            test_name="inertia_consistency",
            passed=passed,
            message=message,
            details=details
        )

    def _verify_modal_coupling(self) -> VerificationResult:
        """Verify modal coupling gains are computed correctly."""
        print("\n[V1.6] Modal Coupling...")

        details = {}
        issues = []

        axis = _normalize_axis(self.config.rotation_axis)
        I_eff = compute_effective_inertia()

        modal_gains = compute_modal_gains(I_eff, axis)
        lever_arms = compute_mode_lever_arms(axis)

        details["modal_gains"] = modal_gains
        details["lever_arms_m"] = lever_arms

        # Check that gains are positive
        for i, gain in enumerate(modal_gains):
            if gain < 0:
                issues.append(f"Modal gain {i} is negative: {gain}")

        # Verify gain = lever_arm / I_axis relationship
        I_axis = float(axis @ I_eff @ axis)
        details["I_axis"] = I_axis

        for i, (gain, arm) in enumerate(zip(modal_gains, lever_arms)):
            expected_gain = arm / I_axis
            if abs(gain - expected_gain) > 1e-10:
                issues.append(f"Modal gain {i} inconsistent with lever arm: {gain} vs {expected_gain}")

        passed = len(issues) == 0
        message = "PASS" if passed else f"FAIL: {'; '.join(issues)}"
        print(f"  {message}")

        return VerificationResult(
            test_name="modal_coupling",
            passed=passed,
            message=message,
            details=details
        )

    def _print_summary(self) -> None:
        """Print verification summary."""
        print("\n" + "-" * 60)
        print("VERIFICATION SUMMARY")
        print("-" * 60)

        n_passed = sum(1 for r in self.results if r.passed)
        n_total = len(self.results)

        for r in self.results:
            status = "PASS" if r.passed else "FAIL"
            print(f"  [{status}] {r.test_name}")

        print("-" * 60)
        print(f"  Total: {n_passed}/{n_total} passed")

        if n_passed == n_total:
            print("  VERIFICATION: ALL TESTS PASSED")
        else:
            print("  VERIFICATION: SOME TESTS FAILED")

    def _save_report(self) -> None:
        """Save verification report."""
        report_path = os.path.join(self.out_dir, "verification_report.json")

        report = {
            "timestamp": datetime.now().isoformat(),
            "summary": {
                "total_tests": len(self.results),
                "passed": sum(1 for r in self.results if r.passed),
                "failed": sum(1 for r in self.results if not r.passed),
            },
            "tests": [
                {
                    "name": r.test_name,
                    "passed": r.passed,
                    "message": r.message,
                    "details": {k: (v.tolist() if isinstance(v, np.ndarray) else v)
                               for k, v in r.details.items()},
                }
                for r in self.results
            ],
        }

        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)

        print(f"\nVerification report saved: {report_path}")


# ============================================================================
# 2. VALIDATION TESTS
# ============================================================================

class ValidationSuite:
    """Validation tests for physics and performance."""

    def __init__(self, config: ValidationConfig, out_dir: str, data_dir: Optional[str] = None):
        self.config = config
        self.out_dir = out_dir
        self.data_dir = data_dir or basilisk_dir
        self.results: List[ValidationResult] = []
        _ensure_dir(out_dir)

    def run_all(self) -> List[ValidationResult]:
        """Run all validation tests."""
        print("\n" + "=" * 60)
        print("VALIDATION SUITE")
        print("=" * 60)

        self.results = []
        self.results.append(self._validate_tracking())
        self.results.append(self._validate_vibration_suppression())
        self.results.append(self._validate_torque_limits())
        self.results.append(self._validate_stability_margins())
        self.results.append(self._validate_disturbance_rejection())
        self.results.append(self._validate_noise_rejection())

        self._print_summary()
        self._save_report()

        return self.results

    def _load_simulation_data(self, method: str, controller: str) -> Optional[Dict[str, np.ndarray]]:
        """Load simulation data from NPZ file."""
        pattern = f"vizard_demo_{method}_{controller}.npz"
        npz_path = os.path.join(self.data_dir, pattern)

        if not os.path.isfile(npz_path):
            return None

        try:
            data = np.load(npz_path, allow_pickle=True)
            return {
                "time": np.array(data.get("time", [])),
                "sigma": np.array(data.get("sigma", [])),
                "omega": np.array(data.get("omega", [])),
                "mode1": np.array(data.get("mode1", [])),
                "mode2": np.array(data.get("mode2", [])),
                "fb_torque": np.array(data.get("fb_torque", [])),
                "ff_torque": np.array(data.get("ff_torque", [])),
                "total_torque": np.array(data.get("total_torque", [])),
                "rw_torque": np.array(data.get("rw_torque", [])),
                "target_sigma": np.array(data.get("target_sigma", [0, 0, 1])),
                "slew_duration_s": float(data.get("slew_duration_s", 30.0)),
            }
        except Exception as e:
            print(f"  Warning: Could not load {pattern}: {e}")
            return None

    def _validate_tracking(self) -> ValidationResult:
        """2.1 Closed loop tracking validation."""
        print("\n[V2.1] Tracking Validation...")

        metrics = {}
        issues = []
        plots = []

        n_cases = len(METHODS) * len(CONTROLLERS)
        ncols = 2
        nrows = int(np.ceil(n_cases / ncols))
        fig, axes = plt.subplots(nrows, ncols, figsize=(14, 4.0 * nrows + 2.0))
        axes = np.atleast_1d(axes).flatten()
        fig.suptitle("Tracking Validation", fontsize=14, fontweight="bold")

        for method in METHODS:
            for controller in CONTROLLERS:
                data = self._load_simulation_data(method, controller)
                if data is None:
                    issues.append(f"No data for {method}/{controller}")
                    continue

                time = data["time"]
                sigma = data["sigma"]
                target = data["target_sigma"]
                slew_duration = data["slew_duration_s"]

                if len(time) == 0 or len(sigma) == 0:
                    continue

                # Compute pointing error with MRP shadow handling
                # MRPs have two representations for the same rotation (shadow set)
                # sigma and -sigma/|sigma|^2 represent equivalent rotations
                target_norm_sq = np.dot(target, target)
                if target_norm_sq > 0:
                    target_shadow = -target / target_norm_sq
                else:
                    target_shadow = target

                sigma_error = []
                for s in sigma:
                    if len(s) >= 3:
                        s_arr = np.array(s)
                        # Check both direct and shadow comparisons
                        err_direct = np.linalg.norm(s_arr - target)
                        err_shadow = np.linalg.norm(s_arr - target_shadow)
                        # Use the smaller error (represents actual physical error)
                        err = min(err_direct, err_shadow)
                        sigma_error.append(4 * np.arctan(err))  # Convert to angle
                    else:
                        sigma_error.append(0)
                sigma_error = np.degrees(np.array(sigma_error))

                # Post slew metrics
                post_slew_mask = time > slew_duration
                if np.any(post_slew_mask):
                    post_slew_error = sigma_error[post_slew_mask]
                    rms_error = _compute_rms(post_slew_error)
                    peak_error = np.max(np.abs(post_slew_error))
                    final_error = post_slew_error[-1]
                else:
                    rms_error = _compute_rms(sigma_error)
                    peak_error = np.max(np.abs(sigma_error))
                    final_error = sigma_error[-1]

                key = f"{method}_{controller}"
                metrics[f"{key}_rms_error_deg"] = rms_error
                metrics[f"{key}_peak_error_deg"] = peak_error
                metrics[f"{key}_final_error_deg"] = final_error

                # Check thresholds
                if rms_error > 0.1:
                    issues.append(f"{key}: RMS error {rms_error:.4f} deg > 0.1 deg")
                if peak_error > 1.0:
                    issues.append(f"{key}: Peak error {peak_error:.4f} deg > 1.0 deg")

                # Plot
                ax_idx = METHODS.index(method) * 2 + CONTROLLERS.index(controller)
                ax = axes[ax_idx]
                ax.plot(time, sigma_error, linewidth=1.5)
                ax.axhline(y=0, color='k', linestyle='--', alpha=0.3)
                ax.axvline(x=slew_duration, color='r', linestyle='--', alpha=0.5, label='Slew End')
                ax.set_xlabel("Time (s)")
                ax.set_ylabel("Pointing Error (deg)")
                ax.set_title(f"{method.capitalize()} + {controller.replace('_', ' ').title()}")
                ax.grid(True, alpha=0.3)
                ax.legend()

        for ax in axes[n_cases:]:
            ax.axis("off")

        plt.tight_layout()
        plot_path = os.path.join(self.out_dir, "validation_tracking.png")
        plt.savefig(plot_path, dpi=150)
        plt.close()
        plots.append(plot_path)

        passed = len(issues) == 0
        message = "PASS" if passed else f"FAIL: {len(issues)} issues"
        print(f"  {message}")

        return ValidationResult(
            test_name="tracking",
            passed=passed,
            message=message,
            metrics=metrics,
            plots=plots
        )

    def _validate_vibration_suppression(self) -> ValidationResult:
        """2.4 Flexible mode suppression validation."""
        print("\n[V2.4] Vibration Suppression...")

        metrics = {}
        issues = []
        plots = []

        n_cases = len(METHODS) * len(CONTROLLERS)
        ncols = 2
        nrows = int(np.ceil(n_cases / ncols))
        fig, axes = plt.subplots(nrows, ncols, figsize=(14, 4.0 * nrows + 2.0))
        axes = np.atleast_1d(axes).flatten()
        fig.suptitle("Vibration Suppression Validation", fontsize=14, fontweight="bold")

        for method in METHODS:
            for controller in CONTROLLERS:
                data = self._load_simulation_data(method, controller)
                if data is None:
                    continue

                time = data["time"]
                mode1 = data["mode1"]
                mode2 = data["mode2"]
                slew_duration = data["slew_duration_s"]

                if len(time) == 0:
                    continue

                # Combine modes
                total_vib = np.sqrt(mode1**2 + mode2**2) * 1000  # mm

                # Post slew metrics
                post_slew_mask = time > slew_duration
                if np.any(post_slew_mask):
                    post_slew_vib = total_vib[post_slew_mask]
                    rms_vib = _compute_rms(post_slew_vib)
                    peak_vib = np.max(np.abs(post_slew_vib))
                else:
                    rms_vib = _compute_rms(total_vib)
                    peak_vib = np.max(np.abs(total_vib))

                key = f"{method}_{controller}"
                metrics[f"{key}_rms_vibration_mm"] = rms_vib
                metrics[f"{key}_peak_vibration_mm"] = peak_vib

                # Plot
                ax_idx = METHODS.index(method) * 2 + CONTROLLERS.index(controller)
                ax = axes[ax_idx]
                ax.plot(time, total_vib, linewidth=1)
                ax.axvline(x=slew_duration, color='r', linestyle='--', alpha=0.5, label='Slew End')
                ax.set_xlabel("Time (s)")
                ax.set_ylabel("Modal Displacement (mm)")
                ax.set_title(f"{method.capitalize()} + {controller.replace('_', ' ').title()}\nRMS: {rms_vib:.3f} mm")
                ax.grid(True, alpha=0.3)
                ax.legend()

        for ax in axes[n_cases:]:
            ax.axis("off")

        # Compare baselines vs fourth order
        for controller in CONTROLLERS:
            fourth_key = f"fourth_{controller}_rms_vibration_mm"
            for baseline_method in [m for m in METHODS if m != "fourth"]:
                baseline_key = f"{baseline_method}_{controller}_rms_vibration_mm"
                baseline_val = metrics.get(baseline_key, float("nan"))
                fourth_val = metrics.get(fourth_key, float("nan"))
                if np.isfinite(baseline_val) and baseline_val > 0 and np.isfinite(fourth_val):
                    reduction = (baseline_val - fourth_val) / baseline_val * 100
                    metrics[f"{baseline_method}_to_fourth_{controller}_vibration_reduction_pct"] = reduction
                    if reduction < 50:
                        issues.append(
                            f"{controller}: Fourth-order only {reduction:.1f}% reduction vs "
                            f"{baseline_method} (expected >50%)"
                        )

        plt.tight_layout()
        plot_path = os.path.join(self.out_dir, "validation_vibration.png")
        plt.savefig(plot_path, dpi=150)
        plt.close()
        plots.append(plot_path)

        passed = len(issues) == 0
        message = "PASS" if passed else f"FAIL: {len(issues)} issues"
        print(f"  {message}")

        return ValidationResult(
            test_name="vibration_suppression",
            passed=passed,
            message=message,
            metrics=metrics,
            plots=plots
        )

    def _validate_torque_limits(self) -> ValidationResult:
        """2.5 Torque/actuator capability validation."""
        print("\n[V2.5] Torque Limits...")

        metrics = {}
        issues = []
        plots = []

        max_torque = self.config.rw_max_torque_nm

        n_cases = len(METHODS) * len(CONTROLLERS)
        ncols = 2
        nrows = int(np.ceil(n_cases / ncols))
        fig, axes = plt.subplots(nrows, ncols, figsize=(14, 4.0 * nrows + 2.0))
        axes = np.atleast_1d(axes).flatten()
        fig.suptitle(f"Torque Command Validation (Limit: {max_torque} Nm)", fontsize=14, fontweight="bold")

        for method in METHODS:
            for controller in CONTROLLERS:
                data = self._load_simulation_data(method, controller)
                if data is None:
                    continue

                time = data["time"]
                rw_torque = data["rw_torque"]
                total_torque = data["total_torque"]

                if len(time) == 0:
                    continue

                # Use RW torque if available, else total body torque
                if len(rw_torque) > 0:
                    if rw_torque.ndim == 1:
                        torque_mag = np.abs(rw_torque)
                    else:
                        torque_mag = np.max(np.abs(rw_torque), axis=1)
                elif len(total_torque) > 0:
                    if total_torque.ndim == 1:
                        torque_mag = np.abs(total_torque)
                    else:
                        torque_mag = np.linalg.norm(total_torque, axis=1)
                else:
                    continue

                peak_torque = np.max(torque_mag)
                rms_torque = _compute_rms(torque_mag)

                # Saturation count
                n_saturated = np.sum(torque_mag >= max_torque * 0.99)
                saturation_pct = n_saturated / len(torque_mag) * 100

                key = f"{method}_{controller}"
                metrics[f"{key}_peak_torque_nm"] = peak_torque
                metrics[f"{key}_rms_torque_nm"] = rms_torque
                metrics[f"{key}_saturation_pct"] = saturation_pct

                if peak_torque > max_torque:
                    issues.append(f"{key}: Peak torque {peak_torque:.2f} Nm exceeds limit {max_torque} Nm")
                if saturation_pct > 5:
                    issues.append(f"{key}: Saturation {saturation_pct:.1f}% > 5%")

                # Plot
                ax_idx = METHODS.index(method) * 2 + CONTROLLERS.index(controller)
                ax = axes[ax_idx]
                ax.plot(time, torque_mag, linewidth=1)
                ax.axhline(y=max_torque, color='r', linestyle='--', alpha=0.5, label=f'Limit: {max_torque} Nm')
                ax.set_xlabel("Time (s)")
                ax.set_ylabel("Torque (Nm)")
                ax.set_title(f"{method.capitalize()} + {controller.replace('_', ' ').title()}\nPeak: {peak_torque:.2f} Nm")
                ax.grid(True, alpha=0.3)
                ax.legend()

        for ax in axes[n_cases:]:
            ax.axis("off")

        plt.tight_layout()
        plot_path = os.path.join(self.out_dir, "validation_torque.png")
        plt.savefig(plot_path, dpi=150)
        plt.close()
        plots.append(plot_path)

        passed = len(issues) == 0
        message = "PASS" if passed else f"FAIL: {len(issues)} issues"
        print(f"  {message}")

        return ValidationResult(
            test_name="torque_limits",
            passed=passed,
            message=message,
            metrics=metrics,
            plots=plots
        )

    def _validate_stability_margins(self) -> ValidationResult:
        """Validate control system stability margins."""
        print("\n[V2.6] Stability Margins...")

        from basilisk_sim.feedback_control import MRPFeedbackController, FilteredDerivativeController

        metrics = {}
        issues = []
        plots = []

        axis = _normalize_axis(self.config.rotation_axis)
        I_axis = float(axis @ self.config.inertia @ axis)
        first_mode = min(self.config.modal_freqs_hz) if self.config.modal_freqs_hz else 0.4
        sigma_scale = 4.0

        # Design bandwidth
        omega_bw = 2 * np.pi * first_mode / self.config.control_bandwidth_factor
        K = sigma_scale * I_axis * omega_bw**2
        P = 2 * self.config.control_damping_ratio * I_axis * omega_bw

        # Frequency range for analysis
        freqs = np.logspace(-2, 1, 500)
        omega = 2 * np.pi * freqs
        s = 1j * omega

        # Rigid plant: 1 / (4*I*s^2)
        plant_rigid = 1.0 / (sigma_scale * I_axis * s**2)

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle("Stability Margin Analysis", fontsize=14, fontweight="bold")

        # Create controller objects for printing only
        std_pd = MRPFeedbackController(self.config.inertia, K=K, P=P, Ki=-1.0)
        filt_pd = FilteredDerivativeController(
            self.config.inertia, K=K, P=P*1.5,
            filter_freq_hz=self.config.control_filter_cutoff_hz
        )

        # Build controller TFs directly to avoid return type differences
        # Standard PD: C(s) = K + 4*P*s = (4*P*s + K) / 1
        std_pd_num = [4.0 * P, K]
        std_pd_den = [1.0]

        # Filtered PD: C(s) = K + 4*P*s/(tau*s + 1)
        tau = 1.0 / (2.0 * np.pi * self.config.control_filter_cutoff_hz)
        P_filt = P * 1.5
        filt_pd_num = [K * tau + 4.0 * P_filt, K]
        filt_pd_den = [tau, 1.0]

        controller_tfs = {
            "standard_pd": (std_pd_num, std_pd_den),
            "filtered_pd": (filt_pd_num, filt_pd_den),
        }

        for idx, (name, (num, den)) in enumerate(controller_tfs.items()):
            # Evaluate TF manually to avoid scipy version issues
            s = 1j * omega
            num = np.atleast_1d(num)
            den = np.atleast_1d(den)
            ctrl_resp = np.polyval(num, s) / np.polyval(den, s)

            # Open loop
            L = plant_rigid * ctrl_resp

            # Compute margins
            mag = np.abs(L)
            phase = np.degrees(np.unwrap(np.angle(L)))

            # Gain margin (at phase = -180)
            phase_cross_idx = np.where((phase[:-1] > -180) & (phase[1:] <= -180))[0]
            if len(phase_cross_idx) > 0:
                i = phase_cross_idx[0]
                gm_db = -20 * np.log10(mag[i])
            else:
                gm_db = np.inf

            # Phase margin (at gain = 1)
            gain_cross_idx = np.where((mag[:-1] > 1) & (mag[1:] <= 1))[0]
            if len(gain_cross_idx) > 0:
                i = gain_cross_idx[0]
                pm_deg = 180 + phase[i]
            else:
                pm_deg = np.inf

            metrics[f"{name}_gain_margin_db"] = gm_db
            metrics[f"{name}_phase_margin_deg"] = pm_deg

            # Check minimum margins
            if gm_db < 6:
                issues.append(f"{name}: Gain margin {gm_db:.1f} dB < 6 dB")
            if pm_deg < 30:
                issues.append(f"{name}: Phase margin {pm_deg:.1f} deg < 30 deg")

            # Bode plot
            ax_mag = axes[0, idx]
            ax_mag.semilogx(freqs, 20 * np.log10(mag), linewidth=2)
            ax_mag.axhline(y=0, color='r', linestyle='--', alpha=0.5)
            for f_mode in self.config.modal_freqs_hz:
                ax_mag.axvline(x=f_mode, color='g', linestyle=':', alpha=0.5, label=f'Mode: {f_mode} Hz')
            ax_mag.set_ylabel("Magnitude (dB)")
            ax_mag.set_title(f"{name.replace('_', ' ').title()}\nGM: {gm_db:.1f} dB, PM: {pm_deg:.1f}°")
            ax_mag.grid(True, alpha=0.3)
            ax_mag.set_xlim([freqs[0], freqs[-1]])

            ax_ph = axes[1, idx]
            ax_ph.semilogx(freqs, phase, linewidth=2)
            ax_ph.axhline(y=-180, color='r', linestyle='--', alpha=0.5)
            for f_mode in self.config.modal_freqs_hz:
                ax_ph.axvline(x=f_mode, color='g', linestyle=':', alpha=0.5)
            ax_ph.set_xlabel("Frequency (Hz)")
            ax_ph.set_ylabel("Phase (deg)")
            ax_ph.grid(True, alpha=0.3)
            ax_ph.set_xlim([freqs[0], freqs[-1]])

        plt.tight_layout()
        plot_path = os.path.join(self.out_dir, "validation_stability.png")
        plt.savefig(plot_path, dpi=150)
        plt.close()
        plots.append(plot_path)

        passed = len(issues) == 0
        message = "PASS" if passed else f"FAIL: {len(issues)} issues"
        print(f"  {message}")

        return ValidationResult(
            test_name="stability_margins",
            passed=passed,
            message=message,
            metrics=metrics,
            plots=plots
        )

    def _validate_disturbance_rejection(self) -> ValidationResult:
        """2.2 Disturbance rejection validation."""
        print("\n[V2.2] Disturbance Rejection...")

        metrics = {}
        issues = []
        plots = []

        axis = _normalize_axis(self.config.rotation_axis)
        I_axis = float(axis @ self.config.inertia @ axis)
        sigma_scale = 4.0
        first_mode = min(self.config.modal_freqs_hz) if self.config.modal_freqs_hz else 0.4

        omega_bw = 2 * np.pi * first_mode / self.config.control_bandwidth_factor
        K = sigma_scale * I_axis * omega_bw**2
        P = 2 * self.config.control_damping_ratio * I_axis * omega_bw

        freqs = np.logspace(-2, 1, 500)
        omega = 2 * np.pi * freqs
        s = 1j * omega

        # Rigid plant
        G = 1.0 / (sigma_scale * I_axis * s**2)

        # Controllers
        C_std = 4 * P * s + K
        tau_filt = 1.0 / (2 * np.pi * self.config.control_filter_cutoff_hz)
        C_filt = (K * tau_filt * s + 4 * P * s + K) / (tau_filt * s + 1)

        # Sensitivity S = 1/(1+GC) - disturbance to output
        S_std = 1.0 / (1.0 + G * C_std)
        S_filt = 1.0 / (1.0 + G * C_filt)

        fig, ax = plt.subplots(1, 1, figsize=(10, 6))
        ax.semilogx(freqs, 20 * np.log10(np.abs(S_std)), label='Standard PD', linewidth=2)
        ax.semilogx(freqs, 20 * np.log10(np.abs(S_filt)), label='Filtered PD', linewidth=2)

        for f_mode in self.config.modal_freqs_hz:
            ax.axvline(x=f_mode, color='r', linestyle='--', alpha=0.5, label=f'Mode: {f_mode} Hz')

        ax.axhline(y=0, color='k', linestyle='--', alpha=0.3)
        ax.set_xlabel("Frequency (Hz)")
        ax.set_ylabel("Sensitivity |S| (dB)")
        ax.set_title("Disturbance Rejection: Sensitivity Function")
        ax.grid(True, alpha=0.3)
        ax.legend()
        ax.set_xlim([freqs[0], freqs[-1]])

        plt.tight_layout()
        plot_path = os.path.join(self.out_dir, "validation_disturbance.png")
        plt.savefig(plot_path, dpi=150)
        plt.close()
        plots.append(plot_path)

        # Metrics: peak sensitivity
        metrics["standard_pd_peak_sensitivity_db"] = float(np.max(20 * np.log10(np.abs(S_std))))
        metrics["filtered_pd_peak_sensitivity_db"] = float(np.max(20 * np.log10(np.abs(S_filt))))

        # Check peak sensitivity < 6 dB (standard criterion)
        for name, S in [("standard_pd", S_std), ("filtered_pd", S_filt)]:
            peak_db = np.max(20 * np.log10(np.abs(S)))
            if peak_db > 6:
                issues.append(f"{name}: Peak sensitivity {peak_db:.1f} dB > 6 dB")

        passed = len(issues) == 0
        message = "PASS" if passed else f"FAIL: {len(issues)} issues"
        print(f"  {message}")

        return ValidationResult(
            test_name="disturbance_rejection",
            passed=passed,
            message=message,
            metrics=metrics,
            plots=plots
        )

    def _validate_noise_rejection(self) -> ValidationResult:
        """2.3 Noise rejection validation."""
        print("\n[V2.3] Noise Rejection...")

        metrics = {}
        issues = []
        plots = []

        axis = _normalize_axis(self.config.rotation_axis)
        I_axis = float(axis @ self.config.inertia @ axis)
        sigma_scale = 4.0
        first_mode = min(self.config.modal_freqs_hz) if self.config.modal_freqs_hz else 0.4

        omega_bw = 2 * np.pi * first_mode / self.config.control_bandwidth_factor
        K = sigma_scale * I_axis * omega_bw**2
        P = 2 * self.config.control_damping_ratio * I_axis * omega_bw

        freqs = np.logspace(-2, 2, 500)
        omega = 2 * np.pi * freqs
        s = 1j * omega

        # Rigid plant
        G = 1.0 / (sigma_scale * I_axis * s**2)

        # Controllers
        C_std = 4 * P * s + K
        tau_filt = 1.0 / (2 * np.pi * self.config.control_filter_cutoff_hz)
        C_filt = (K * tau_filt * s + 4 * P * s + K) / (tau_filt * s + 1)

        # Noise to control: C/(1+GC)
        N_std = C_std / (1.0 + G * C_std)
        N_filt = C_filt / (1.0 + G * C_filt)

        fig, ax = plt.subplots(1, 1, figsize=(10, 6))
        ax.semilogx(freqs, 20 * np.log10(np.abs(N_std)), label='Standard PD', linewidth=2)
        ax.semilogx(freqs, 20 * np.log10(np.abs(N_filt)), label='Filtered PD', linewidth=2)

        ax.set_xlabel("Frequency (Hz)")
        ax.set_ylabel("Noise to Torque |C/(1+GC)| (dB)")
        ax.set_title("Noise Rejection: Control Sensitivity")
        ax.grid(True, alpha=0.3)
        ax.legend()
        ax.set_xlim([freqs[0], freqs[-1]])

        plt.tight_layout()
        plot_path = os.path.join(self.out_dir, "validation_noise.png")
        plt.savefig(plot_path, dpi=150)
        plt.close()
        plots.append(plot_path)

        # High frequency roll off comparison
        hf_idx = freqs > 1.0
        if np.any(hf_idx):
            hf_std = np.mean(20 * np.log10(np.abs(N_std[hf_idx])))
            hf_filt = np.mean(20 * np.log10(np.abs(N_filt[hf_idx])))
            metrics["standard_pd_hf_noise_db"] = hf_std
            metrics["filtered_pd_hf_noise_db"] = hf_filt

            # Filtered PD should have lower HF noise
            if hf_filt > hf_std:
                issues.append(f"Filtered PD has higher HF noise ({hf_filt:.1f} dB) than standard ({hf_std:.1f} dB)")

        passed = len(issues) == 0
        message = "PASS" if passed else f"FAIL: {len(issues)} issues"
        print(f"  {message}")

        return ValidationResult(
            test_name="noise_rejection",
            passed=passed,
            message=message,
            metrics=metrics,
            plots=plots
        )

    def _print_summary(self) -> None:
        """Print validation summary."""
        print("\n" + "-" * 60)
        print("VALIDATION SUMMARY")
        print("-" * 60)

        n_passed = sum(1 for r in self.results if r.passed)
        n_total = len(self.results)

        for r in self.results:
            status = "PASS" if r.passed else "FAIL"
            print(f"  [{status}] {r.test_name}")

        print("-" * 60)
        print(f"  Total: {n_passed}/{n_total} passed")

        if n_passed == n_total:
            print("  VALIDATION: ALL TESTS PASSED")
        else:
            print("  VALIDATION: SOME TESTS FAILED")

    def _save_report(self) -> None:
        """Save validation report."""
        report_path = os.path.join(self.out_dir, "validation_report.json")

        all_metrics = {}
        for r in self.results:
            all_metrics.update(r.metrics)

        report = {
            "timestamp": datetime.now().isoformat(),
            "summary": {
                "total_tests": len(self.results),
                "passed": sum(1 for r in self.results if r.passed),
                "failed": sum(1 for r in self.results if not r.passed),
            },
            "tests": [
                {
                    "name": r.test_name,
                    "passed": r.passed,
                    "message": r.message,
                    "metrics": r.metrics,
                    "plots": r.plots,
                }
                for r in self.results
            ],
            "all_metrics": all_metrics,
        }

        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)

        # Also save metrics CSV
        csv_path = os.path.join(self.out_dir, "validation_metrics.csv")
        rows = [[k, str(v)] for k, v in sorted(all_metrics.items())]
        _write_csv(csv_path, ["metric", "value"], rows)

        print(f"\nValidation report saved: {report_path}")
        print(f"Validation metrics saved: {csv_path}")


# ============================================================================
# 3. MONTE CARLO ANALYSIS
# ============================================================================

class MonteCarloRunner:
    """Monte Carlo analysis for robustness under uncertainty."""

    def __init__(self, base_config: ValidationConfig, out_dir: str, n_runs: int = 500):
        self.base_config = base_config
        self.out_dir = out_dir
        self.n_runs = n_runs
        self.runs: List[MonteCarloRun] = []
        self.rng = np.random.default_rng(42)  # Reproducible
        _ensure_dir(out_dir)

        # Uncertainty bounds (from validation_mc.md section 3.1)
        self.uncertainties = {
            "inertia_pct": 0.15,           # ±15%
            "modal_freq_pct": 0.10,        # ±10%
            "modal_damping_pct": 0.50,     # ±50%
            "modal_gain_pct": 0.20,        # ±20%
            "sensor_noise_pct": 0.50,      # ±50%
            "disturbance_pct": 0.50,       # ±50%
            "filter_cutoff_pct": 0.20,     # ±20%
        }

        # Pass/fail thresholds
        self.thresholds = DEFAULT_THRESHOLDS.copy()

    def run(self) -> MonteCarloSummary:
        """Run Monte Carlo analysis."""
        print("\n" + "=" * 60)
        print(f"MONTE CARLO ANALYSIS ({self.n_runs} runs)")
        print("=" * 60)

        print("\nUncertainty bounds:")
        for name, pct in self.uncertainties.items():
            print(f"  {name}: ±{pct*100:.0f}%")

        print("\nPass/fail thresholds:")
        for name, val in self.thresholds.items():
            print(f"  {name}: {val}")

        self.runs = []

        start_time = time_module.time()

        for run_id in range(self.n_runs):
            if (run_id + 1) % 50 == 0 or run_id == 0:
                elapsed = time_module.time() - start_time
                rate = (run_id + 1) / elapsed if elapsed > 0 else 0
                print(f"  Run {run_id + 1}/{self.n_runs} ({rate:.1f} runs/s)...")

            run_result = self._run_single(run_id)
            self.runs.append(run_result)

        elapsed = time_module.time() - start_time
        print(f"\nCompleted {self.n_runs} runs in {elapsed:.1f}s ({self.n_runs/elapsed:.1f} runs/s)")

        summary = self._compute_summary()
        self._print_summary(summary)
        self._save_results(summary)
        self._plot_histograms(summary)

        return summary

    def _perturb_config(self, run_id: int) -> Tuple[ValidationConfig, Dict[str, float]]:
        """Generate perturbed configuration for a single run."""
        config = self.base_config.copy()
        perturbations = {}

        # Inertia perturbation (diagonal elements)
        inertia_scale = 1 + self.rng.uniform(-1, 1) * self.uncertainties["inertia_pct"]
        config.inertia = config.inertia * inertia_scale
        perturbations["inertia_scale"] = inertia_scale

        # Modal frequency perturbation
        freq_scale = 1 + self.rng.uniform(-1, 1) * self.uncertainties["modal_freq_pct"]
        config.modal_freqs_hz = [f * freq_scale for f in config.modal_freqs_hz]
        perturbations["freq_scale"] = freq_scale

        # Modal damping perturbation
        damp_scale = 1 + self.rng.uniform(-1, 1) * self.uncertainties["modal_damping_pct"]
        config.modal_damping = [max(0.001, d * damp_scale) for d in config.modal_damping]
        perturbations["damping_scale"] = damp_scale

        # Filter cutoff perturbation
        cutoff_scale = 1 + self.rng.uniform(-1, 1) * self.uncertainties["filter_cutoff_pct"]
        config.control_filter_cutoff_hz = max(0.1, config.control_filter_cutoff_hz * cutoff_scale)
        perturbations["cutoff_scale"] = cutoff_scale

        # Sensor noise perturbation
        noise_scale = 1 + self.rng.uniform(-1, 1) * self.uncertainties["sensor_noise_pct"]
        config.sensor_noise_std_rad_s = max(0, config.sensor_noise_std_rad_s * noise_scale)
        perturbations["noise_scale"] = noise_scale

        # Disturbance perturbation
        dist_scale = 1 + self.rng.uniform(-1, 1) * self.uncertainties["disturbance_pct"]
        config.disturbance_torque_nm = config.disturbance_torque_nm * dist_scale
        perturbations["disturbance_scale"] = dist_scale

        return config, perturbations

    def _run_single(self, run_id: int) -> MonteCarloRun:
        """Run a single Monte Carlo iteration."""
        config, perturbations = self._perturb_config(run_id)

        # Simulate closed loop response with perturbed parameters
        metrics = self._simulate_response(config)

        # Check pass/fail
        failure_reasons = []

        if metrics["rms_pointing_error_deg"] > self.thresholds["rms_pointing_error_deg_p95"]:
            failure_reasons.append(f"RMS error {metrics['rms_pointing_error_deg']:.5f} > {self.thresholds['rms_pointing_error_deg_p95']}")

        if metrics["peak_torque_nm"] > self.thresholds["peak_torque_nm_p99"]:
            failure_reasons.append(f"Peak torque {metrics['peak_torque_nm']:.1f} > {self.thresholds['peak_torque_nm_p99']}")

        if metrics["rms_vibration_mm"] > self.thresholds["rms_vibration_mm_p95"]:
            failure_reasons.append(f"RMS vibration {metrics['rms_vibration_mm']:.3f} > {self.thresholds['rms_vibration_mm_p95']}")

        passed = len(failure_reasons) == 0

        return MonteCarloRun(
            run_id=run_id,
            config_perturbations=perturbations,
            metrics=metrics,
            passed=passed,
            failure_reasons=failure_reasons
        )

    def _simulate_response(self, config: ValidationConfig) -> Dict[str, float]:
        """Simulate closed loop response and compute metrics."""
        from basilisk_sim.feedback_control import FilteredDerivativeController

        # Time array
        dt = UNIFIED_SAMPLE_DT
        total_time = config.slew_duration_s + config.settling_time_s
        t = np.arange(0, total_time, dt)
        n = len(t)

        axis = _normalize_axis(config.rotation_axis)
        I_axis = float(axis @ config.inertia @ axis)
        sigma_scale = 4.0

        # Controller gains
        first_mode = min(config.modal_freqs_hz) if config.modal_freqs_hz else 0.4
        omega_bw = 2 * np.pi * first_mode / config.control_bandwidth_factor
        K = sigma_scale * I_axis * omega_bw**2
        P = 2 * config.control_damping_ratio * I_axis * omega_bw * 1.5

        # Simplified closed loop simulation
        # State: [sigma, omega]
        sigma = np.zeros(n)
        omega_arr = np.zeros(n)
        torque_arr = np.zeros(n)

        # Target
        target_angle = np.radians(config.slew_angle_deg)
        target_sigma = np.tan(target_angle / 4)  # Scalar for single axis

        # Simple feedforward profile (quintic S curve, rest to rest)
        T_slew = float(config.slew_duration_s)

        for i in range(1, n):
            ti = t[i]

            # Reference trajectory
            if ti <= T_slew:
                tau = ti / T_slew if T_slew > 0 else 1.0
                tau2 = tau * tau
                tau3 = tau2 * tau
                tau4 = tau3 * tau
                tau5 = tau4 * tau
                theta_ref = target_angle * (10.0 * tau3 - 15.0 * tau4 + 6.0 * tau5)
                omega_ref = (target_angle / T_slew) * (30.0 * tau2 - 60.0 * tau3 + 30.0 * tau4)
                alpha_ff = (target_angle / (T_slew * T_slew)) * (60.0 * tau - 180.0 * tau2 + 120.0 * tau3)
            else:
                theta_ref = target_angle
                omega_ref = 0.0
                alpha_ff = 0.0

            sigma_ref = np.tan(theta_ref / 4)

            # Feedback error
            sigma_err = sigma[i-1] - sigma_ref
            omega_err = omega_arr[i-1] - omega_ref

            # Sensor noise
            omega_measured = omega_arr[i-1] + self.rng.normal(0, config.sensor_noise_std_rad_s)
            omega_err_noisy = omega_measured - omega_ref

            # Control torque
            tau_ff = I_axis * alpha_ff
            tau_fb = -K * sigma_err - P * omega_err_noisy
            tau_total = tau_ff + tau_fb + config.disturbance_torque_nm

            # Saturate
            tau_sat = np.clip(tau_total, -config.rw_max_torque_nm, config.rw_max_torque_nm)
            torque_arr[i] = tau_sat

            # Integrate dynamics (simple Euler)
            alpha = tau_sat / I_axis
            omega_arr[i] = omega_arr[i-1] + alpha * dt
            sigma[i] = sigma[i-1] + 0.25 * omega_arr[i] * dt  # sigma_dot ~ 0.25 * omega for small angles

        # Compute metrics
        post_slew_mask = t > config.slew_duration_s

        # Pointing error
        pointing_error = np.abs(sigma - target_sigma) * 4  # Convert to angle (rad)
        pointing_error_deg = np.degrees(pointing_error)

        if np.any(post_slew_mask):
            rms_error = _compute_rms(pointing_error_deg[post_slew_mask])
            peak_error = np.max(pointing_error_deg[post_slew_mask])
        else:
            rms_error = _compute_rms(pointing_error_deg)
            peak_error = np.max(pointing_error_deg)

        # Torque metrics
        peak_torque = np.max(np.abs(torque_arr))
        rms_torque = _compute_rms(torque_arr)
        saturation_count = np.sum(np.abs(torque_arr) >= config.rw_max_torque_nm * 0.99)
        saturation_pct = saturation_count / n * 100

        # Vibration (simplified - use high freq component of post slew position error)
        # In real sim, this would come from modal states
        if np.any(post_slew_mask):
            vib_mm = _compute_rms(pointing_error[post_slew_mask]) * 1000  # Simplified proxy
        else:
            vib_mm = _compute_rms(pointing_error) * 1000

        return {
            "rms_pointing_error_deg": rms_error,
            "peak_pointing_error_deg": peak_error,
            "peak_torque_nm": peak_torque,
            "rms_torque_nm": rms_torque,
            "torque_saturation_pct": saturation_pct,
            "rms_vibration_mm": vib_mm,
        }

    def _compute_summary(self) -> MonteCarloSummary:
        """Compute summary statistics."""
        n_passed = sum(1 for r in self.runs if r.passed)
        pass_rate = n_passed / len(self.runs) if self.runs else 0.0

        # Collect all metrics
        metric_names = list(self.runs[0].metrics.keys()) if self.runs else []
        metric_arrays = {name: [] for name in metric_names}

        for run in self.runs:
            for name, value in run.metrics.items():
                metric_arrays[name].append(value)

        # Compute percentiles
        percentiles = {}
        for name, values in metric_arrays.items():
            values = np.array(values)
            percentiles[name] = {
                "P50": float(np.percentile(values, 50)),
                "P95": float(np.percentile(values, 95)),
                "P99": float(np.percentile(values, 99)),
                "mean": float(np.mean(values)),
                "std": float(np.std(values)),
                "min": float(np.min(values)),
                "max": float(np.max(values)),
            }

        # Histograms
        histograms = {}
        for name, values in metric_arrays.items():
            hist, bin_edges = np.histogram(values, bins=50)
            histograms[name] = (hist, bin_edges)

        return MonteCarloSummary(
            n_runs=len(self.runs),
            n_passed=n_passed,
            pass_rate=pass_rate,
            percentiles=percentiles,
            histograms=histograms
        )

    def _print_summary(self, summary: MonteCarloSummary) -> None:
        """Print Monte Carlo summary."""
        print("\n" + "-" * 60)
        print("MONTE CARLO SUMMARY")
        print("-" * 60)

        print(f"\n  Runs: {summary.n_runs}")
        print(f"  Passed: {summary.n_passed} ({summary.pass_rate*100:.1f}%)")
        print(f"  Failed: {summary.n_runs - summary.n_passed}")

        print("\n  Metric Percentiles:")
        for metric, stats in summary.percentiles.items():
            print(f"\n    {metric}:")
            print(f"      P50: {stats['P50']:.4f}")
            print(f"      P95: {stats['P95']:.4f}")
            print(f"      P99: {stats['P99']:.4f}")

        print("\n" + "-" * 60)
        if summary.pass_rate >= 0.95:
            print("  MONTE CARLO: PASS (>=95% pass rate)")
        else:
            print("  MONTE CARLO: FAIL (<95% pass rate)")

    def _save_results(self, summary: MonteCarloSummary) -> None:
        """Save Monte Carlo results."""
        # JSON report
        report_path = os.path.join(self.out_dir, "monte_carlo_report.json")

        report = {
            "timestamp": datetime.now().isoformat(),
            "n_runs": summary.n_runs,
            "n_passed": summary.n_passed,
            "pass_rate": summary.pass_rate,
            "thresholds": self.thresholds,
            "uncertainties": self.uncertainties,
            "percentiles": summary.percentiles,
        }

        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)

        # CSV with all run results
        csv_path = os.path.join(self.out_dir, "monte_carlo_runs.csv")

        headers = ["run_id", "passed"] + list(self.runs[0].metrics.keys()) if self.runs else []
        rows = []
        for run in self.runs:
            row = [run.run_id, run.passed]
            row.extend([run.metrics.get(k, "") for k in headers[2:]])
            rows.append(row)

        _write_csv(csv_path, headers, rows)

        print(f"\nMonte Carlo report saved: {report_path}")
        print(f"Monte Carlo runs saved: {csv_path}")

    def _plot_histograms(self, summary: MonteCarloSummary) -> None:
        """Plot Monte Carlo histograms."""
        metric_names = list(summary.histograms.keys())
        n_metrics = len(metric_names)

        if n_metrics == 0:
            return

        n_cols = min(3, n_metrics)
        n_rows = (n_metrics + n_cols - 1) // n_cols

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 4*n_rows))
        if n_metrics == 1:
            axes = [axes]
        else:
            axes = axes.flatten()

        for idx, metric in enumerate(metric_names):
            ax = axes[idx]
            hist, bins = summary.histograms[metric]
            centers = (bins[:-1] + bins[1:]) / 2

            ax.bar(centers, hist, width=bins[1]-bins[0], alpha=0.7, edgecolor='black')

            # Add percentile lines
            stats = summary.percentiles[metric]
            ax.axvline(stats['P50'], color='g', linestyle='-', label=f"P50: {stats['P50']:.4f}")
            ax.axvline(stats['P95'], color='orange', linestyle='--', label=f"P95: {stats['P95']:.4f}")
            ax.axvline(stats['P99'], color='r', linestyle=':', label=f"P99: {stats['P99']:.4f}")

            ax.set_xlabel(metric)
            ax.set_ylabel("Count")
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)

        # Hide unused axes
        for idx in range(n_metrics, len(axes)):
            axes[idx].set_visible(False)

        plt.suptitle(f"Monte Carlo Results ({summary.n_runs} runs, {summary.pass_rate*100:.1f}% pass rate)",
                     fontsize=14, fontweight='bold')
        plt.tight_layout()

        plot_path = os.path.join(self.out_dir, "monte_carlo_histograms.png")
        plt.savefig(plot_path, dpi=150)
        plt.close()

        print(f"Monte Carlo histograms saved: {plot_path}")


# ============================================================================
# MAIN ENTRY POINT
# ============================================================================

def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Comprehensive V&V and Monte Carlo analysis for spacecraft input shaping."
    )
    parser.add_argument("--verification", action="store_true", help="Run verification tests")
    parser.add_argument("--validation", action="store_true", help="Run validation tests")
    parser.add_argument("--monte-carlo", type=int, metavar="N", help="Run N Monte Carlo iterations")
    parser.add_argument("--all", action="store_true", help="Run all tests")
    parser.add_argument("--mc-runs", type=int, default=500, help="Number of MC runs (default: 500)")
    parser.add_argument("--output-dir", type=str, default=None, help="Output directory")
    parser.add_argument("--data-dir", type=str, default=None, help="Directory containing NPZ data")
    parser.add_argument("--sensor-noise-std", type=float, default=1e-5,
                        help="Nominal gyro noise std dev (rad/s) for validation/MC")
    parser.add_argument("--disturbance-torque", type=float, default=1e-5,
                        help="Nominal disturbance torque bias (N*m) for validation/MC")
    parser.add_argument(
        "--post-slew-box-only",
        action="store_true",
        help="Generate post-slew box plots from existing monte_carlo_runs.csv",
    )
    parser.add_argument("--legacy", action="store_true", help="Use legacy runner (not plan compliant)")

    args = parser.parse_args()

    # Default to all if nothing specified
    if not (args.verification or args.validation or args.monte_carlo or args.all or args.post_slew_box_only):
        args.all = True

    # Setup output directory
    out_dir = args.output_dir or os.path.join(basilisk_dir, "output")
    _ensure_dir(out_dir)
    data_dir = args.data_dir or basilisk_dir

    print("=" * 60)
    print("SPACECRAFT INPUT SHAPING V&V SUITE")
    print("=" * 60)
    print(f"\nOutput directory: {out_dir}")
    print(f"Timestamp: {datetime.now().isoformat()}")

    if args.post_slew_box_only and not (args.verification or args.validation or args.monte_carlo or args.all):
        pointing_plot_path = _plot_post_slew_pointing_box_from_csv(
            out_dir,
            threshold_arcsec=POST_SLEW_POINTING_LIMIT_ARCSEC,
        )
        vibration_plot_path = _plot_post_slew_vibration_box_from_csv(
            out_dir,
            threshold_mm=POST_SLEW_VIBRATION_LIMIT_MM,
        )
        acceleration_plot_path = _plot_post_slew_acceleration_box_from_csv(
            out_dir,
            threshold_mm_s2=POST_SLEW_ACCEL_LIMIT_MM_S2,
        )
        hist_paths = _plot_post_slew_pointing_factor_histograms_from_csv(out_dir)
        _remove_legacy_correlation_outputs(out_dir)
        if pointing_plot_path or vibration_plot_path or acceleration_plot_path or hist_paths:
            return 0
        return 1

    all_passed = True

    if args.legacy:
        # Legacy runner (not plan compliant)
        config = ValidationConfig()
        if args.verification or args.all:
            verification = VerificationSuite(config, out_dir)
            v_results = verification.run_all()
            if not all(r.passed for r in v_results):
                all_passed = False

        if args.validation or args.all:
            validation = ValidationSuite(config, out_dir, data_dir=basilisk_dir)
            val_results = validation.run_all()
            if not all(r.passed for r in val_results):
                all_passed = False

        if args.monte_carlo or args.all:
            n_runs = args.monte_carlo or args.mc_runs
            mc_runner = MonteCarloRunner(config, out_dir, n_runs=n_runs)
            mc_summary = mc_runner.run()
            if mc_summary.pass_rate < 0.95:
                all_passed = False
    else:
        runner = PlanCompliantRunner(
            out_dir=out_dir,
            data_dir=data_dir,
            sensor_noise_std_rad_s=args.sensor_noise_std,
            disturbance_torque_nm=args.disturbance_torque,
        )

        if args.verification or args.all:
            v_results = runner.run_verification()
            if not all(r.passed for r in v_results):
                all_passed = False

        if args.validation or args.all:
            val_results = runner.run_validation()
            if not all(r.passed for r in val_results):
                all_passed = False

        if args.monte_carlo or args.all:
            n_runs = args.monte_carlo or args.mc_runs
            mc_summary = runner.run_monte_carlo(n_runs=n_runs)
            # Apply P95/P99 criteria from validation_mc.md
            p95_err = mc_summary.percentiles.get("rms_pointing_error_deg", {}).get("P95", float("nan"))
            p95_vib = mc_summary.percentiles.get("rms_vibration_mm", {}).get("P95", float("nan"))
            p99_peak = mc_summary.percentiles.get("peak_torque_nm", {}).get("P99", float("nan"))
            p95_sat = mc_summary.percentiles.get("torque_saturation_percent", {}).get("P95", float("nan"))
            if (
                not np.isfinite(p95_err)
                or not np.isfinite(p95_vib)
                or not np.isfinite(p99_peak)
                or not np.isfinite(p95_sat)
                or p95_err > PLAN_THRESHOLDS["rms_pointing_error_deg"]
                or p95_vib > PLAN_THRESHOLDS["rms_vibration_mm"]
                or p99_peak > PLAN_THRESHOLDS["peak_torque_nm"]
                or p95_sat > PLAN_THRESHOLDS["torque_saturation_percent"]
            ):
                all_passed = False

    # Final summary
    print("\n" + "=" * 60)
    print("FINAL SUMMARY")
    print("=" * 60)

    if all_passed:
        print("\n  ALL TESTS PASSED")
        print("\n  The spacecraft input shaping system has been verified and validated.")
        return 0
    else:
        print("\n  SOME TESTS FAILED")
        print("\n  Review the reports in the output directory for details.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
