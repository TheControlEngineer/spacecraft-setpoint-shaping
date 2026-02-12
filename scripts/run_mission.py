"""Unified mission simulation analysis for spacecraft slew maneuvers.

Evaluates feedforward torque shaping profiles (S Curve and fourth order)
alongside closed loop PD feedback control on a flexible spacecraft model.

Generates:
    - Torque profiles and residual vibration comparisons across shaping methods
    - Closed loop stability margins, sensitivity, and Nyquist analysis
    - Pointing error from combined rigid body and flexible mode response
    - PSD diagnostics for torque commands, vibration, and tracking error
    - CSV exports and publication quality plots for all computed metrics

The flexible dynamics use a coupled rigid flex state space model with
reaction wheel torque input and MRP attitude / camera pointing outputs.
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

# Add src directory to path for basilisk_sim imports
_script_dir = Path(__file__).parent.resolve()
_src_dir = _script_dir.parent / "src"
if str(_src_dir) not in sys.path:
    sys.path.insert(0, str(_src_dir))

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.ticker import LogLocator
from scipy import signal
from scipy.fft import fft, fftfreq

from basilisk_sim.spacecraft_properties import (
    HUB_INERTIA,
    compute_effective_inertia,
    compute_modal_gains,
    compute_mode_lever_arms,
    FLEX_MODE_MASS,
)
from basilisk_sim.design_shaper import (
    design_s_curve_trajectory,
)


# ===========================================================================
# Configuration
# ===========================================================================


@dataclass
class MissionConfig:
    """Parameters defining a spacecraft slew mission and its flexible dynamics.

    Attributes:
        inertia: 3x3 spacecraft inertia tensor (kg m^2).
        rotation_axis: Unit vector for the single axis slew.
        modal_freqs_hz: Natural frequencies of flexible appendage modes (Hz).
        modal_damping: Damping ratios for each flexible mode.
        modal_gains: Torque to modal acceleration coupling gains (rad/s^2 per N m).
        slew_angle_deg: Total slew angle (degrees).
        slew_duration_s: Nominal duration of the slew maneuver (seconds).
        feedforward_inertia: Inertia used to size feedforward torque (defaults to inertia).
        control_modal_gains: Modal gains used in the controller plant model.
        control_filter_cutoff_hz: Derivative filter cutoff frequency (Hz).
        control_filter_phase_lag_deg: Allowable phase lag from the derivative filter.
        vibration_highpass_hz: High pass cutoff for isolating flex vibration (Hz).
        pointing_error_spec_asd_deg: Pointing error ASD specification (deg/sqrt(Hz)).
        pointing_error_spec_label: Label for specification line in plots.
        feedback_method: Feedforward method used in combined feedback runs.
        camera_lever_arm_m: Camera offset from spacecraft center of mass (m).
        modal_mass_kg: Effective mass per flexible mode (kg).
        rw_max_torque_nm: Maximum reaction wheel torque capacity (N m).
    """

    inertia: np.ndarray
    rotation_axis: np.ndarray
    modal_freqs_hz: List[float]
    modal_damping: List[float]
    modal_gains: List[float]
    slew_angle_deg: float
    slew_duration_s: float
    feedforward_inertia: Optional[np.ndarray] = None
    control_modal_gains: Optional[List[float]] = None
    control_filter_cutoff_hz: Optional[float] = None
    control_filter_phase_lag_deg: float = 3.0
    vibration_highpass_hz: Optional[float] = 0.2
    pointing_error_spec_asd_deg: Optional[float] = None
    pointing_error_spec_label: Optional[str] = None
    feedback_method: Optional[str] = None
    camera_lever_arm_m: float = 4.0
    modal_mass_kg: Optional[float] = None
    rw_max_torque_nm: Optional[float] = 70.0


METHODS = ["s_curve", "fourth"]
# Mission comparison is intentionally constrained to standard PD only.
CONTROLLERS = ["standard_pd"]
UNIFIED_SAMPLE_DT = 0.01  # 100 Hz to match Basilisk simulation

METHOD_LABELS = {
    "s_curve": "S-Curve",
    "fourth": "Fourth-Order",
}

METHOD_COLORS = {
    "s_curve": "#17becf", # cyan
    "fourth": "#2ca02c", # green
}

CONTROLLER_LABELS = {
    "standard_pd": "Standard PD",
}

CONTROLLER_COLORS = {
    "standard_pd": "#ff7f0e", # orange
}

# Comparison colors for method + controller combinations (solid lines).
COMBO_COLORS = {
    ("s_curve", "standard_pd"): "#17becf", # cyan
    ("fourth", "standard_pd"): "#2ca02c", # green
}


def _combo_label(method: str, controller: str) -> str:
    """Format legend label for a method + controller pair."""
    return f"{METHOD_LABELS.get(method, method)} + {CONTROLLER_LABELS.get(controller, controller)}"


def _combo_color(method: str, controller: str) -> str:
    """Pick the requested color for a method + controller pair."""
    return COMBO_COLORS.get((method, controller), CONTROLLER_COLORS.get(controller, "#333333"))


def default_config() -> MissionConfig:
    """Return default mission configuration.

    Modal gains map torque to modal acceleration for base excitation:
        q_ddot + 2*zeta*omega*q_dot + omega^2*q = gain * torque
        gain = r / I_axis

    The static displacement per unit torque is gain / omega^2.
    Typical values at the modal frequency are 1e-4 to 1e-2 m/(N.m).

    IMPORTANT: modal_gains and control_modal_gains should be CONSISTENT.
    Modal gains are computed from lever arm / I_axis.
    
    Controller Design (based on analysis in optimal_controller_design.py):
    - Bandwidth = first_mode/2.5 = 0.16 Hz (still below mode 1)
    - Controller bandwidth = first_mode/2.5 for adequate phase margin
    - Derivative filter cutoff is set high enough to preserve phase margin
    """
    inertia_eff = compute_effective_inertia(hub_inertia=HUB_INERTIA.copy())
    rotation_axis = np.array([0.0, 0.0, 1.0])
    modal_freqs_hz = [0.4, 1.3]
    modal_gains = compute_modal_gains(inertia_eff, rotation_axis)
    if not modal_gains:
        modal_gains = [0.0] * len(modal_freqs_hz)
    elif len(modal_gains) < len(modal_freqs_hz):
        modal_gains = (modal_gains + [modal_gains[-1]] * len(modal_freqs_hz))[: len(modal_freqs_hz)]
    first_mode = min(modal_freqs_hz) if modal_freqs_hz else 0.4
    control_bandwidth_hz = first_mode / 2.5
    # Filtered PD cutoff (user requested): 8.0 Hz for current tuning.
    control_filter_cutoff_hz = 8.0
    return MissionConfig(
        inertia=inertia_eff.copy(),
        feedforward_inertia=inertia_eff.copy(),
        rotation_axis=rotation_axis,
        modal_freqs_hz=modal_freqs_hz,
        modal_damping=[0.02, 0.015],
        modal_gains=modal_gains,
        control_modal_gains=modal_gains,
        control_filter_cutoff_hz=control_filter_cutoff_hz,
        control_filter_phase_lag_deg=3.0,
        slew_angle_deg=180.0,
        slew_duration_s=30.0,
        vibration_highpass_hz=None,
        pointing_error_spec_asd_deg=None,
        pointing_error_spec_label=None,
        feedback_method="fourth",
        camera_lever_arm_m=4.0,
        modal_mass_kg=FLEX_MODE_MASS,
    )


# ===========================================================================
# Utility Functions
# ===========================================================================


def _ensure_dir(path: str) -> None:
    """Create directory if it doesn't exist."""
    os.makedirs(path, exist_ok=True)


def _mirror_output(path: str, mirror_dir: str) -> None:
    """Copy an output file to a mirror directory if needed."""
    if not path or not mirror_dir:
        return
    if not os.path.isdir(mirror_dir):
        return
    dst = os.path.join(mirror_dir, os.path.basename(path))
    if os.path.abspath(dst) == os.path.abspath(path):
        return
    try:
        shutil.copy2(path, dst)
    except OSError:
        pass


def _write_csv(path: str, headers: List[str], rows: Iterable[List[str]]) -> None:
    """Write CSV file."""
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        writer.writerows(rows)


def _get_feedforward_inertia(config: MissionConfig) -> np.ndarray:
    """Return inertia used for feedforward sizing."""
    if config.feedforward_inertia is None:
        return np.array(config.inertia, dtype=float)
    return np.array(config.feedforward_inertia, dtype=float)


def _highpass_filter(data: np.ndarray, time: np.ndarray, cutoff_hz: float) -> np.ndarray:
    """Apply a second order Butterworth high pass filter to remove slow trends."""
    if len(data) < 10 or cutoff_hz <= 0:
        return data
    dt = np.median(np.diff(time))
    fs = 1.0 / dt
    nyq = fs / 2.0
    if cutoff_hz >= nyq:
        return data
    b, a = signal.butter(2, cutoff_hz / nyq, btype="high")
    return signal.filtfilt(b, a, data)


def _get_vibration_highpass_hz(config: MissionConfig) -> float:
    """Select a high pass cutoff frequency to isolate flexible vibration."""
    if config.vibration_highpass_hz is not None:
        return float(config.vibration_highpass_hz)
    duration = float(config.slew_duration_s)
    base = 2.0 / duration if duration > 0 else 0.05
    if config.modal_freqs_hz:
        return max(base, 0.5 * min(config.modal_freqs_hz))
    return base


def _sigma_to_angle(sigma: np.ndarray, axis: np.ndarray) -> np.ndarray:
    """Project MRP onto a rotation axis and return signed rotation angle [rad]."""
    sigma = np.atleast_2d(np.array(sigma, dtype=float))
    if sigma.size == 0:
        return np.array([])
    axis = _normalize_axis(axis)
    dot = sigma @ axis
    mag = np.linalg.norm(sigma, axis=1)
    angle = 4.0 * np.arctan(mag)
    sign = np.sign(dot)
    sign[sign == 0] = 1.0
    return angle * sign


def _wrap_angle_rad(angle: np.ndarray) -> np.ndarray:
    """Wrap angle(s) to [-pi, pi] for shortest-distance errors."""
    return (angle + np.pi) % (2 * np.pi) - np.pi


def _extract_vibration_signals(
    time: np.ndarray,
    displacement: np.ndarray,
    config: MissionConfig,
    acceleration: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """High pass filter displacement and derive vibration acceleration.

    If acceleration is provided, it is filtered and detrended directly without
    numerical differentiation of displacement.
    """
    if len(time) == 0 or len(displacement) == 0:
        return np.array([]), np.array([])
    cutoff_hz = _get_vibration_highpass_hz(config)
    disp = _highpass_filter(np.array(displacement, dtype=float), time, cutoff_hz)
    disp = signal.detrend(disp, type="linear")
    if acceleration is None or len(acceleration) == 0:
        acc = _compute_acceleration_from_displacement(time, disp)
    else:
        acc = np.array(acceleration, dtype=float)
        if len(acc) != len(time):
            n = min(len(acc), len(time))
            acc = acc[:n]
    acc = _highpass_filter(np.array(acc, dtype=float), time, cutoff_hz)
    acc = signal.detrend(acc, type="linear")
    return disp, acc


def _detrend_mean(data: np.ndarray) -> np.ndarray:
    """Remove mean from data."""
    return data - np.mean(data)


def _choose_psd_params(time: np.ndarray, signal_data: np.ndarray) -> Optional[Dict[str, object]]:
    """Choose Welch PSD parameters for balanced resolution and variance."""
    n = len(signal_data)
    if n < 16:
        return None
    dt = np.median(np.diff(time))
    if not np.isfinite(dt) or dt <= 0:
        return None
    fs = 1.0 / dt

    max_nperseg = min(4096, n)
    if max_nperseg < 8:
        return None

    # Target resolution around 0.02 Hz if possible, otherwise use the longest feasible segment.
    target_df = 0.02
    nperseg_target = int(round(fs / target_df))
    nperseg = min(max_nperseg, max(8, nperseg_target))

    # Use a power of two length for efficiency and stable grids.
    nperseg = 2 ** int(np.floor(np.log2(nperseg)))
    nperseg = max(8, min(nperseg, max_nperseg))

    # Ensure enough averages for a meaningful PSD (variance reduction).
    min_segments = 6
    overlap_ratio = 0.5
    while nperseg >= 8:
        noverlap = int(nperseg * overlap_ratio)
        step = nperseg - noverlap
        segments = 1 + max(0, (n - nperseg) // step) if step > 0 else 0
        if segments >= min_segments or nperseg <= 64:
            break
        nperseg //= 2

    if nperseg < 8:
        return None

    noverlap = int(nperseg * overlap_ratio)
    return {
        "fs": fs,
        "nperseg": nperseg,
        "noverlap": noverlap,
        "window": "hann",
        "detrend": "constant",
        "scaling": "density",
    }


def _compute_psd(time: np.ndarray, signal_data: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Compute power spectral density using Welch's method."""
    params = _choose_psd_params(time, signal_data)
    if not params:
        return np.array([]), np.array([])
    freq, psd = signal.welch(
        signal_data,
        fs=params["fs"],
        window=params["window"],
        nperseg=params["nperseg"],
        noverlap=params["noverlap"],
        detrend=params["detrend"],
        scaling=params["scaling"],
    )
    return freq, psd


def _compute_psd_high_resolution(
    time: np.ndarray,
    signal_data: np.ndarray,
    zero_pad_factor: int = 8,
    max_nfft: int = 1 << 19,
    window: object = "hann",
    detrend: object = "constant",
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute a high resolution PSD using zero padded periodogram for visual diagnostics."""
    n = len(signal_data)
    if n < 16:
        return np.array([]), np.array([])
    dt = np.median(np.diff(time))
    if not np.isfinite(dt) or dt <= 0:
        return np.array([]), np.array([])
    fs = 1.0 / dt

    n_target = max(n, 8) * max(1, int(zero_pad_factor))
    nfft = 2 ** int(np.ceil(np.log2(n_target)))
    nfft = int(min(max_nfft, max(256, nfft)))

    freq, psd = signal.periodogram(
        signal_data,
        fs=fs,
        window=window,
        detrend=detrend,
        scaling="density",
        nfft=nfft,
    )
    return freq, psd


def _detect_mode_lines_from_flexible_plant(
    control_data: Dict[str, object],
    config: MissionConfig,
    fmax_hz: float = 10.0,
) -> Tuple[List[float], List[float]]:
    """Detect resonance and antiresonance frequencies from flexible plant magnitude."""
    freqs = np.array(control_data.get("freqs", []), dtype=float)
    plant = np.array(control_data.get("plant_flex_body", []), dtype=complex)
    if len(freqs) == 0 or len(plant) == 0 or not config.modal_freqs_hz:
        return [], []

    mask = (
        np.isfinite(freqs)
        & np.isfinite(np.real(plant))
        & np.isfinite(np.imag(plant))
        & (freqs > 0)
        & (freqs <= fmax_hz)
    )
    if not np.any(mask):
        return [], []

    f = freqs[mask]
    mag_db = 20.0 * np.log10(np.abs(plant[mask]) + 1e-12)

    peak_idx, _ = signal.find_peaks(mag_db, prominence=0.25)
    dip_idx, _ = signal.find_peaks(-mag_db, prominence=0.25)
    if len(peak_idx) == 0 and len(dip_idx) == 0:
        return [], []

    resonances: List[float] = []
    antiresonances: List[float] = []

    for f_mode in sorted(float(v) for v in config.modal_freqs_hz):
        if f_mode <= 0:
            continue
        left = max(1e-3, 0.6 * f_mode)
        right = min(float(f[-1]), 2.2 * f_mode)

        local_peaks = [idx for idx in peak_idx if left <= f[idx] <= right]
        f_res: Optional[float] = None
        if local_peaks:
            # Pick dominant local resonance by peak magnitude.
            best_peak = max(local_peaks, key=lambda idx: float(mag_db[idx]))
            f_res = float(f[best_peak])
            resonances.append(f_res)

        dip_right = f_res if f_res is not None else min(float(f[-1]), 1.2 * f_mode)
        local_dips = [idx for idx in dip_idx if left <= f[idx] <= dip_right]
        if local_dips:
            # Pick deepest local antiresonance.
            best_dip = min(local_dips, key=lambda idx: float(mag_db[idx]))
            antiresonances.append(float(f[best_dip]))

    def _dedupe(values: List[float], rel_tol: float = 0.03) -> List[float]:
        """Remove nearly duplicate values within relative tolerance."""
        out: List[float] = []
        for val in sorted(values):
            if not out:
                out.append(val)
                continue
            if abs(val - out[-1]) / max(out[-1], 1e-6) > rel_tol:
                out.append(val)
        return out

    return _dedupe(resonances), _dedupe(antiresonances)


def _draw_mode_lines_on_axis(
    ax: plt.Axes,
    resonances_hz: List[float],
    antiresonances_hz: List[float],
) -> None:
    """Draw resonance and antiresonance vertical markers."""
    for i, f_res in enumerate(resonances_hz):
        ax.axvline(
            f_res,
            color="#d62728",
            linestyle="--",
            linewidth=1.4,
            alpha=0.9,
            label="Resonance" if i == 0 else None,
        )
    for i, f_ar in enumerate(antiresonances_hz):
        ax.axvline(
            f_ar,
            color="#9467bd",
            linestyle=":",
            linewidth=1.5,
            alpha=0.9,
            label="Antiresonance" if i == 0 else None,
        )


def _compute_band_rms(freq: np.ndarray, psd: np.ndarray, fmin: float, fmax: float) -> float:
    """Compute band limited RMS by integrating a PSD between fmin and fmax."""
    if len(freq) == 0 or len(psd) == 0:
        return float("nan")
    mask = (freq >= fmin) & (freq <= fmax) & np.isfinite(psd) & (psd >= 0)
    if not np.any(mask):
        return float("nan")
    freq_sel = freq[mask]
    psd_sel = psd[mask]
    df = np.gradient(freq_sel)
    return float(np.sqrt(np.sum(psd_sel * df)))


def _compute_psd_asd(
    time: np.ndarray,
    signal_data: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute PSD and ASD."""
    freq, psd = _compute_psd(time, signal_data)
    asd = np.sqrt(psd) if len(psd) else np.array([])
    return freq, psd, asd


def _compute_cumulative_rms(freq: np.ndarray, psd: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Compute cumulative RMS from PSD."""
    if len(freq) == 0 or len(psd) == 0:
        return np.array([]), np.array([])
    mask = (freq > 0) & np.isfinite(psd) & (psd >= 0)
    if not np.any(mask):
        return np.array([]), np.array([])
    freq_sel = freq[mask]
    psd_sel = psd[mask]
    df = np.gradient(freq_sel)
    cumulative = np.cumsum(psd_sel * df)
    return freq_sel, np.sqrt(cumulative)


def _mrp_shadow(sigma: np.ndarray) -> np.ndarray:
    """Return the shadow MRP if |sigma| > 1 to enforce the shortest rotation."""
    sigma = np.array(sigma, dtype=float).flatten()
    sigma_norm_sq = float(np.dot(sigma, sigma))
    if sigma_norm_sq > 1.0 + 1e-12:
        return -sigma / sigma_norm_sq
    return sigma


def _mrp_subtract(sigma_body: np.ndarray, sigma_ref: np.ndarray) -> np.ndarray:
    """Compute MRP attitude error sigma_BR = sigma_BN (-) sigma_RN."""
    sigma_body = np.array(sigma_body, dtype=float).flatten()
    sigma_ref = np.array(sigma_ref, dtype=float).flatten()

    sigma_body_sq = float(np.dot(sigma_body, sigma_body))
    sigma_ref_sq = float(np.dot(sigma_ref, sigma_ref))
    dot_product = float(np.dot(sigma_ref, sigma_body))

    denom = 1.0 + sigma_body_sq * sigma_ref_sq + 2.0 * dot_product
    if abs(denom) < 1e-12:
        return np.zeros(3)

    sigma_error = (
        (1.0 - sigma_ref_sq) * sigma_body
        - (1.0 - sigma_body_sq) * sigma_ref
        + 2.0 * np.cross(sigma_body, sigma_ref)
    ) / denom

    return _mrp_shadow(sigma_error)


def _normalize_axis(axis: np.ndarray) -> np.ndarray:
    """Return a unit vector for the rotation axis."""
    axis = np.array(axis, dtype=float).flatten()
    norm = np.linalg.norm(axis)
    if norm <= 0:
        return np.array([0.0, 0.0, 1.0])
    return axis / norm


def _pad_list(values: Optional[Iterable[float]], count: int, default: float) -> List[float]:
    """Pad or trim a list to a fixed count."""
    if count <= 0:
        return []
    vals = []
    if values is not None:
        vals = [float(v) for v in values]
    if len(vals) >= count:
        return vals[:count]
    if vals:
        vals.extend([vals[-1]] * (count - len(vals)))
        return vals
    return [float(default)] * count


def _infer_modal_lever_arms(config: MissionConfig, axis: np.ndarray, inertia_axis: float) -> List[float]:
    """Infer modal lever arms from gains or geometry."""
    n_modes = len(config.modal_freqs_hz)
    lever_arms: List[float] = []
    if config.modal_gains:
        lever_arms = [float(gain) * float(inertia_axis) for gain in config.modal_gains]
    if not lever_arms:
        lever_arms = compute_mode_lever_arms(rotation_axis=axis)
    if len(lever_arms) < n_modes:
        if lever_arms:
            lever_arms.extend([lever_arms[-1]] * (n_modes - len(lever_arms)))
        else:
            lever_arms = [0.0] * n_modes
    return lever_arms[:n_modes]


def _infer_modal_masses(config: MissionConfig, count: int) -> List[float]:
    """Infer modal masses for coupled dynamics."""
    modal_mass = float(config.modal_mass_kg) if config.modal_mass_kg is not None else float(FLEX_MODE_MASS)
    return [modal_mass] * count


def _align_series(time: np.ndarray, *series: np.ndarray) -> Tuple[np.ndarray, List[np.ndarray]]:
    """Align time and series arrays to the shortest length."""
    lengths = [len(time)] + [len(s) for s in series if s is not None]
    if not lengths:
        return time, list(series)
    n = min(lengths)
    aligned = []
    for s in series:
        if s is None:
            aligned.append(s)
        else:
            aligned.append(np.array(s, dtype=float)[:n])
    return np.array(time, dtype=float)[:n], aligned


def _resample_time_series(
    time: np.ndarray,
    *series: np.ndarray,
    dt_target: float = UNIFIED_SAMPLE_DT,
) -> Tuple[np.ndarray, List[np.ndarray]]:
    """Resample time series to a uniform target timestep."""
    time = np.array(time, dtype=float)
    if len(time) < 2:
        return time, [np.array(s, dtype=float) for s in series]
    dt = float(np.median(np.diff(time)))
    if not np.isfinite(dt) or abs(dt - dt_target) < 1e-9:
        return time, [np.array(s, dtype=float) for s in series]
    t_new = np.arange(time[0], time[-1] + dt_target * 0.5, dt_target)
    resampled = []
    for s in series:
        if s is None or len(s) == 0:
            resampled.append(np.array([]))
        else:
            resampled.append(np.interp(t_new, time, np.array(s, dtype=float)))
    print(f"Warning: resampled trajectory from dt={dt:.6f}s to {dt_target:.2f}s")
    return t_new, resampled


def _combine_modal_displacement(mode1: np.ndarray, mode2: np.ndarray) -> np.ndarray:
    """Combine modal displacements using linear sum.

    Uses linear combination (mode1 + mode2) to match the transfer function
    formulation: y_camera = theta + q1/L + q2/L. This preserves resonance
    peaks that RSS combination would dampen.
    """
    if mode1 is None and mode2 is None:
        return np.array([])
    if mode1 is None or len(mode1) == 0:
        return np.array(mode2, dtype=float)
    if mode2 is None or len(mode2) == 0:
        return np.array(mode1, dtype=float)
    mode1 = np.array(mode1, dtype=float)
    mode2 = np.array(mode2, dtype=float)
    n = min(len(mode1), len(mode2))
    return mode1[:n] + mode2[:n]


def _compute_acceleration_from_displacement(time: np.ndarray, displacement: np.ndarray) -> np.ndarray:
    """Compute acceleration from displacement with finite differences."""
    if len(time) < 3 or len(displacement) < 3:
        return np.zeros_like(displacement)
    vel = np.gradient(displacement, time)
    return np.gradient(vel, time)


def _project_axis_torque(torque: Optional[np.ndarray], axis: np.ndarray) -> np.ndarray:
    """Project a 3-axis torque vector onto a rotation axis."""
    if torque is None or len(torque) == 0:
        return np.array([])
    torque = np.array(torque, dtype=float)
    if torque.ndim == 1:
        return torque
    axis = _normalize_axis(axis)
    return torque @ axis


def _infer_maneuver_end(
    time: np.ndarray,
    control_mode: Optional[List[object]],
    torque: Optional[np.ndarray],
    fallback: float,
) -> float:
    """Infer feedforward maneuver end time from control mode or torque."""
    if len(time) == 0:
        return fallback
    if control_mode:
        ff_mask = np.array(
            [("FF" in str(mode)) and ("FF(0)" not in str(mode)) for mode in control_mode],
            dtype=bool,
        )
        if np.any(ff_mask):
            return float(time[np.where(ff_mask)[0][-1]])
    if torque is not None and len(torque) > 0:
        torque = np.array(torque, dtype=float)
        peak = np.max(np.abs(torque))
        if peak > 0:
            threshold = 0.01 * peak
            idx = np.where(np.abs(torque) > threshold)[0]
            if len(idx) > 0:
                return float(time[idx[-1]])
    return float(time[-1]) if len(time) > 0 else fallback


def _npz_matches_config(npz_data: Dict[str, object], config: MissionConfig) -> bool:
    """Check whether NPZ metadata matches the current configuration."""
    def _match_array(key: str, ref: Optional[Iterable[float]], rtol: float = 0.1) -> bool:
        """Return True if the NPZ array for key matches ref within rtol."""
        value = npz_data.get(key)
        if key not in npz_data or ref is None or value is None:
            return True
        try:
            data_arr = np.array(value, dtype=float).ravel()
            ref_arr = np.array(ref, dtype=float).ravel()
            if data_arr.shape != ref_arr.shape:
                return False
            return np.allclose(data_arr, ref_arr, rtol=rtol, atol=1e-12)
        except (TypeError, ValueError):
            return True

    def _match_scalar(key: str, ref: Optional[float], tol: float) -> bool:
        """Return True if the NPZ scalar for key matches ref within tol."""
        value = npz_data.get(key)
        if key not in npz_data or ref is None or value is None:
            return True
        try:
            return abs(float(value) - float(ref)) <= tol
        except (TypeError, ValueError):
            return True

    required_keys = [
        "modal_freqs_hz",
        "modal_damping",
        "modal_gains",
        "control_filter_cutoff_hz",
        "inertia_control",
        "slew_duration_s",
    ]
    if any(npz_data.get(key) is None for key in required_keys):
        return False

    # Check slew angle
    angle = npz_data.get("slew_angle_deg")
    if angle is not None:
        try:
            if abs(float(angle) - float(config.slew_angle_deg)) > 1.0:
                return False
        except (TypeError, ValueError):
            pass

    # Check slew duration
    if not _match_scalar("slew_duration_s", config.slew_duration_s, tol=0.5):
        return False

    # Check modal frequencies and damping
    if not _match_array("modal_freqs_hz", config.modal_freqs_hz, rtol=0.02):
        return False
    if not _match_array("modal_damping", config.modal_damping, rtol=0.2):
        return False

    # Check control filter cutoff only for filtered PD (not relevant for standard PD).
    controller = str(npz_data.get("controller", "")).lower()
    if "filtered_pd" in controller:
        if not _match_scalar("control_filter_cutoff_hz", config.control_filter_cutoff_hz, tol=0.05):
            return False

    # Check inertia used for control if present
    if not _match_array("inertia_control", config.inertia.flatten(), rtol=0.02):
        return False
    
    # Check modal gains because they strongly drive vibration magnitude
    if not _match_array("modal_gains", config.modal_gains, rtol=0.1):
        return False

    return True


# ===========================================================================
# Feedforward Analysis
# ===========================================================================


def _integrate_trajectory(
    alpha: np.ndarray, dt: float, theta_final: float, scale_to_final: bool = True
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Integrate acceleration to velocity/position."""
    if len(alpha) == 0 or dt <= 0:
        return np.array([]), np.array([]), alpha
    omega = np.cumsum(alpha) * dt
    theta = np.cumsum(omega) * dt
    if scale_to_final and abs(theta[-1]) > 1e-12:
        scale = theta_final / theta[-1]
        alpha = alpha * scale
        omega = omega * scale
        theta = theta * scale
    return theta, omega, alpha


def _compute_torque_profile(
    config: MissionConfig, method: str, settling_time: float = 30.0
) -> Dict[str, np.ndarray]:
    """Compute torque profile for a given shaping method.

    Args:
        config: Mission configuration
        method: Shaping method ('s_curve' or 'fourth')
        settling_time: Time after maneuver for vibration settling (seconds)

    Returns:
        Dictionary with time, torque, theta, omega, alpha arrays and maneuver_end time
    """
    theta_final = np.radians(config.slew_angle_deg)
    duration = config.slew_duration_s
    axis = _normalize_axis(config.rotation_axis)
    ff_inertia = _get_feedforward_inertia(config)
    I_axis = float(axis @ ff_inertia @ axis)

    if method == "fourth":
        traj_candidates = [
            os.path.join(os.path.dirname(__file__), "..", "data", "trajectories", "spacecraft_trajectory_4th_180deg_30s.npz"),
            os.path.join(os.path.dirname(__file__), "spacecraft_trajectory_4th_180deg_30s.npz"),
            os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "spacecraft_trajectory_4th_180deg_30s.npz")),
        ]
        traj_path = next((path for path in traj_candidates if os.path.isfile(path)), None)
        if traj_path is None:
            raise FileNotFoundError(
                "Fourth-order trajectory file not found. Expected "
                "`spacecraft_trajectory_4th_180deg_30s.npz` in data/trajectories/."
            )
        traj = np.load(traj_path, allow_pickle=True)
        t = np.array(traj.get("time", []), dtype=float)
        theta = np.array(traj.get("theta", []), dtype=float)
        omega = np.array(traj.get("omega", []), dtype=float)
        alpha = np.array(traj.get("alpha", []), dtype=float)

        min_len = min(len(t), len(theta), len(omega), len(alpha))
        if min_len == 0:
            raise ValueError(f"Fourth-order trajectory file has no usable data: {traj_path}")
        t = t[:min_len]
        theta = theta[:min_len]
        omega = omega[:min_len]
        alpha = alpha[:min_len]

        t, resampled = _resample_time_series(t, theta, omega, alpha)
        if resampled:
            theta, omega, alpha = resampled

        maneuver_end = float(t[-1])
        if len(t) > 1 and settling_time > 0:
            dt = float(np.median(np.diff(t)))
            if dt > 0:
                n_settling = int(round(settling_time / dt))
                if n_settling > 0:
                    t_extra = t[-1] + np.arange(1, n_settling + 1) * dt
                    t = np.concatenate([t, t_extra])
                    alpha = np.concatenate([alpha, np.zeros(n_settling)])
                    theta = np.concatenate([theta, np.full(n_settling, theta[-1])])
                    omega = np.concatenate([omega, np.full(n_settling, omega[-1])])

        achieved_deg = float(np.degrees(theta[-1]))
        if abs(abs(achieved_deg) - abs(config.slew_angle_deg)) > 0.5:
            print(
                f"Warning: fourth-order trajectory achieves {achieved_deg:.2f} deg "
                f"(target {config.slew_angle_deg:.2f} deg)."
            )

        torque = I_axis * alpha
        return {
            "time": t,
            "torque": torque,
            "theta": theta,
            "omega": omega,
            "alpha": alpha,
            "maneuver_end": maneuver_end,
        }

    if method == "s_curve":
        t, theta, omega, alpha, _, _, _ = design_s_curve_trajectory(
            target_duration=duration,
            theta_final=theta_final,
            I_axis=I_axis,
            max_torque=config.rw_max_torque_nm,
            dt=UNIFIED_SAMPLE_DT,
            settling_time=settling_time,
        )
    else:
        raise ValueError(f"Unknown shaping method: {method}")

    maneuver_end = duration
    torque = I_axis * alpha

    return {
        "time": t,
        "torque": torque,
        "theta": theta,
        "omega": omega,
        "alpha": alpha,
        "maneuver_end": maneuver_end,
    }


def _simulate_modal_response(
    time: np.ndarray,
    torque: np.ndarray,
    config: MissionConfig,
) -> Dict[str, np.ndarray]:
    """Simulate modal response using coupled rigid flex state space dynamics.

    Builds a multi degree of freedom state space model coupling the rigid body
    rotation with flexible appendage modes, then runs a linear simulation
    driven by the feedforward torque profile. Returns per mode and combined
    displacement and acceleration time histories.
    """
    time = np.array(time, dtype=float)
    torque = np.array(torque, dtype=float)
    dt = time[1] - time[0] if len(time) > 1 else 0.01
    n = len(time)

    if n == 0:
        return {
            "time": time,
            "mode1": np.array([]),
            "mode2": np.array([]),
            "displacement": np.array([]),
            "acceleration": np.array([]),
        }

    axis = _normalize_axis(config.rotation_axis)
    inertia_axis = float(axis @ config.inertia @ axis)
    lever_arms = _infer_modal_lever_arms(config, axis, inertia_axis)
    modal_masses = _infer_modal_masses(config, len(config.modal_freqs_hz))

    flex_ss = _build_coupled_flex_state_space(
        config.inertia,
        axis,
        config.modal_freqs_hz,
        config.modal_damping,
        modal_masses,
        lever_arms,
        sigma_scale=4.0,
        camera_lever_arm_m=config.camera_lever_arm_m,
    )
    pos_ss = flex_ss["positions"]
    _, pos, _ = signal.lsim(pos_ss, U=torque, T=time)

    mode1_disp = np.zeros(n)
    mode2_disp = np.zeros(n)
    if pos.ndim == 1:
        pos = pos.reshape(-1, 1)
    if pos.shape[1] >= 2:
        mode1_disp = np.array(pos[:, 1], dtype=float)
    if pos.shape[1] >= 3:
        mode2_disp = np.array(pos[:, 2], dtype=float)

    total_disp = mode1_disp + mode2_disp
    total_acc = np.gradient(np.gradient(total_disp, dt), dt)

    return {
        "time": time,
        "mode1": mode1_disp,
        "mode2": mode2_disp,
        "displacement": total_disp,
        "acceleration": total_acc,
    }


def _compute_feedforward_metrics(
    torque_data: Dict[str, np.ndarray],
    vibration_data: Dict[str, np.ndarray],
    config: MissionConfig,
) -> Dict[str, float]:
    """Compute feedforward performance metrics.

    Metrics computed:
        - RMS and peak torque during maneuver
        - RMS and peak residual vibration after maneuver ends

    The residual vibration is computed over a settling window after the maneuver
    completes. If insufficient settling data exists, uses the last 10% of the
    available data.
    """
    time = torque_data["time"]
    torque = torque_data["torque"]
    disp = vibration_data["displacement"]

    # Determine maneuver end time
    maneuver_end = vibration_data.get("maneuver_end", torque_data.get("maneuver_end", config.slew_duration_s))
    maneuver_end = float(maneuver_end) if maneuver_end is not None else config.slew_duration_s

    # Find index where maneuver ends
    maneuver_end_idx = int(np.searchsorted(time, maneuver_end)) if len(time) > 1 else len(time)

    # Compute residual vibration
    min_residual_samples = 100  # Minimum samples needed for meaningful statistics
    if maneuver_end_idx < len(disp) - min_residual_samples:
        # Sufficient settling data available
        residual = disp[maneuver_end_idx:]
    elif len(disp) > min_residual_samples:
        # Use last 10% of data as fallback
        fallback_start = int(0.9 * len(disp))
        residual = disp[fallback_start:]
    else:
        # If the record is too short, use all available samples
        residual = disp

    if len(residual) > 0 and np.max(np.abs(residual)) > 1e-15:
        rms_residual = float(np.sqrt(np.mean(residual**2))) * 1000  # mm
        peak_residual = float(np.max(np.abs(residual))) * 1000  # mm
    else:
        rms_residual = 0.0
        peak_residual = 0.0

    torque_window = torque[:maneuver_end_idx] if maneuver_end_idx > 0 else torque
    if len(torque_window) > 0:
        rms_torque = float(np.sqrt(np.mean(torque_window**2)))
        peak_torque = float(np.max(np.abs(torque_window)))
    else:
        rms_torque = 0.0
        peak_torque = 0.0

    return {
        "rms_torque_nm": rms_torque,
        "peak_torque_nm": peak_torque,
        "rms_vibration_mm": rms_residual,
        "peak_vibration_mm": peak_residual,
    }


def _collect_feedforward_data(
    config: MissionConfig,
    data_dir: Optional[str] = None,
    prefer_npz: bool = False,
) -> Tuple[Dict[str, Dict[str, np.ndarray]], Dict[str, Dict[str, np.ndarray]]]:
    """Collect feedforward torque and vibration data (NPZ preferred if requested)."""
    data_dir = data_dir or os.path.dirname(__file__)
    axis = _normalize_axis(config.rotation_axis)
    ff_inertia = _get_feedforward_inertia(config)
    I_axis = float(axis @ ff_inertia @ axis)

    torque_data: Dict[str, Dict[str, np.ndarray]] = {}
    vibration_data: Dict[str, Dict[str, np.ndarray]] = {}

    for method in METHODS:
        npz_data = None
        if prefer_npz:
            npz_data = _load_pointing_npz(
                method,
                data_dir,
                controller=None,
                mode="ff_only",
                generate_if_missing=False,
                allow_legacy=False,
            )
            if npz_data is not None and not _npz_matches_config(npz_data, config):
                npz_data = None

        if npz_data is not None:
            time = np.array(npz_data.get("time", []), dtype=float)
            torque_vec = npz_data.get("ff_torque")
            if torque_vec is None or len(torque_vec) == 0:
                torque_vec = npz_data.get("total_torque")
            mode1_acc = npz_data.get("mode1_acc")
            mode2_acc = npz_data.get("mode2_acc")
            if mode1_acc is not None and len(mode1_acc) == 0:
                mode1_acc = None
            if mode2_acc is not None and len(mode2_acc) == 0:
                mode2_acc = None
            torque_axis = _project_axis_torque(torque_vec, axis)
            time, aligned = _align_series(
                time, torque_axis, npz_data.get("mode1"), npz_data.get("mode2"), mode1_acc, mode2_acc
            )
            torque_axis = aligned[0]
            mode1 = aligned[1]
            mode2 = aligned[2]
            mode1_acc = aligned[3] if len(aligned) > 3 else np.array([])
            mode2_acc = aligned[4] if len(aligned) > 4 else np.array([])

            # Raw modal state signals (no filtering/detrending) for plotting/debug.
            displacement_raw = _combine_modal_displacement(mode1, mode2)
            displacement_raw = displacement_raw[: len(time)] if len(time) > 0 else displacement_raw
            acceleration_raw = np.array([])
            if mode1_acc is not None and len(mode1_acc) > 0:
                acceleration_raw = _combine_modal_displacement(mode1_acc, mode2_acc)
                acceleration_raw = acceleration_raw[: len(time)] if len(time) > 0 else acceleration_raw
            if len(acceleration_raw) == 0 and len(displacement_raw) > 0:
                acceleration_raw = _compute_acceleration_from_displacement(time, displacement_raw)

            # Legacy filtered signals retained for existing metrics/CSVs.
            displacement, acceleration = _extract_vibration_signals(
                time, displacement_raw, config, acceleration_raw
            )
            maneuver_end = _infer_maneuver_end(
                time, npz_data.get("control_mode"), torque_axis, config.slew_duration_s
            )

            dt = float(np.median(np.diff(time))) if len(time) > 1 else 0.0
            if dt > 0 and len(torque_axis) > 0:
                alpha = torque_axis / I_axis
                theta, omega, alpha = _integrate_trajectory(
                    alpha, dt, np.radians(config.slew_angle_deg), scale_to_final=False
                )
                if len(theta) > 0:
                    achieved_deg = float(np.degrees(theta[-1]))
                    target_deg = float(config.slew_angle_deg)
                    if abs(abs(achieved_deg) - abs(target_deg)) > 0.5:
                        print(
                            f"Warning: feedforward {method} achieves {achieved_deg:.2f} deg "
                            f"(target {target_deg:.2f} deg)."
                        )
            else:
                theta = np.array([])
                omega = np.array([])
                alpha = np.array([])

            torque_data[method] = {
                "time": time,
                "torque": torque_axis,
                "theta": theta,
                "omega": omega,
                "alpha": alpha,
                "maneuver_end": maneuver_end,
            }
            vibration_data[method] = {
                "time": time,
                "displacement": displacement,
                "acceleration": acceleration,
                "displacement_modal_raw": displacement_raw,
                "acceleration_modal_raw": acceleration_raw,
                "maneuver_end": maneuver_end,
            }
        else:
            td = _compute_torque_profile(config, method)
            vd = _simulate_modal_response(td["time"], td["torque"], config)
            maneuver_end = _infer_maneuver_end(td["time"], None, td["torque"], config.slew_duration_s)
            displacement_raw = np.array(vd["displacement"], dtype=float)
            acceleration_raw = np.array(vd["acceleration"], dtype=float)
            displacement, acceleration = _extract_vibration_signals(
                vd["time"], displacement_raw, config, acceleration_raw
            )
            torque_data[method] = {
                "time": td["time"],
                "torque": td["torque"],
                "theta": td["theta"],
                "omega": td["omega"],
                "alpha": td["alpha"],
                "maneuver_end": maneuver_end,
            }
            vibration_data[method] = {
                "time": vd["time"],
                "displacement": displacement,
                "acceleration": acceleration,
                "displacement_modal_raw": displacement_raw,
                "acceleration_modal_raw": acceleration_raw,
                "maneuver_end": maneuver_end,
            }

        psd_freq, psd_vals = _compute_psd(torque_data[method]["time"], torque_data[method]["torque"])
        torque_data[method]["psd_freq"] = psd_freq
        torque_data[method]["psd"] = psd_vals

    return torque_data, vibration_data


def _collect_feedback_data(
    config: MissionConfig,
    data_dir: Optional[str] = None,
    prefer_npz: bool = False,
) -> Dict[str, Dict[str, object]]:
    """Collect feedback vibration and torque data (combined preferred)."""
    data_dir = data_dir or os.path.dirname(__file__)
    axis = _normalize_axis(config.rotation_axis)
    feedback_data: Dict[str, Dict[str, object]] = {}

    if not prefer_npz:
        return feedback_data

    for method in METHODS:
        for controller in CONTROLLERS:
            npz_data = _load_pointing_npz(
                method,
                data_dir,
                controller=controller,
                mode="combined",
                generate_if_missing=False,
                allow_legacy=True,
            )
            if npz_data is None or not _npz_matches_config(npz_data, config):
                continue

            time = np.array(npz_data.get("time", []), dtype=float)
            torque_vec = npz_data.get("fb_torque")
            ff_vec = npz_data.get("ff_torque")
            total_vec = npz_data.get("total_torque")
            if ff_vec is not None and len(ff_vec) == 0:
                ff_vec = None
            if total_vec is not None and len(total_vec) == 0:
                total_vec = None
            if torque_vec is None or len(torque_vec) == 0:
                torque_vec = total_vec
            mode1_acc = npz_data.get("mode1_acc")
            mode2_acc = npz_data.get("mode2_acc")
            if mode1_acc is not None and len(mode1_acc) == 0:
                mode1_acc = None
            if mode2_acc is not None and len(mode2_acc) == 0:
                mode2_acc = None
            torque_axis = _project_axis_torque(torque_vec, axis) if torque_vec is not None else np.array([])
            torque_ff = _project_axis_torque(ff_vec, axis) if ff_vec is not None else np.array([])
            torque_total = _project_axis_torque(total_vec, axis) if total_vec is not None else np.array([])
            time, aligned = _align_series(
                time,
                torque_axis if len(torque_axis) > 0 else None,
                torque_ff if len(torque_ff) > 0 else None,
                torque_total if len(torque_total) > 0 else None,
                npz_data.get("mode1"),
                npz_data.get("mode2"),
                mode1_acc,
                mode2_acc,
            )
            torque_axis = aligned[0] if aligned[0] is not None else np.array([])
            torque_ff = aligned[1] if aligned[1] is not None else np.array([])
            torque_total = aligned[2] if aligned[2] is not None else np.array([])
            mode1 = aligned[3]
            mode2 = aligned[4]
            mode1_acc = aligned[5] if len(aligned) > 5 else np.array([])
            mode2_acc = aligned[6] if len(aligned) > 6 else np.array([])

            # Raw modal state signals (no filtering/detrending) for plotting/debug.
            displacement_raw = _combine_modal_displacement(mode1, mode2)
            displacement_raw = displacement_raw[: len(time)] if len(time) > 0 else displacement_raw
            acceleration_raw = np.array([])
            if mode1_acc is not None and len(mode1_acc) > 0:
                acceleration_raw = _combine_modal_displacement(mode1_acc, mode2_acc)
                acceleration_raw = acceleration_raw[: len(time)] if len(time) > 0 else acceleration_raw
            if len(acceleration_raw) == 0 and len(displacement_raw) > 0:
                acceleration_raw = _compute_acceleration_from_displacement(time, displacement_raw)

            # Legacy filtered signals retained for existing metrics/CSVs.
            displacement, acceleration = _extract_vibration_signals(
                time, displacement_raw, config, acceleration_raw
            )
            psd_freq, psd_vals = _compute_psd(time, torque_axis) if len(time) > 0 else (np.array([]), np.array([]))

            key = f"{method}_{controller}"
            feedback_data[key] = {
                "time": time,
                "displacement": displacement,
                "acceleration": acceleration,
                "displacement_modal_raw": displacement_raw,
                "acceleration_modal_raw": acceleration_raw,
                "mode1": np.array(mode1, dtype=float) if mode1 is not None else np.array([]),
                "mode2": np.array(mode2, dtype=float) if mode2 is not None else np.array([]),
                "mode1_acc": np.array(mode1_acc, dtype=float) if mode1_acc is not None else np.array([]),
                "mode2_acc": np.array(mode2_acc, dtype=float) if mode2_acc is not None else np.array([]),
                "torque": torque_axis,
                "torque_ff": torque_ff,
                "torque_total": torque_total,
                "rw_torque": npz_data.get("rw_torque"),
                "psd_freq": psd_freq,
                "psd": psd_vals,
                "method": npz_data.get("method", method),
                "controller": controller,
                "run_mode": npz_data.get("run_mode", "combined"),
            }

    return feedback_data


def run_feedforward_comparison(
    config: MissionConfig,
    out_dir: str,
    make_plots: bool = True,
    export_csv: bool = True,
    data_dir: Optional[str] = None,
    prefer_npz: bool = False,
) -> Dict[str, Dict[str, float]]:
    """Run feedforward comparison for all shaping methods."""
    results: Dict[str, Dict[str, float]] = {}
    torque_data, vibration_data = _collect_feedforward_data(
        config, data_dir=data_dir, prefer_npz=prefer_npz
    )

    for method in METHODS:
        metrics = _compute_feedforward_metrics(torque_data[method], vibration_data[method], config)
        results[method] = metrics

    if export_csv:
        _ensure_dir(out_dir)
        # Export metrics
        rows = []
        for method, m in results.items():
            rows.append([method, f"{m['rms_torque_nm']:.4f}", f"{m['peak_torque_nm']:.4f}",
                         f"{m['rms_vibration_mm']:.4f}", f"{m['peak_vibration_mm']:.4f}"])
        path = os.path.abspath(os.path.join(out_dir, "feedforward_metrics.csv"))
        _write_csv(path, ["method", "rms_torque_nm", "peak_torque_nm", "rms_vibration_mm", "peak_vibration_mm"], rows)
        print(f"Wrote feedforward metrics CSV: {path}")

        # Export torque profiles
        rows = []
        for method, td in torque_data.items():
            for i, (t, tau) in enumerate(zip(td["time"], td["torque"])):
                rows.append([f"{t:.6f}", method, f"{tau:.6e}"])
        path = os.path.abspath(os.path.join(out_dir, "feedforward_torque.csv"))
        _write_csv(path, ["time_s", "method", "torque_nm"], rows)
        print(f"Wrote feedforward torque CSV: {path}")

        # Export spectrum
        rows = []
        for method, td in torque_data.items():
            for f, p in zip(td.get("psd_freq", []), td.get("psd", [])):
                rows.append([f"{f:.6f}", method, f"{p:.6e}"])
        path = os.path.abspath(os.path.join(out_dir, "feedforward_spectrum.csv"))
        _write_csv(path, ["frequency_hz", "method", "psd_n2m2_per_hz"], rows)
        print(f"Wrote feedforward spectrum CSV: {path}")

    return results


# ===========================================================================
# Feedback Control Analysis
# ===========================================================================


def _build_flexible_plant_tf(
    inertia: np.ndarray,
    axis: int,
    modal_freqs_hz: List[float],
    modal_damping: List[float],
    modal_gains: List[float],
    output_scale: float = 1.0,
) -> signal.TransferFunction:
    """Build flexible plant transfer function from a coupled rigid flex state space model.

    Constructs a MIMO state space for the rigid body plus all flexible modes,
    then extracts the SISO body angle transfer function via ss2tf conversion.
    """
    axis_vec = np.zeros(3)
    axis_vec[axis] = 1.0
    inertia_axis = float(axis_vec @ inertia @ axis_vec)
    if modal_gains:
        lever_arms = [float(gain) * inertia_axis for gain in modal_gains]
    else:
        lever_arms = compute_mode_lever_arms(rotation_axis=axis_vec)
    if len(lever_arms) < len(modal_freqs_hz):
        lever_arms = _pad_list(lever_arms, len(modal_freqs_hz), 0.0)
    modal_masses = [float(FLEX_MODE_MASS)] * len(modal_freqs_hz)
    flex_ss = _build_coupled_flex_state_space(
        inertia,
        axis_vec,
        modal_freqs_hz,
        modal_damping,
        modal_masses,
        lever_arms,
        sigma_scale=4.0,
        camera_lever_arm_m=0.0,
    )
    ss_body = flex_ss["body"]
    num, den = signal.ss2tf(ss_body.A, ss_body.B, ss_body.C, ss_body.D)
    return signal.TransferFunction(np.squeeze(num) * float(output_scale), np.squeeze(den))


def _build_coupled_flex_state_space(
    inertia: np.ndarray,
    axis: np.ndarray,
    modal_freqs_hz: List[float],
    modal_damping: List[float],
    modal_masses: List[float],
    lever_arms: List[float],
    sigma_scale: float,
    camera_lever_arm_m: float,
) -> Dict[str, object]:
    """Build coupled rigid flex state space models for body and camera outputs.

    Assembles mass, damping, and stiffness matrices for the rigid angle
    plus N flexible modes, inverts the mass matrix, and forms a 2*(N+1)
    state vector [positions; velocities]. Three output variants are returned:
    body (rigid angle only), camera (rigid plus flex contribution scaled by
    lever arm), and full position states for per mode extraction.
    """
    axis = _normalize_axis(axis)
    inertia_axis = float(axis @ np.array(inertia, dtype=float) @ axis)
    hub_axis = float(axis @ HUB_INERTIA @ axis)
    base_inertia = inertia_axis if abs(inertia_axis - hub_axis) < 1e-6 else hub_axis
    n_modes = len(modal_freqs_hz)

    if n_modes == 0:
        a = np.array([[0.0, 1.0], [0.0, 0.0]])
        b = np.array([[0.0], [1.0 / inertia_axis]])
        c_body = np.array([[1.0 / sigma_scale, 0.0]])
        c_camera = c_body.copy()
        d = np.array([[0.0]])
        return {
            "body": signal.StateSpace(a, b, c_body, d),
            "camera": signal.StateSpace(a, b, c_camera, d),
            "positions": signal.StateSpace(a, b, np.array([[1.0, 0.0]]), d),
            "n_modes": 0,
        }

    zetas = _pad_list(modal_damping, n_modes, 0.01)
    masses = _pad_list(modal_masses, n_modes, FLEX_MODE_MASS)
    arms = _pad_list(lever_arms, n_modes, 0.0)

    m_mat = np.zeros((n_modes + 1, n_modes + 1))
    # Rigid body element: hub inertia plus rotational inertia of each appendage mass.
    m_mat[0, 0] = base_inertia + sum(m * r * r for m, r in zip(masses, arms))
    for idx, (m_i, r_i) in enumerate(zip(masses, arms), start=1):
        # Off diagonal coupling: appendage mass times lever arm links rigid and flex DOFs.
        m_mat[0, idx] = m_i * r_i
        m_mat[idx, 0] = m_i * r_i
        m_mat[idx, idx] = m_i

    # Damping and stiffness are diagonal (no cross coupling between modes).
    d_mat = np.zeros_like(m_mat)
    k_mat = np.zeros_like(m_mat)
    for idx, (freq_hz, zeta, m_i) in enumerate(zip(modal_freqs_hz, zetas, masses), start=1):
        omega = 2.0 * np.pi * float(freq_hz)
        d_mat[idx, idx] = 2.0 * float(zeta) * omega * m_i  # viscous damping coefficient
        k_mat[idx, idx] = omega**2 * m_i  # modal stiffness

    # Invert mass matrix to convert M*qddot + D*qdot + K*q = F into first order form.
    m_inv = np.linalg.inv(m_mat)
    n_state = n_modes + 1
    # State vector x = [q0, q1, ..., qN, q0dot, q1dot, ..., qNdot]
    a = np.zeros((2 * n_state, 2 * n_state))
    a[:n_state, n_state:] = np.eye(n_state)            # qdot = velocity states
    a[n_state:, :n_state] = -m_inv @ k_mat             # acceleration from stiffness
    a[n_state:, n_state:] = -m_inv @ d_mat             # acceleration from damping

    # Input: external torque applied to the rigid body DOF only.
    b = np.zeros((2 * n_state, 1))
    b[n_state:, 0] = (m_inv @ np.array([1.0] + [0.0] * n_modes)).ravel()

    # Position output extracts all generalized coordinates (rigid + modal).
    c_pos = np.zeros((n_state, 2 * n_state))
    c_pos[:, :n_state] = np.eye(n_state)

    # Body output: rigid body angle scaled from MRP to radians.
    c_body = np.zeros((1, 2 * n_state))
    c_body[0, 0] = 1.0 / sigma_scale

    # Camera output: rigid angle plus flex displacement projected through lever arm.
    c_camera = np.zeros((1, 2 * n_state))
    c_camera[0, 0] = 1.0 / sigma_scale
    if camera_lever_arm_m > 0:
        for idx in range(n_modes):
            c_camera[0, idx + 1] = 1.0 / (sigma_scale * camera_lever_arm_m)

    d = np.zeros((1, 1))
    return {
        "body": signal.StateSpace(a, b, c_body, d),
        "camera": signal.StateSpace(a, b, c_camera, d),
        "positions": signal.StateSpace(a, b, c_pos, np.zeros((n_state, 1))),
        "n_modes": n_modes,
    }


def _tf_add(
    num1: np.ndarray, den1: np.ndarray, num2: np.ndarray, den2: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """Add two transfer functions num1/den1 + num2/den2."""
    num1 = np.atleast_1d(np.squeeze(num1))
    den1 = np.atleast_1d(np.squeeze(den1))
    num2 = np.atleast_1d(np.squeeze(num2))
    den2 = np.atleast_1d(np.squeeze(den2))

    term1 = np.convolve(num1, den2)
    term2 = np.convolve(num2, den1)

    if len(term1) > len(term2):
        term2 = np.pad(term2, (len(term1) - len(term2), 0), mode="constant")
    elif len(term2) > len(term1):
        term1 = np.pad(term1, (len(term2) - len(term1), 0), mode="constant")

    num = term1 + term2
    den = np.convolve(den1, den2)
    return num, den


def _compute_stability_margins(L: np.ndarray, freqs: np.ndarray) -> Dict[str, float]:
    """Compute gain and phase margins from loop transfer function."""
    mag = np.abs(L)
    phase_deg = np.degrees(np.unwrap(np.angle(L)))

    gm_db = np.inf
    pm_deg = np.inf

    # Phase crossover: find where phase crosses -180 deg (gain margin frequency).
    idx = np.where((phase_deg[:-1] > -180.0) & (phase_deg[1:] <= -180.0))[0]
    if len(idx) > 0:
        i = idx[0]
        f1, f2 = freqs[i], freqs[i + 1]
        ph1, ph2 = phase_deg[i], phase_deg[i + 1]
        # Linear interpolation to find exact phase crossover frequency.
        if ph2 != ph1:
            f_pc = f1 + (f2 - f1) * (-180.0 - ph1) / (ph2 - ph1)
        else:
            f_pc = f1
        # Log interpolation of magnitude at the phase crossover frequency.
        if mag[i] > 0 and mag[i + 1] > 0 and f_pc > 0:
            logm1, logm2 = np.log10(mag[i]), np.log10(mag[i + 1])
            logf1, logf2 = np.log10(f1), np.log10(f2)
            logf_pc = np.log10(f_pc)
            if logf2 != logf1:
                logm_pc = logm1 + (logm2 - logm1) * (logf_pc - logf1) / (logf2 - logf1)
            else:
                logm_pc = logm1
            mag_pc = 10 ** logm_pc
        else:
            mag_pc = mag[i]
        gm_db = -20 * np.log10(mag_pc + 1e-12)

    # Gain crossover: find where magnitude crosses 0 dB (phase margin frequency).
    idx = np.where((mag[:-1] > 1.0) & (mag[1:] <= 1.0))[0]
    if len(idx) > 0:
        i = idx[0]
        f1, f2 = freqs[i], freqs[i + 1]
        mag1, mag2 = mag[i], mag[i + 1]
        # Log interpolation to find exact gain crossover frequency.
        if mag1 > 0 and mag2 > 0:
            logm1, logm2 = np.log10(mag1), np.log10(mag2)
            logf1, logf2 = np.log10(f1), np.log10(f2)
            if logm2 != logm1:
                logf_gc = logf1 + (0.0 - logm1) * (logf2 - logf1) / (logm2 - logm1)
                f_gc = 10 ** logf_gc
            else:
                f_gc = f1
        else:
            f_gc = f1

        if f_gc > 0 and f1 > 0 and f2 > 0:
            logf1, logf2 = np.log10(f1), np.log10(f2)
            logf_gc = np.log10(f_gc)
            # Log interpolation of phase at the gain crossover frequency.
            if logf2 != logf1:
                phase_gc = phase_deg[i] + (phase_deg[i + 1] - phase_deg[i]) * (logf_gc - logf1) / (logf2 - logf1)
            else:
                phase_gc = phase_deg[i]
        else:
            phase_gc = phase_deg[i]
        pm_deg = 180.0 + phase_gc

    return {"gain_margin_db": float(gm_db), "phase_margin_deg": float(pm_deg)}


def _compute_control_analysis(config: MissionConfig) -> Dict[str, object]:
    """Compute control system analysis including sensitivity and stability margins.

    Controller Design Philosophy:
        For flexible spacecraft, the control bandwidth must be placed well
        below the first structural mode to avoid exciting vibrations.

    Analysis computes rigid body S/T for reference and flexible loop
    margins (Nyquist, sensitivity, complementary sensitivity) for the
    actual sigma feedback plant:
        - Rigid body S/T for nominal pointing dynamics
        - Flexible loop margins for stability assessment
        - Modal excitation transfer: q(s) = G_q(s) * C(s) / (1 + L(s))

    This mission analysis uses standard PD only.
    """
    from basilisk_sim.feedback_control import (
        MRPFeedbackController,
    )

    axis = 2  # Z axis
    axis_vec = _normalize_axis(config.rotation_axis)
    I = float(axis_vec @ config.inertia @ axis_vec)
    sigma_scale = 4.0  # For small angles, sigma ~ theta/4

    # Frequency range
    freqs = np.logspace(-2, 2, 1000)
    omega = 2 * np.pi * freqs

    # Get first modal frequency for design
    if config.modal_freqs_hz:
        first_mode = min(config.modal_freqs_hz)
    else:
        first_mode = 1.0

    # =========================================================================
    # STANDARD PD CONTROLLER
    # LOWER bandwidth to avoid exciting modes
    # Rule: Bandwidth should be 1/4 to 1/6 of first modal frequency
    # For 0.4 Hz mode: bandwidth = 0.4/2.5 = 0.16 Hz
    # =========================================================================
    bandwidth_std = first_mode / 2.5  # 0.16 Hz for 0.4 Hz mode
    omega_bw_std = 2 * np.pi * bandwidth_std
    k_std = sigma_scale * I * omega_bw_std**2
    p_std = 2 * 0.9 * I * omega_bw_std

    # Create standard PD controller (no filtering)
    controller_std = MRPFeedbackController(
        inertia=config.inertia,
        K=k_std,
        P=p_std,
        Ki=-1.0
    )

    # Build rigid plant for MRP attitude output (torque to sigma)
    plant_rigid = signal.TransferFunction([1.0], [sigma_scale * I, 0.0, 0.0])

    # Controller in sigma domain: K + 4*P*s
    controller_std_tf = signal.TransferFunction([4.0 * p_std, k_std], [1.0])
    def _open_loop_tf(plant: signal.TransferFunction, controller: signal.TransferFunction) -> signal.TransferFunction:
        """Multiply plant and controller transfer functions to form the open loop."""
        plant_num = np.atleast_1d(np.squeeze(plant.num))
        plant_den = np.atleast_1d(np.squeeze(plant.den))
        ctrl_num = np.atleast_1d(np.squeeze(controller.num))
        ctrl_den = np.atleast_1d(np.squeeze(controller.den))
        return signal.TransferFunction(np.convolve(plant_num, ctrl_num), np.convolve(plant_den, ctrl_den))

    lever_arms = _infer_modal_lever_arms(config, axis_vec, I)
    modal_masses = _infer_modal_masses(config, len(config.modal_freqs_hz))
    flex_ss = _build_coupled_flex_state_space(
        config.inertia,
        axis_vec,
        config.modal_freqs_hz,
        config.modal_damping,
        modal_masses,
        lever_arms,
        sigma_scale=sigma_scale,
        camera_lever_arm_m=config.camera_lever_arm_m,
    )


    # Evaluate frequency responses
    _, plant_rigid_resp = signal.freqresp(plant_rigid, omega)
    _, plant_flex_body = signal.freqresp(flex_ss["body"], omega)

    _, c_std_resp = signal.freqresp(controller_std_tf, omega)
    # Rate feedback path (omega noise to torque).
    c_rate_std_resp = p_std * np.ones_like(omega, dtype=complex)

    # Open loop rigid body (torque to sigma) and sensitivity
    l_std_rigid = plant_rigid_resp * c_std_resp

    s_std = 1 / (1 + l_std_rigid)
    t_std = l_std_rigid / (1 + l_std_rigid)

    # Open loop using flexible sigma output
    l_std_flex = plant_flex_body * c_std_resp

    s_std_flex = 1 / (1 + l_std_flex)
    t_std_flex = l_std_flex / (1 + l_std_flex)

    disturbance_body = {
        "standard_pd": plant_flex_body / (1 + l_std_flex),
    }

    # Stability margins
    margins = {
        "standard_pd": _compute_stability_margins(l_std_flex, freqs),
    }

    # Modal excitation transfer: q(s) = G_q(s) * C(s) / (1 + L(s))
    modal_response: Dict[str, List[np.ndarray]] = {name: [] for name in CONTROLLERS}
    modal_excitation_db: Dict[str, List[float]] = {name: [] for name in CONTROLLERS}
    controllers_tf = {
        "standard_pd": controller_std_tf,
    }
    loops = {"standard_pd": l_std_flex}

    pos_ss = flex_ss["positions"]
    a_mat = pos_ss.A
    b_mat = pos_ss.B
    for mode_idx, f_mode in enumerate(config.modal_freqs_hz):
        c_q = np.zeros((1, a_mat.shape[0]))
        if mode_idx + 1 < a_mat.shape[0]:
            c_q[0, mode_idx + 1] = 1.0
        ss_q = signal.StateSpace(a_mat, b_mat, c_q, np.zeros((1, 1)))
        _, gq_resp = signal.freqresp(ss_q, omega)
        for name in CONTROLLERS:
            _, c_resp = signal.freqresp(controllers_tf[name], omega)
            u_resp = c_resp / (1.0 + loops[name])
            modal_resp = gq_resp * u_resp
            modal_response[name].append(modal_resp)
            idx = np.argmin(np.abs(freqs - f_mode))
            modal_excitation_db[name].append(20 * np.log10(np.abs(modal_resp[idx]) + 1e-12))


    return {
        "freqs": freqs,
        "omega": omega,
        "L": {"standard_pd": l_std_rigid},
        "S": {"standard_pd": s_std},
        "T": {"standard_pd": t_std},
        "L_flex": {"standard_pd": l_std_flex},
        "S_flex": {"standard_pd": s_std_flex},
        "T_flex": {"standard_pd": t_std_flex},
        "plant_rigid": plant_rigid_resp,
        "plant_flex_body": plant_flex_body,
        "controller_resp": {
            "standard_pd": c_std_resp,
        },
        "rate_path_resp": {
            "standard_pd": c_rate_std_resp,
        },
        "disturbance_body": disturbance_body,
        "margins": margins,
        "gains": {"K": k_std, "P": p_std},
        "modal_response": modal_response,
        "modal_excitation_db": modal_excitation_db,
    }


def run_control_analysis(
    config: MissionConfig,
    out_dir: str,
    make_plots: bool = True,
    export_csv: bool = True,
) -> Dict[str, Dict[str, float]]:
    """Run control system analysis."""
    ctrl = _compute_control_analysis(config)

    results = {}
    for name in CONTROLLERS:
        m = ctrl["margins"][name]
        results[name] = {
            "gain_margin_db": m["gain_margin_db"],
            "phase_margin_deg": m["phase_margin_deg"],
        }

    if export_csv:
        _ensure_dir(out_dir)
        # Export metrics
        rows = []
        for name, m in results.items():
            rows.append([name, f"{m['gain_margin_db']:.2f}", f"{m['phase_margin_deg']:.2f}"])
        path = os.path.abspath(os.path.join(out_dir, "control_metrics.csv"))
        _write_csv(path, ["controller", "gain_margin_db", "phase_margin_deg"], rows)
        print(f"Wrote control metrics CSV: {path}")

        # Export curves
        rows = []
        freqs = ctrl["freqs"]
        for i, f in enumerate(freqs):
            for name in CONTROLLERS:
                s_mag = 20 * np.log10(np.abs(ctrl["S"][name][i]) + 1e-12)
                t_mag = 20 * np.log10(np.abs(ctrl["T"][name][i]) + 1e-12)
                rows.append([f"{f:.6f}", name, f"{s_mag:.4f}", f"{t_mag:.4f}"])
        path = os.path.abspath(os.path.join(out_dir, "control_curves.csv"))
        _write_csv(path, ["frequency_hz", "controller", "S_mag_db", "T_mag_db"], rows)
        print(f"Wrote control curves CSV: {path}")

    return results


# ===========================================================================
# NPZ File Loading (for Basilisk simulation data)
# ===========================================================================


def _find_npz(
    method: str,
    data_dir: str,
    controller: Optional[str] = None,
    mode: str = "combined",
    allow_legacy: bool = False,
) -> Optional[str]:
    """Find NPZ file for given method and controller.

    Searches data_dir first, then current working directory.
    """
    candidates: List[str] = []
    if mode == "combined":
        if controller:
            candidates.append(f"vizard_demo_{method}_{controller}.npz")
        if controller is None or allow_legacy:
            candidates.append(f"vizard_demo_{method}.npz")
    elif mode == "fb_only":
        if controller:
            candidates.append(f"vizard_demo_{method}_{controller}_fb_only.npz")
    elif mode == "ff_only":
        candidates.append(f"vizard_demo_{method}_ff_only.npz")

    script_dir = os.path.dirname(__file__)
    parent_dir = os.path.abspath(os.path.join(script_dir, ".."))
    output_cache_dir = os.path.abspath(os.path.join(parent_dir, "output", "cache"))
    output_dir = os.path.abspath(os.path.join(parent_dir, "output"))
    search_dirs = [
        data_dir,
        os.getcwd(),
        script_dir,
        parent_dir,
        output_cache_dir,
        output_dir,
        os.path.abspath(os.path.join(str(data_dir), "..")) if data_dir else None,
    ]
    seen_dirs = set()
    for search_dir in search_dirs:
        if not search_dir or search_dir in seen_dirs:
            continue
        seen_dirs.add(search_dir)
        for filename in candidates:
            path = os.path.join(search_dir, filename)
            if os.path.exists(path):
                return path
    return None


def _maybe_generate_npz(
    method: str,
    data_dir: str,
    controller: Optional[str] = None,
    mode: str = "combined",
) -> bool:
    """Try to generate NPZ file by running run_vizard_demo.py."""
    script_dir = os.path.dirname(__file__)
    script_path = os.path.join(script_dir, "run_vizard_demo.py")

    if not os.path.exists(script_path):
        return False

    try:
        cmd = [sys.executable, script_path, method]
        if controller:
            cmd.extend(["--controller", controller])
        if mode != "combined":
            cmd.extend(["--mode", mode])
        if data_dir:
            cmd.extend(["--output-dir", data_dir])
        subprocess.run(
            cmd,
            check=True,
            cwd=script_dir,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False


def _infer_controller_from_control_mode(control_mode: Optional[List[object]]) -> Optional[str]:
    """Infer controller type from control mode strings."""
    if not control_mode:
        return None
    modes = [str(mode) for mode in control_mode]
    if any("FB(FILT)" in mode or "Filtered" in mode for mode in modes):
        return "filtered_pd"
    if any("FB(MRP)" in mode or "MRP" in mode for mode in modes):
        return "standard_pd"
    return None


def _load_pointing_npz(
    method: str,
    data_dir: str,
    controller: Optional[str] = None,
    mode: str = "combined",
    generate_if_missing: bool = False,
    allow_legacy: bool = False,
) -> Optional[Dict[str, object]]:
    """Load pointing data from NPZ file."""
    npz_path = _find_npz(method, data_dir, controller=controller, mode=mode, allow_legacy=allow_legacy)
    if npz_path is None and generate_if_missing:
        if _maybe_generate_npz(method, data_dir, controller=controller, mode=mode):
            npz_path = _find_npz(method, data_dir, controller=controller, mode=mode, allow_legacy=allow_legacy)

    if npz_path is None:
        return None

    try:
        data = np.load(npz_path, allow_pickle=True)
    except Exception:
        return None

    time = np.array(data["time"], dtype=float) if "time" in data else np.array([])
    if "mode1_signed" in data:
        mode1 = np.array(data["mode1_signed"], dtype=float)
    elif "mode1" in data:
        mode1 = np.array(data["mode1"], dtype=float)
    else:
        mode1 = np.zeros_like(time)

    if "mode2_signed" in data:
        mode2 = np.array(data["mode2_signed"], dtype=float)
    elif "mode2" in data:
        mode2 = np.array(data["mode2"], dtype=float)
    else:
        mode2 = np.zeros_like(time)
    if "mode1_acc_signed" in data:
        mode1_acc = np.array(data["mode1_acc_signed"], dtype=float)
    elif "mode1_acc" in data:
        mode1_acc = np.array(data["mode1_acc"], dtype=float)
    else:
        mode1_acc = np.array([])

    if "mode2_acc_signed" in data:
        mode2_acc = np.array(data["mode2_acc_signed"], dtype=float)
    elif "mode2_acc" in data:
        mode2_acc = np.array(data["mode2_acc"], dtype=float)
    else:
        mode2_acc = np.array([])
    sigma = np.array(data["sigma"], dtype=float) if "sigma" in data else None
    omega = np.array(data["omega"], dtype=float) if "omega" in data else None
    control_mode = data["control_mode"].tolist() if "control_mode" in data else []
    target_sigma = np.array(data["target_sigma"], dtype=float) if "target_sigma" in data else np.zeros(3)
    controller_in_file = str(data["controller"]) if "controller" in data else None
    run_mode = str(data["run_mode"]) if "run_mode" in data else mode
    slew_angle_deg = float(data["slew_angle_deg"]) if "slew_angle_deg" in data else None
    slew_duration_s = float(data["slew_duration_s"]) if "slew_duration_s" in data else None
    camera_error_deg = np.array(data["camera_error_deg"], dtype=float) if "camera_error_deg" in data else None
    camera_body = np.array(data["camera_body"], dtype=float) if "camera_body" in data else None
    comet_direction = np.array(data["comet_direction"], dtype=float) if "comet_direction" in data else None

    modal_freqs_hz = np.array(data["modal_freqs_hz"], dtype=float) if "modal_freqs_hz" in data else None
    modal_damping = np.array(data["modal_damping"], dtype=float) if "modal_damping" in data else None
    modal_gains = np.array(data["modal_gains"], dtype=float) if "modal_gains" in data else None
    control_filter_cutoff_hz = float(data["control_filter_cutoff_hz"]) if "control_filter_cutoff_hz" in data else None
    inertia_control = np.array(data["inertia_control"], dtype=float) if "inertia_control" in data else None

    ff_torque = np.array(data["ff_torque"], dtype=float) if "ff_torque" in data else None
    fb_torque = np.array(data["fb_torque"], dtype=float) if "fb_torque" in data else None
    total_torque = np.array(data["total_torque"], dtype=float) if "total_torque" in data else None
    rw_torque = np.array(data["rw_torque"], dtype=float) if "rw_torque" in data else None

    controller_used = controller or controller_in_file or _infer_controller_from_control_mode(control_mode)

    return {
        "time": time,
        "mode1": mode1,
        "mode2": mode2,
        "mode1_acc": mode1_acc,
        "mode2_acc": mode2_acc,
        "sigma": sigma,
        "omega": omega,
        "control_mode": control_mode,
        "target_sigma": target_sigma,
        "controller": controller_used,
        "run_mode": run_mode,
        "slew_angle_deg": slew_angle_deg,
        "slew_duration_s": slew_duration_s,
        "camera_error_deg": camera_error_deg,
        "camera_body": camera_body,
        "comet_direction": comet_direction,
        "ff_torque": ff_torque,
        "fb_torque": fb_torque,
        "total_torque": total_torque,
        "rw_torque": rw_torque,
        "modal_freqs_hz": modal_freqs_hz,
        "modal_damping": modal_damping,
        "modal_gains": modal_gains,
        "control_filter_cutoff_hz": control_filter_cutoff_hz,
        "inertia_control": inertia_control,
    }


def _load_all_pointing_data(
    data_dir: str,
    config: Optional[MissionConfig] = None,
    generate_if_missing: bool = False,
) -> Dict[str, Dict[str, object]]:
    """Load pointing data for all methods and controllers."""
    data_dir = data_dir or os.path.dirname(__file__)
    pointing: Dict[str, Dict[str, object]] = {}

    for method in METHODS:
        method_data: Dict[str, object] = {}
        for controller in CONTROLLERS:
            npz_data = _load_pointing_npz(
                method,
                data_dir,
                controller=controller,
                generate_if_missing=generate_if_missing,
                allow_legacy=False,
            )
            if npz_data is not None and config is not None and not _npz_matches_config(npz_data, config):
                npz_data = None
                if generate_if_missing and _maybe_generate_npz(
                    method, data_dir, controller=controller, mode="combined"
                ):
                    npz_data = _load_pointing_npz(
                        method,
                        data_dir,
                        controller=controller,
                        generate_if_missing=False,
                        allow_legacy=False,
                    )
                    if npz_data is not None and not _npz_matches_config(npz_data, config):
                        npz_data = None
            if npz_data is not None:
                method_data[controller] = npz_data

        if not method_data:
            # Try legacy format
            legacy = _load_pointing_npz(
                method,
                data_dir,
                controller=None,
                generate_if_missing=generate_if_missing,
                allow_legacy=True,
            )
            if legacy is not None and config is not None and not _npz_matches_config(legacy, config):
                legacy = None
                if generate_if_missing and _maybe_generate_npz(method, data_dir, mode="combined"):
                    legacy = _load_pointing_npz(
                        method,
                        data_dir,
                        controller=None,
                        generate_if_missing=False,
                        allow_legacy=True,
                    )
                    if legacy is not None and not _npz_matches_config(legacy, config):
                        legacy = None
            if legacy is not None:
                ctrl = legacy.get("controller") or "standard_pd"
                method_data[ctrl] = legacy

        if method_data:
            pointing[method] = method_data

    return pointing


# ===========================================================================
# Pointing Summary
# ===========================================================================


def _compute_pointing_error(sigma: np.ndarray, target_sigma: np.ndarray) -> np.ndarray:
    """Compute scalar pointing error in degrees from MRP attitude vectors."""
    if sigma is None or len(sigma) == 0:
        return np.array([])
    sigma = np.atleast_2d(np.array(sigma, dtype=float))
    target = np.array(target_sigma, dtype=float).flatten()
    errors = np.zeros(len(sigma))
    for i, sigma_row in enumerate(sigma):
        sigma_error = _mrp_subtract(sigma_row, target)
        errors[i] = np.degrees(4 * np.arctan(np.linalg.norm(sigma_error)))
    return errors


def _extract_pointing_error(
    data: Dict[str, object],
    config: Optional[MissionConfig] = None,
) -> np.ndarray:
    """Return pointing error time series, optionally including flex induced jitter.

    Uses linear combination (base + flex) instead of RSS to match the transfer
    function formulation: y_camera = theta + q/L. This preserves resonance peaks
    that would otherwise be dampened by RSS combination.
    """
    camera_error = data.get("camera_error_deg")
    if camera_error is not None and len(camera_error) > 0:
        base_error = np.array(camera_error, dtype=float)
    else:
        sigma = data.get("sigma")
        target = data.get("target_sigma", np.zeros(3))
        base_error = _compute_pointing_error(sigma, target)

    if config is None:
        return base_error

    lever_arm = float(config.camera_lever_arm_m or 0.0)
    if lever_arm <= 0:
        return base_error

    mode1 = data.get("mode1")
    mode2 = data.get("mode2")
    if mode1 is None and mode2 is None:
        return base_error

    displacement = _combine_modal_displacement(
        np.array(mode1, dtype=float) if mode1 is not None else np.array([]),
        np.array(mode2, dtype=float) if mode2 is not None else np.array([]),
    )
    if len(displacement) == 0:
        return base_error

    # Convert modal displacement to pointing angle (keep sign for proper phase)
    flex_angle_deg = np.degrees(displacement / lever_arm)

    n = min(len(base_error), len(flex_angle_deg))
    if n == 0:
        return base_error if len(base_error) else np.abs(flex_angle_deg)

    base = base_error[:n]
    flex = flex_angle_deg[:n]
    # Linear combination to match transfer function: y = theta + q/L
    # This preserves resonance peaks that RSS combination would dampen
    total = base + flex
    return np.abs(total)


def run_pointing_summary(
    config: MissionConfig,
    out_dir: str,
    data_dir: Optional[str] = None,
    make_plots: bool = True,
    export_csv: bool = True,
    generate_pointing: bool = False,
) -> Dict[str, Dict[str, float]]:
    """Run pointing performance summary."""
    data_dir = data_dir or os.path.dirname(__file__)
    pointing_data = _load_all_pointing_data(
        data_dir,
        config=config,
        generate_if_missing=generate_pointing,
    )

    results: Dict[str, Dict[str, float]] = {}

    for method, method_data in pointing_data.items():
        # Check if this is legacy single controller data
        is_legacy = len(method_data) == 1
        for controller, data in method_data.items():
            # Use just method name for legacy format, combined for multi controller
            key = method if is_legacy else f"{method}_{controller}"
            time = np.array(data.get("time", np.array([])), dtype=float)
            mode1 = data.get("mode1", np.array([]))
            mode2 = data.get("mode2", np.array([]))
            maneuver_end = _infer_maneuver_end(
                time,
                data.get("control_mode"),
                None,
                config.slew_duration_s,
            )
            maneuver_end_idx = int(np.searchsorted(time, maneuver_end)) if len(time) > 1 else len(time)
            min_residual_samples = 100

            # Vibration metrics
            if len(mode1) > 0 and len(time) > 0:
                total_vib = _combine_modal_displacement(mode1, mode2)
                vib_disp, _ = _extract_vibration_signals(time, total_vib, config)
                if maneuver_end_idx < len(vib_disp) - min_residual_samples:
                    residual = vib_disp[maneuver_end_idx:]
                elif len(vib_disp) > min_residual_samples:
                    residual = vib_disp[int(0.9 * len(vib_disp)):]
                else:
                    residual = vib_disp
                rms_vib = float(np.sqrt(np.mean(residual**2))) * 1000 if len(residual) > 0 else 0.0
            else:
                rms_vib = 0.0

            # Pointing error metrics
            error = _extract_pointing_error(data, config=config)
            if len(error) > 0 and len(time) > 0:
                if maneuver_end_idx < len(error) - min_residual_samples:
                    residual_error = error[maneuver_end_idx:]
                elif len(error) > min_residual_samples:
                    residual_error = error[int(0.9 * len(error)):]
                else:
                    residual_error = error
                rms_error = float(np.sqrt(np.mean(residual_error**2))) if len(residual_error) > 0 else 0.0
            else:
                rms_error = 0.0

            results[key] = {
                "rms_vibration_mm": rms_vib,
                "rms_pointing_error_deg": rms_error,
            }

    if export_csv:
        _ensure_dir(out_dir)
        rows = []
        for key, m in results.items():
            rows.append([key, f"{m['rms_vibration_mm']:.4f}", f"{m['rms_pointing_error_deg']:.6f}"])
        path = os.path.abspath(os.path.join(out_dir, "pointing_metrics.csv"))
        _write_csv(path, ["method", "rms_vibration_mm", "rms_pointing_error_deg"], rows)
        print(f"Wrote pointing metrics CSV: {path}")

    return results


# ===========================================================================
# PSD Data Building
# ===========================================================================


def _build_mission_psd_data(
    config: MissionConfig,
    data_dir: str,
    generate_if_missing: bool = False,
) -> Dict[str, Dict[str, Dict[str, object]]]:
    """Build vibration PSD data structure from combined NPZ outputs."""
    mission_psd: Dict[str, Dict[str, Dict[str, object]]] = {}

    for method in METHODS:
        mission_psd[method] = {}
        for controller in CONTROLLERS:
            npz_data = _load_pointing_npz(
                method,
                data_dir,
                controller=controller,
                mode="combined",
                generate_if_missing=generate_if_missing,
                allow_legacy=True,
            )
            if npz_data is not None and not _npz_matches_config(npz_data, config):
                npz_data = None
                if generate_if_missing and _maybe_generate_npz(
                    method, data_dir, controller=controller, mode="combined"
                ):
                    npz_data = _load_pointing_npz(
                        method,
                        data_dir,
                        controller=controller,
                        mode="combined",
                        generate_if_missing=False,
                        allow_legacy=True,
                    )
                    if npz_data is not None and not _npz_matches_config(npz_data, config):
                        npz_data = None

            if npz_data is None:
                psd_freq, psd_vals = np.array([]), np.array([])
            else:
                time = np.array(npz_data.get("time", []), dtype=float)
                mode1 = np.array(npz_data.get("mode1", []), dtype=float)
                mode2 = np.array(npz_data.get("mode2", []), dtype=float)
                time, aligned = _align_series(time, mode1, mode2)
                mode1 = aligned[0]
                mode2 = aligned[1]
                displacement = _combine_modal_displacement(mode1, mode2)
                displacement = displacement[: len(time)] if len(time) > 0 else displacement
                displacement, _ = _extract_vibration_signals(time, displacement, config)
                psd_freq, psd_vals = _compute_psd(time, displacement) if len(time) > 0 else (np.array([]), np.array([]))

            mission_psd[method][controller] = {
                "psd_freq": psd_freq,
                "psd": psd_vals,
            }

    return mission_psd


# ===========================================================================
# Plotting Functions
# ===========================================================================


def _plot_vibration_comparison(
    feedforward_vibration: Dict[str, Dict[str, object]],
    feedback_vibration: Dict[str, Dict[str, object]],
    config: MissionConfig,
    out_dir: str,
) -> Optional[str]:
    """Plot vibration comparison from raw modal states (no filtering/detrending)."""
    if not feedforward_vibration and not feedback_vibration:
        return None

    plt.rcParams.update({
        "font.size": 10,
        "axes.labelsize": 11,
        "axes.titlesize": 12,
        "legend.fontsize": 9,
    })

    fig, (ax_disp, ax_acc) = plt.subplots(1, 2, figsize=(16, 5.8), sharex=True)

    maneuver_end = None
    cutoff_hz = _get_vibration_highpass_hz(config)

    plot_feedback_only = bool(feedback_vibration)

    def _compute_total_and_post_rms(
        time_arr: np.ndarray,
        signal_arr: np.ndarray,
        maneuver_end_time: Optional[float],
    ) -> Tuple[float, float]:
        """Return RMS over the full timeline and the post slew timeline."""
        if len(signal_arr) == 0:
            return 0.0, 0.0
        rms_total = float(np.sqrt(np.mean(signal_arr**2)))
        if maneuver_end_time is None or len(time_arr) == 0:
            return rms_total, rms_total
        post_mask = np.array(time_arr, dtype=float) >= float(maneuver_end_time)
        if np.any(post_mask):
            post_vals = signal_arr[post_mask]
        else:
            post_vals = signal_arr[int(0.9 * len(signal_arr)):] if len(signal_arr) > 10 else signal_arr
        rms_post = float(np.sqrt(np.mean(post_vals**2))) if len(post_vals) > 0 else rms_total
        return rms_total, rms_post

    plot_entries = []
    if not plot_feedback_only:
        # Plot FEEDFORWARD data (solid lines) only when feedback data is missing
        for method in METHODS:
            data = feedforward_vibration.get(method)
            if not data:
                continue
            time = data.get("time", np.array([]))
            disp = data.get("displacement", np.array([]))
            acc = data.get("acceleration", np.array([]))

            if len(time) == 0:
                continue

            if maneuver_end is None:
                maneuver_end = data.get("maneuver_end", 30.0)

            disp_raw = data.get("displacement_modal_raw")
            acc_raw = data.get("acceleration_modal_raw")
            if disp_raw is not None and len(disp_raw) > 0:
                disp = np.array(disp_raw, dtype=float)
            if acc_raw is not None and len(acc_raw) > 0:
                acc = np.array(acc_raw, dtype=float)
            elif len(disp) > 0:
                acc = _compute_acceleration_from_displacement(np.array(time, dtype=float), np.array(disp, dtype=float))

            disp_arr = np.array(disp, dtype=float)
            acc_arr = np.array(acc, dtype=float)
            if len(time) > 2 and len(disp_arr) > 2:
                disp_arr = _highpass_filter(disp_arr, np.array(time, dtype=float), cutoff_hz)
            if len(time) > 2 and len(acc_arr) > 2:
                acc_arr = _highpass_filter(acc_arr, np.array(time, dtype=float), cutoff_hz)
            disp_mm = disp_arr * 1000.0
            acc_mm = acc_arr * 1000.0
            disp_rms_total_mm, disp_rms_post_mm = _compute_total_and_post_rms(
                np.array(time, dtype=float), disp_mm, maneuver_end
            )
            acc_rms_total_mms2, acc_rms_post_mms2 = _compute_total_and_post_rms(
                np.array(time, dtype=float), acc_mm, maneuver_end
            )

            label = f"FF: {METHOD_LABELS.get(method, method)}"
            plot_entries.append({
                "time": time,
                "disp": disp_mm,
                "acc": acc_mm,
                "label": label,
                "disp_rms_total_mm": disp_rms_total_mm,
                "disp_rms_post_mm": disp_rms_post_mm,
                "acc_rms_total_mms2": acc_rms_total_mms2,
                "acc_rms_post_mms2": acc_rms_post_mms2,
                "linestyle": "-",
                "color": METHOD_COLORS.get(method, "#555555"),
            })

    # Plot FEEDBACK data from the combined feedforward and feedback run
    for method in METHODS:
        for controller in CONTROLLERS:
            key = f"{method}_{controller}"
            data = feedback_vibration.get(key)
            if not data:
                continue
            time = data.get("time", np.array([]))
            disp = data.get("displacement", np.array([]))
            acc = data.get("acceleration", np.array([]))
            run_mode = str(data.get("run_mode", "")).lower()

            if len(time) == 0:
                continue
            if run_mode and run_mode != "combined":
                continue
            if maneuver_end is None:
                maneuver_end = data.get("maneuver_end", config.slew_duration_s)

            disp_raw = data.get("displacement_modal_raw")
            acc_raw = data.get("acceleration_modal_raw")
            if disp_raw is not None and len(disp_raw) > 0:
                disp = np.array(disp_raw, dtype=float)
            if acc_raw is not None and len(acc_raw) > 0:
                acc = np.array(acc_raw, dtype=float)
            elif len(disp) > 0:
                acc = _compute_acceleration_from_displacement(np.array(time, dtype=float), np.array(disp, dtype=float))

            disp_arr = np.array(disp, dtype=float)
            acc_arr = np.array(acc, dtype=float)
            if len(time) > 2 and len(disp_arr) > 2:
                disp_arr = _highpass_filter(disp_arr, np.array(time, dtype=float), cutoff_hz)
            if len(time) > 2 and len(acc_arr) > 2:
                acc_arr = _highpass_filter(acc_arr, np.array(time, dtype=float), cutoff_hz)
            disp_mm = disp_arr * 1000.0
            acc_mm = acc_arr * 1000.0
            disp_rms_total_mm, disp_rms_post_mm = _compute_total_and_post_rms(
                np.array(time, dtype=float), disp_mm, maneuver_end
            )
            acc_rms_total_mms2, acc_rms_post_mms2 = _compute_total_and_post_rms(
                np.array(time, dtype=float), acc_mm, maneuver_end
            )

            label = _combo_label(method, controller)
            plot_entries.append({
                "time": time,
                "disp": disp_mm,
                "acc": acc_mm,
                "label": label,
                "disp_rms_total_mm": disp_rms_total_mm,
                "disp_rms_post_mm": disp_rms_post_mm,
                "acc_rms_total_mms2": acc_rms_total_mms2,
                "acc_rms_post_mms2": acc_rms_post_mms2,
                "linestyle": "-",
                "color": _combo_color(method, controller),
            })

    if not plot_entries:
        plt.close(fig)
        return None

    time_min = min(float(np.min(np.array(entry["time"], dtype=float))) for entry in plot_entries)
    time_max = max(float(np.max(np.array(entry["time"], dtype=float))) for entry in plot_entries)

    if maneuver_end is not None and time_max > float(maneuver_end):
        post_color = "#e8f5df"
        ax_disp.axvspan(float(maneuver_end), time_max, color=post_color, alpha=0.30, zorder=0, label="Post slew region")
        ax_acc.axvspan(float(maneuver_end), time_max, color=post_color, alpha=0.30, zorder=0)

    for entry in plot_entries:
        color = entry.get("color", "#555555")
        ax_disp.plot(
            entry["time"],
            entry["disp"],
            color=color,
            label=(
                f'{entry["label"]} '
                f'(RMS total={entry["disp_rms_total_mm"]:.3f} mm, '
                f'post-slew={entry["disp_rms_post_mm"]:.3f} mm)'
            ),
            linewidth=1.5,
            linestyle=entry["linestyle"],
        )
        ax_acc.plot(
            entry["time"],
            entry["acc"],
            color=color,
            label=(
                f'{entry["label"]} '
                f'(RMS total={entry["acc_rms_total_mms2"]:.3f} mm/s^2, '
                f'post-slew={entry["acc_rms_post_mms2"]:.3f} mm/s^2)'
            ),
            linewidth=1.5,
            linestyle=entry["linestyle"],
        )

    # Mark maneuver end
    if maneuver_end is not None:
        ax_disp.axvline(maneuver_end, color="gray", linestyle=":", linewidth=1.5, alpha=0.7,
                        label=f"Maneuver end ({maneuver_end:.0f}s)")
        ax_acc.axvline(maneuver_end, color="gray", linestyle=":", linewidth=1.5, alpha=0.7)

    # Displacement subplot
    ax_disp.set_title("Modal Displacement Response", fontweight="bold")
    ax_disp.set_ylabel("Displacement (mm)")
    ax_disp.grid(True, alpha=0.3)
    ax_disp.legend(loc="upper right", fontsize=8)
    ax_disp.axhline(0, color="black", linewidth=0.5, alpha=0.3)

    # Acceleration subplot
    ax_acc.set_title("Modal Acceleration Response", fontweight="bold")
    ax_acc.set_ylabel(r"Acceleration (mm/s$^2$)")
    ax_acc.grid(True, alpha=0.3)
    ax_acc.legend(loc="upper right", fontsize=8)
    ax_acc.axhline(0, color="black", linewidth=0.5, alpha=0.3)

    ax_disp.set_xlabel("Time (s)")
    ax_acc.set_xlabel("Time (s)")

    plt.tight_layout()
    _ensure_dir(out_dir)
    plot_path = os.path.abspath(os.path.join(out_dir, "mission_vibration.png"))
    plt.savefig(plot_path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return plot_path


def _plot_modal_acceleration_psd(
    feedback_vibration: Dict[str, Dict[str, object]],
    config: MissionConfig,
    out_dir: str,
) -> Optional[str]:
    """Plot high resolution PSD of combined modal acceleration."""
    if not feedback_vibration:
        return None

    plt.rcParams.update({
        "font.size": 12,
        "axes.labelsize": 13,
        "axes.titlesize": 14,
        "legend.fontsize": 10,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
    })

    fig, ax = plt.subplots(1, 1, figsize=(12, 6))
    plotted = False
    psd_min_db = None
    psd_max_db = None

    for method in METHODS:
        for controller in CONTROLLERS:
            key = f"{method}_{controller}"
            data = feedback_vibration.get(key)
            if not data:
                continue
            run_mode = str(data.get("run_mode", "")).lower()
            if run_mode and run_mode != "combined":
                continue

            time = np.array(data.get("time", np.array([])), dtype=float)
            mode1_acc = np.array(data.get("mode1_acc", np.array([])), dtype=float)
            mode2_acc = np.array(data.get("mode2_acc", np.array([])), dtype=float)
            combined_acc = _combine_modal_displacement(mode1_acc, mode2_acc)
            if len(combined_acc) == 0:
                combined_acc = np.array(data.get("acceleration_modal_raw", np.array([])), dtype=float)
            time, aligned = _align_series(time, combined_acc)
            combined_acc = aligned[0]
            if len(time) == 0 or len(combined_acc) == 0:
                continue

            maneuver_end = data.get("maneuver_end", config.slew_duration_s)
            try:
                maneuver_end = float(maneuver_end)
            except (TypeError, ValueError):
                maneuver_end = float(config.slew_duration_s)

            # Add a short guard after slew end to avoid edge transients dominating leakage.
            post_guard_s = 0.5
            post_mask = time >= (maneuver_end + post_guard_s)
            if np.count_nonzero(post_mask) >= 32:
                time_use = np.array(time[post_mask], dtype=float)
                acc_use = np.array(combined_acc[post_mask], dtype=float)
            else:
                # Fallback when post slew window is too short for PSD estimation.
                time_use = time
                acc_use = combined_acc

            freq, psd = _compute_psd_high_resolution(
                time_use,
                acc_use,
                zero_pad_factor=16,
                window=("tukey", 0.12),
                detrend="linear",
            )
            mask = (freq > 0) & (freq <= 10.0) & np.isfinite(psd) & (psd > 0)
            if not np.any(mask):
                continue
            freq_f = freq[mask]
            psd_db = 10.0 * np.log10(psd[mask])
            psd_min_db = float(np.min(psd_db)) if psd_min_db is None else min(psd_min_db, float(np.min(psd_db)))
            psd_max_db = float(np.max(psd_db)) if psd_max_db is None else max(psd_max_db, float(np.max(psd_db)))

            ax.semilogx(
                freq_f,
                psd_db,
                color=_combo_color(method, controller),
                linewidth=1.8,
                linestyle="-",
                label=_combo_label(method, controller),
            )
            plotted = True

    if not plotted:
        plt.close(fig)
        return None

    for f_mode in config.modal_freqs_hz:
        ax.axvline(f_mode, color="gray", linestyle=":", linewidth=1.0, alpha=0.7)

    ax.set_title("Combined Modal Acceleration PSD (Post-Slew)", fontweight="bold")
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel(r"PSD (dB re (m/s$^2$)$^2$/Hz)")
    ax.set_xlim(2e-2, 10.0)
    ax.xaxis.set_major_locator(LogLocator(base=10, numticks=12))
    ax.xaxis.set_minor_locator(LogLocator(base=10, subs=np.arange(2, 10) * 0.1, numticks=120))
    if psd_min_db is not None and psd_max_db is not None:
        span = max(psd_max_db - psd_min_db, 20.0)
        pad = max(4.0, 0.12 * span)
        ax.set_ylim(
            5.0 * np.floor((psd_min_db - pad) / 5.0),
            5.0 * np.ceil((psd_max_db + pad) / 5.0),
        )
    ax.grid(True, alpha=0.3, which="both")
    ax.legend(loc="best")

    plt.tight_layout()
    _ensure_dir(out_dir)
    plot_path = os.path.abspath(os.path.join(out_dir, "modal_acceleration_psd.png"))
    plt.savefig(plot_path, dpi=500, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return plot_path


def _plot_sensitivity_functions(
    control_data: Dict[str, object],
    config: MissionConfig,
    out_dir: str,
) -> Optional[str]:
    """Plot sensitivity functions."""
    freqs = control_data["freqs"]
    l_data = control_data.get("L")
    s_data = control_data["S"]
    t_data = control_data["T"]
    l_flex = control_data.get("L_flex")
    s_flex = control_data.get("S_flex")
    t_flex = control_data.get("T_flex")

    if l_data is not None:
        s_data = {name: 1 / (1 + l_data[name]) for name in CONTROLLERS}
        t_data = {name: l_data[name] / (1 + l_data[name]) for name in CONTROLLERS}

    if l_flex is not None:
        s_flex = {name: 1 / (1 + l_flex[name]) for name in CONTROLLERS}
        t_flex = {name: l_flex[name] / (1 + l_flex[name]) for name in CONTROLLERS}

    plt.rcParams.update({
        "font.size": 10,
        "axes.labelsize": 11,
        "axes.titlesize": 12,
        "legend.fontsize": 9,
    })

    if s_flex is not None and t_flex is not None:
        fig, axes = plt.subplots(2, 2, figsize=(13, 8), sharex="col")
        (ax_s, ax_t), (ax_s_flex, ax_t_flex) = axes
    else:
        fig, axes = plt.subplots(1, 2, figsize=(13, 5))
        ax_s, ax_t = axes
        ax_s_flex, ax_t_flex = None, None

    line_styles = {"standard_pd": "-", "filtered_pd": "--"}

    def _plot_set(ax_left, ax_right, s_vals, t_vals) -> None:
        """Plot sensitivity and complementary sensitivity on two axes."""
        for name in CONTROLLERS:
            ls = line_styles.get(name, "-")
            s_mag_db = 20 * np.log10(np.abs(s_vals[name]) + 1e-12)
            t_mag_db = 20 * np.log10(np.abs(t_vals[name]) + 1e-12)
            ax_left.semilogx(freqs, s_mag_db, label=CONTROLLER_LABELS[name],
                             color=CONTROLLER_COLORS[name], linewidth=1.5, linestyle=ls)
            ax_right.semilogx(freqs, t_mag_db, label=CONTROLLER_LABELS[name],
                              color=CONTROLLER_COLORS[name], linewidth=1.5, linestyle=ls)

    _plot_set(ax_s, ax_t, s_data, t_data)
    if ax_s_flex is not None and ax_t_flex is not None and s_flex is not None and t_flex is not None:
        _plot_set(ax_s_flex, ax_t_flex, s_flex, t_flex)
        if config.modal_freqs_hz:
            for f_mode in config.modal_freqs_hz:
                idx = int(np.argmin(np.abs(freqs - f_mode)))
                for name in CONTROLLERS:
                    s_mag = 20 * np.log10(np.abs(s_flex[name][idx]) + 1e-12)
                    t_mag = 20 * np.log10(np.abs(t_flex[name][idx]) + 1e-12)
                    ax_s_flex.plot(
                        freqs[idx],
                        s_mag,
                        marker="o",
                        markersize=4,
                        color=CONTROLLER_COLORS[name],
                        linestyle="None",
                        alpha=0.9,
                    )
                    ax_t_flex.plot(
                        freqs[idx],
                        t_mag,
                        marker="o",
                        markersize=4,
                        color=CONTROLLER_COLORS[name],
                        linestyle="None",
                        alpha=0.9,
                    )

    for f_mode in config.modal_freqs_hz:
        ax_s.axvline(f_mode, color="gray", linestyle=":", alpha=0.7)
        ax_t.axvline(f_mode, color="gray", linestyle=":", alpha=0.7)
        if ax_s_flex is not None and ax_t_flex is not None:
            ax_s_flex.axvline(f_mode, color="gray", linestyle=":", alpha=0.7)
            ax_t_flex.axvline(f_mode, color="gray", linestyle=":", alpha=0.7)

    ax_s.axhline(0, color="black", linewidth=0.5, alpha=0.5)
    ax_t.axhline(0, color="black", linewidth=0.5, alpha=0.5)
    if ax_s_flex is not None and ax_t_flex is not None:
        ax_s_flex.axhline(0, color="black", linewidth=0.5, alpha=0.5)
        ax_t_flex.axhline(0, color="black", linewidth=0.5, alpha=0.5)

    ax_s.set_title(r"Rigid-Body Sensitivity $S(j\omega)$", fontweight="bold")
    ax_s.set_xlabel("Frequency (Hz)")
    ax_s.set_ylabel("Magnitude (dB)")
    ax_s.grid(True, alpha=0.3, which="both")
    ax_s.legend(loc="lower right")
    ax_s.set_ylim([-40, 10])
    ax_s.set_xlim([freqs[0], 10])

    ax_t.set_title(r"Rigid-Body Complementary Sensitivity $T(j\omega)$", fontweight="bold")
    ax_t.set_xlabel("Frequency (Hz)")
    ax_t.set_ylabel("Magnitude (dB)")
    ax_t.grid(True, alpha=0.3, which="both")
    ax_t.legend(loc="lower left")
    ax_t.set_ylim([-50, 10])
    ax_t.set_xlim([freqs[0], 10])

    note = "Rigid-body loop only; mode markers are references"
    ax_s.text(0.02, 0.95, note, transform=ax_s.transAxes, fontsize=9, color="gray", va="top")
    ax_t.text(0.02, 0.95, note, transform=ax_t.transAxes, fontsize=9, color="gray", va="top")

    if ax_s_flex is not None and ax_t_flex is not None:
        ax_s_flex.set_title(r"Flexible Sigma-Loop Sensitivity $S(j\omega)$", fontweight="bold")
        ax_s_flex.set_xlabel("Frequency (Hz)")
        ax_s_flex.set_ylabel("Magnitude (dB)")
        ax_s_flex.grid(True, alpha=0.3, which="both")
        ax_s_flex.legend(loc="lower right")
        ax_s_flex.set_ylim([-60, 20])
        ax_s_flex.set_xlim([freqs[0], 10])

        ax_t_flex.set_title(r"Flexible Sigma-Loop Complementary Sensitivity $T(j\omega)$", fontweight="bold")
        ax_t_flex.set_xlabel("Frequency (Hz)")
        ax_t_flex.set_ylabel("Magnitude (dB)")
        ax_t_flex.grid(True, alpha=0.3, which="both")
        ax_t_flex.legend(loc="lower left")
        ax_t_flex.set_ylim([-60, 20])
        ax_t_flex.set_xlim([freqs[0], 10])

        flex_err = 0.0
        if s_flex is not None and t_flex is not None:
            flex_err = max(
                float(np.max(np.abs(s_flex[name] + t_flex[name] - 1.0)))
                for name in CONTROLLERS
            )
        note_flex = (
            "Flexible sigma loop includes rigid + modal dynamics; "
            "mode markers are nominal\n"
            f"max|S+T-1|={flex_err:.1e}"
        )
        ax_s_flex.text(0.02, 0.95, note_flex, transform=ax_s_flex.transAxes,
                       fontsize=9, color="gray", va="top")
        ax_t_flex.text(0.02, 0.95, note_flex, transform=ax_t_flex.transAxes,
                       fontsize=9, color="gray", va="top")

    plt.tight_layout()
    _ensure_dir(out_dir)
    plot_path = os.path.abspath(os.path.join(out_dir, "mission_sensitivity.png"))
    plt.savefig(plot_path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return plot_path


def _plot_modal_excitation(
    control_data: Dict[str, object],
    config: MissionConfig,
    out_dir: str,
) -> Optional[str]:
    """Plot closed loop modal excitation magnitude."""
    freqs = control_data["freqs"]
    modal_response = control_data.get("modal_response", {})
    if not modal_response or not config.modal_freqs_hz:
        return None

    n_modes = len(config.modal_freqs_hz)
    if n_modes == 0:
        return None

    plt.rcParams.update({
        "font.size": 10,
        "axes.labelsize": 11,
        "axes.titlesize": 12,
        "legend.fontsize": 9,
    })

    ncols = 2 if n_modes > 1 else 1
    nrows = int(np.ceil(n_modes / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(6.5 * ncols, 4.5 * nrows), sharex=True)
    axes = np.atleast_1d(axes).ravel()

    line_styles = {"standard_pd": "-", "filtered_pd": "--"}

    for idx, f_mode in enumerate(config.modal_freqs_hz):
        ax = axes[idx]
        for name in CONTROLLERS:
            responses = modal_response.get(name, [])
            if idx >= len(responses):
                continue
            resp = responses[idx]
            mag_db = 20 * np.log10(np.abs(resp) + 1e-12)
            ax.semilogx(
                freqs,
                mag_db,
                label=CONTROLLER_LABELS[name],
                color=CONTROLLER_COLORS[name],
                linewidth=1.5,
                linestyle=line_styles.get(name, "-"),
            )
        ax.axvline(float(f_mode), color="gray", linestyle=":", alpha=0.7)
        ax.set_title(f"Mode {idx + 1} Response ({f_mode:.2f} Hz)", fontweight="bold")
        ax.set_xlabel("Frequency (Hz)")
        ax.set_ylabel("Magnitude (dB)")
        ax.grid(True, alpha=0.3, which="both")
        ax.axhline(0, color="black", linewidth=0.5, alpha=0.4)
        if idx == 0:
            ax.legend(loc="lower left")

    for extra_ax in axes[n_modes:]:
        extra_ax.axis("off")

    plt.tight_layout()
    _ensure_dir(out_dir)
    plot_path = os.path.abspath(os.path.join(out_dir, "mission_modal_excitation.png"))
    plt.savefig(plot_path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return plot_path


def _plot_pointing_error(
    pointing_errors: Dict[str, Dict[str, object]],
    out_dir: str,
    config: Optional[MissionConfig] = None,
) -> Optional[str]:
    """Plot mission pointing error with full and post slew subplots."""
    if not pointing_errors:
        return None

    plt.rcParams.update({
        "font.size": 10,
        "axes.labelsize": 11,
        "axes.titlesize": 12,
        "legend.fontsize": 9,
    })

    fig, (ax_full, ax_post) = plt.subplots(1, 2, figsize=(16, 6), sharey=False)

    plot_entries = []
    maneuver_end_candidates: List[float] = []
    post_slew_rms_arcsec: Dict[str, float] = {}
    time_start = float("inf")
    time_end = 0.0
    min_positive_time = float("inf")
    fallback_maneuver_end = float(config.slew_duration_s) if config is not None else 30.0

    target_controller = "standard_pd"
    target_methods = [m for m in METHODS if m in pointing_errors]

    for method in target_methods:
        method_data = pointing_errors.get(method, {})
        data = method_data.get(target_controller)
        if not data:
            continue

        run_mode = str(data.get("run_mode", "")).lower()
        if run_mode and run_mode != "combined":
            continue

        time = np.array(data.get("time", np.array([])), dtype=float)
        if len(time) == 0:
            continue

        errors_deg = _extract_pointing_error(data, config=config)
        if len(errors_deg) == 0:
            continue

        time, aligned = _align_series(time, errors_deg)
        errors_deg = aligned[0]
        if len(time) == 0:
            continue

        maneuver_end = _infer_maneuver_end(
            time,
            data.get("control_mode"),
            data.get("torque"),
            fallback_maneuver_end,
        )
        maneuver_end_candidates.append(float(maneuver_end))
        idx_end = int(np.searchsorted(time, maneuver_end))
        if 0 <= idx_end < len(errors_deg) - 1:
            residual = errors_deg[idx_end:]
        else:
            residual = errors_deg[int(0.9 * len(errors_deg)):] if len(errors_deg) > 10 else errors_deg
        rms_arcsec = float(np.sqrt(np.mean(np.square(residual))) * 3600.0) if len(residual) else 0.0
        post_slew_rms_arcsec[method] = rms_arcsec

        errors_deg_abs = np.abs(errors_deg)
        errors_arcsec = errors_deg_abs * 3600.0
        positive_time = time[time > 0]
        time_floor = float(np.min(positive_time)) if len(positive_time) else 1e-6
        time_plot = np.where(time > 0, time, time_floor)
        label = f"{_combo_label(method, target_controller)} (post-slew RMS={rms_arcsec:.2f} arcsec)"
        plot_entries.append({
            "time": time_plot,
            "errors_deg": errors_deg_abs,
            "errors_arcsec": errors_arcsec,
            "label": label,
            "color": _combo_color(method, target_controller),
        })
        time_start = min(time_start, float(np.min(time_plot)))
        time_end = max(time_end, float(time[-1]))
        min_positive_time = min(min_positive_time, time_floor)

    if not plot_entries:
        plt.close(fig)
        return None

    if not np.isfinite(time_start):
        time_start = 1e-6
    if not np.isfinite(min_positive_time):
        min_positive_time = 1e-6
    maneuver_end = (
        float(np.median(maneuver_end_candidates))
        if maneuver_end_candidates
        else min(fallback_maneuver_end, time_end if time_end > 0 else fallback_maneuver_end)
    )
    maneuver_end = max(time_start, min(maneuver_end, time_end if time_end > time_start else maneuver_end))
    post_start = max(maneuver_end, min_positive_time)

    if time_end > time_start:
        ax_full.axvspan(time_start, maneuver_end, color="#d9ecff", alpha=0.30, zorder=0, label="During slew")
        ax_full.axvspan(maneuver_end, time_end, color="#e8f5df", alpha=0.30, zorder=0, label="Post slew")

    for entry in plot_entries:
        ax_full.semilogx(
            entry["time"],
            entry["errors_deg"],
            color=entry["color"],
            label=entry["label"],
            linewidth=1.8,
            linestyle="-",
        )
        mask_post = entry["time"] >= post_start
        if np.any(mask_post):
            ax_post.plot(
                entry["time"][mask_post],
                entry["errors_arcsec"][mask_post],
                color=entry["color"],
                label=entry["label"],
                linewidth=1.8,
                linestyle="-",
            )

    ax_full.axvline(maneuver_end, color="gray", linestyle=":", linewidth=1.5, alpha=0.9)
    ax_full.text(
        maneuver_end,
        0.98,
        f" Maneuver end ({maneuver_end:.1f}s)",
        transform=ax_full.get_xaxis_transform(),
        ha="left",
        va="top",
        fontsize=9,
        color="dimgray",
    )

    ax_full.set_title("Full Mission Pointing Error", fontweight="bold")
    ax_full.set_xlabel("Time (s)")
    ax_full.set_ylabel("Absolute Pointing Error (deg)")
    ax_full.grid(True, alpha=0.3, which="both")
    if time_end > time_start:
        ax_full.set_xlim([time_start, time_end])
    y_max = max(float(np.max(entry["errors_deg"])) for entry in plot_entries if len(entry["errors_deg"]) > 0)
    ax_full.set_ylim([0.0, max(y_max * 1.05, 1e-3)])

    ax_post.set_title("Post-Slew Pointing Error", fontweight="bold")
    ax_post.set_xlabel("Time (s)")
    ax_post.set_ylabel("Absolute Pointing Error (arcsec)")
    ax_post.grid(True, alpha=0.3, which="both")
    if time_end > post_start:
        ax_post.set_xlim([post_start, time_end])
    else:
        ax_post.set_xlim([time_start, time_end if time_end > time_start else time_start * 10.0])
    post_max = 0.0
    for entry in plot_entries:
        mask = entry["time"] >= post_start
        if np.any(mask):
            post_max = max(post_max, float(np.max(entry["errors_arcsec"][mask])))
    ax_post.set_ylim([0.0, max(post_max * 1.10, 1e-3)])

    handles, labels = ax_full.get_legend_handles_labels()
    unique_handles = []
    unique_labels = []
    for handle, label in zip(handles, labels):
        if label not in unique_labels:
            unique_handles.append(handle)
            unique_labels.append(label)
    ax_full.legend(unique_handles, unique_labels, loc="lower left")
    ax_post.legend(loc="upper right")

    fig.suptitle("Pointing Error Comparison: Feedforward Profiles (Standard PD)", fontweight="bold", y=0.98)
    fig.subplots_adjust(left=0.06, right=0.99, top=0.90, bottom=0.14, wspace=0.22)
    _ensure_dir(out_dir)
    plot_path = os.path.abspath(os.path.join(out_dir, "mission_pointing_error.png"))
    plt.savefig(plot_path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return plot_path


def _plot_psd_comparison(
    mission_psd_data: Dict[str, Dict[str, Dict[str, object]]],
    config: MissionConfig,
    out_dir: str,
) -> Optional[str]:
    """Plot vibration PSD comparison."""
    if not mission_psd_data:
        return None

    plt.rcParams.update({
        "font.size": 12,
        "axes.labelsize": 14,
        "axes.titlesize": 15,
        "legend.fontsize": 11,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
    })

    fig, ax = plt.subplots(1, 1, figsize=(16, 10))

    plot_entries = []
    psd_min_db = None
    psd_max_db = None

    for method in METHODS:
        if method not in mission_psd_data:
            continue
        for name in CONTROLLERS:
            if name not in mission_psd_data[method]:
                continue
            psd_freq = mission_psd_data[method][name]["psd_freq"]
            psd_vals = mission_psd_data[method][name]["psd"]
            if len(psd_freq) == 0:
                continue

            # Filter to the 0 to 10 Hz range
            mask = (psd_freq > 0) & (psd_freq <= 10.0) & np.isfinite(psd_vals) & (psd_vals > 0)
            if not np.any(mask):
                continue

            freq_filtered = psd_freq[mask]
            psd_filtered = psd_vals[mask]

            psd_db = 10.0 * np.log10(psd_filtered)
            psd_min_db = float(np.min(psd_db)) if psd_min_db is None else min(psd_min_db, float(np.min(psd_db)))
            psd_max_db = float(np.max(psd_db)) if psd_max_db is None else max(psd_max_db, float(np.max(psd_db)))

            label = _combo_label(method, name)
            plot_entries.append({
                "freq": freq_filtered,
                "psd_db": psd_db,
                "label": label,
                "linestyle": "-",
                "color": _combo_color(method, name),
            })

    if not plot_entries:
        plt.close(fig)
        return None

    for entry in plot_entries:
        color = entry.get("color", "#555555")
        ax.semilogx(
            entry["freq"],
            entry["psd_db"],
            color=color,
            linestyle=entry["linestyle"],
            linewidth=2.0,
            alpha=0.9,
            label=entry["label"],
        )

    # Mark modal frequencies
    if psd_max_db is not None:
        label_y = psd_max_db - 3.0
        for f_mode in config.modal_freqs_hz:
            if f_mode <= 10.0:
                ax.axvline(f_mode, color="red", linestyle="--", alpha=0.6, linewidth=1.5)
                ax.text(f_mode + 0.05, label_y, f"Mode: {f_mode:.2f} Hz",
                        rotation=90, va="bottom", ha="left", fontsize=10, alpha=0.9,
                        fontweight="bold", color="red")

    ax.set_title("Mission Vibration Displacement PSD (Combined Feedforward + Feedback)",
                 fontweight="bold", fontsize=16, pad=15)
    ax.set_xlabel("Frequency (Hz)", fontsize=14, fontweight="bold")
    ax.set_ylabel(r"PSD (dB re m$^2$/Hz)", fontsize=14, fontweight="bold")

    ax.grid(True, alpha=0.6, which="major", linestyle="-", linewidth=0.8, color="gray")
    ax.grid(True, alpha=0.3, which="minor", linestyle="-", linewidth=0.4, color="lightgray")

    if plot_entries:
        ax.legend(loc="upper right", framealpha=0.95, ncol=1, fontsize=11,
                  fancybox=True, shadow=True)

    plt.tight_layout()
    _ensure_dir(out_dir)
    plot_path = os.path.abspath(os.path.join(out_dir, "mission_psd.png"))
    plt.savefig(plot_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return plot_path


def _plot_torque_command_time(
    feedback_vibration: Dict[str, Dict[str, object]],
    out_dir: str,
) -> Optional[str]:
    """Plot commanded torque vs time for combined FF+FB runs."""
    if not feedback_vibration:
        return None

    plt.rcParams.update({
        "font.size": 10,
        "axes.labelsize": 11,
        "axes.titlesize": 12,
        "legend.fontsize": 9,
    })

    fig, ax = plt.subplots(1, 1, figsize=(12, 6))
    plot_entries = []

    for method in METHODS:
        for controller in CONTROLLERS:
            key = f"{method}_{controller}"
            data = feedback_vibration.get(key)
            if not data:
                continue
            run_mode = str(data.get("run_mode", "")).lower()
            if run_mode and run_mode != "combined":
                continue
            time = data.get("time", np.array([]))
            torque = data.get("torque_total")
            if torque is None or len(torque) == 0:
                torque = data.get("torque", np.array([]))
            if len(time) == 0 or len(torque) == 0:
                continue
            time, aligned = _align_series(time, torque)
            torque = aligned[0]
            if len(time) == 0:
                continue

            plot_entries.append({
                "time": time,
                "torque": torque,
                "label": _combo_label(method, controller),
                "linestyle": "-",
                "color": _combo_color(method, controller),
            })

    if not plot_entries:
        plt.close(fig)
        return None

    for entry in plot_entries:
        color = entry.get("color", "#555555")
        ax.plot(
            entry["time"],
            entry["torque"],
            color=color,
            label=entry["label"],
            linewidth=1.5,
            linestyle=entry["linestyle"],
        )

    ax.set_title("Torque Command vs Time", fontweight="bold")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Commanded Torque (N·m)")
    ax.grid(True, alpha=0.3, which="both")
    ax.legend(loc="upper right")

    plt.tight_layout()
    _ensure_dir(out_dir)
    plot_path = os.path.abspath(os.path.join(out_dir, "mission_torque_command.png"))
    plt.savefig(plot_path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return plot_path


def _plot_torque_command_psd(
    feedback_vibration: Dict[str, Dict[str, object]],
    control_data: Dict[str, object],
    config: MissionConfig,
    out_dir: str,
) -> Optional[str]:
    """Plot commanded torque PSD for combined FF+FB runs with high frequency detail."""
    if not feedback_vibration:
        return None

    plt.rcParams.update({
        "font.size": 12,
        "axes.labelsize": 14,
        "axes.titlesize": 15,
        "legend.fontsize": 11,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
    })

    fig, ax = plt.subplots(1, 1, figsize=(16, 10))
    plot_entries = []
    psd_min_db = None
    psd_max_db = None
    resonances_hz, antiresonances_hz = _detect_mode_lines_from_flexible_plant(control_data, config, fmax_hz=10.0)

    for method in METHODS:
        for controller in CONTROLLERS:
            key = f"{method}_{controller}"
            data = feedback_vibration.get(key)
            if not data:
                continue
            run_mode = str(data.get("run_mode", "")).lower()
            if run_mode and run_mode != "combined":
                continue

            time = data.get("time", np.array([]))
            torque = data.get("torque_total")
            if torque is None or len(torque) == 0:
                torque = data.get("torque", np.array([]))
            if len(time) == 0 or len(torque) == 0:
                continue
            time, aligned = _align_series(time, torque)
            torque = aligned[0]
            if len(time) == 0:
                continue
            psd_freq, psd_vals = _compute_psd_high_resolution(time, torque)
            if len(psd_freq) == 0:
                continue

            mask = (psd_freq > 0) & (psd_freq <= 10.0) & np.isfinite(psd_vals) & (psd_vals > 0)
            if not np.any(mask):
                continue

            freq_filtered = psd_freq[mask]
            psd_filtered = psd_vals[mask]
            psd_db = 10.0 * np.log10(psd_filtered)

            psd_min_db = float(np.min(psd_db)) if psd_min_db is None else min(psd_min_db, float(np.min(psd_db)))
            psd_max_db = float(np.max(psd_db)) if psd_max_db is None else max(psd_max_db, float(np.max(psd_db)))

            plot_entries.append({
                "freq": freq_filtered,
                "psd_db": psd_db,
                "label": _combo_label(method, controller),
                "linestyle": "-",
                "color": _combo_color(method, controller),
            })

    if not plot_entries:
        plt.close(fig)
        return None

    for entry in plot_entries:
        color = entry.get("color", "#555555")
        ax.semilogx(
            entry["freq"],
            entry["psd_db"],
            color=color,
            linestyle=entry["linestyle"],
            linewidth=1.8,
            alpha=0.95,
            label=entry["label"],
        )

    ax.set_title("Torque Command PSD (Combined Feedforward + Feedback)", fontweight="bold", fontsize=16, pad=15)
    ax.set_xlabel("Frequency (Hz)", fontsize=14, fontweight="bold")
    ax.set_ylabel(r"PSD (dB re N$^2$m$^2$/Hz)", fontsize=14, fontweight="bold")
    ax.set_xlim(2e-2, 10.0)
    if psd_min_db is not None and psd_max_db is not None:
        span = max(psd_max_db - psd_min_db, 20.0)
        pad = max(4.0, 0.12 * span)
        ax.set_ylim(5.0 * np.floor((psd_min_db - pad) / 5.0), 5.0 * np.ceil((psd_max_db + pad) / 5.0))
    ax.xaxis.set_major_locator(LogLocator(base=10, numticks=12))
    ax.xaxis.set_minor_locator(LogLocator(base=10, subs=np.arange(2, 10) * 0.1, numticks=120))
    _draw_mode_lines_on_axis(ax, resonances_hz, antiresonances_hz)
    ax.grid(True, alpha=0.6, which="major", linestyle="-", linewidth=0.8, color="gray")
    ax.grid(True, alpha=0.3, which="minor", linestyle="-", linewidth=0.4, color="lightgray")

    ax.legend(loc="upper right", framealpha=0.95, ncol=1, fontsize=11,
              fancybox=True, shadow=True)

    plt.tight_layout()
    _ensure_dir(out_dir)
    plot_path = os.path.abspath(os.path.join(out_dir, "mission_torque_command_psd.png"))
    plt.savefig(plot_path, dpi=600, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return plot_path


def _plot_tracking_response(
    pointing_data: Dict[str, Dict[str, object]],
    config: MissionConfig,
    out_dir: str,
) -> Optional[str]:
    """Plot absolute tracking error vs time for mission methods."""
    if not pointing_data:
        return None

    plt.rcParams.update({
        "font.size": 10,
        "axes.labelsize": 11,
        "axes.titlesize": 12,
        "legend.fontsize": 9,
    })

    fig, ax = plt.subplots(1, 1, figsize=(11, 5.5))
    axis = _normalize_axis(config.rotation_axis)
    plotted = False

    for method in METHODS:
        method_data = pointing_data.get(method, {})
        if not method_data:
            continue

        max_time = None
        for controller in CONTROLLERS:
            data = method_data.get(controller)
            if not data:
                continue
            time = np.array(data.get("time", []), dtype=float)
            if len(time) == 0:
                continue
            max_time = float(time[-1]) if max_time is None else max(max_time, float(time[-1]))

        if max_time is None:
            continue

        settling_time = max(0.0, (max_time or config.slew_duration_s) - config.slew_duration_s)
        ref_data = _compute_torque_profile(config, method, settling_time=settling_time)
        ref_time = np.array(ref_data.get("time", []), dtype=float)
        ref_theta = np.array(ref_data.get("theta", []), dtype=float)
        if len(ref_time) == 0 or len(ref_theta) == 0:
            continue
        ref_time, aligned = _align_series(ref_time, ref_theta)
        ref_theta = aligned[0]
        if len(ref_time) == 0:
            continue
        ref_unwrapped = np.unwrap(ref_theta)

        for controller in CONTROLLERS:
            data = method_data.get(controller)
            if not data:
                continue
            run_mode = str(data.get("run_mode", "")).lower()
            if run_mode and run_mode != "combined":
                continue
            time = np.array(data.get("time", []), dtype=float)
            sigma = np.array(data.get("sigma", []), dtype=float)
            if len(time) == 0 or len(sigma) == 0:
                continue
            theta_actual = _sigma_to_angle(sigma, axis)
            time, aligned = _align_series(time, theta_actual)
            theta_actual = aligned[0]
            if len(time) == 0:
                continue
            actual_unwrapped = np.unwrap(theta_actual)
            ref_interp = np.interp(time, ref_time, ref_unwrapped, left=ref_unwrapped[0], right=ref_unwrapped[-1])
            err_rad = _wrap_angle_rad(ref_interp - actual_unwrapped)
            err_abs_deg = np.abs(np.degrees(err_rad))

            color = _combo_color(method, controller)
            ax.plot(time, err_abs_deg, color=color, linewidth=1.7, label=_combo_label(method, controller))
            plotted = True

    if not plotted:
        plt.close(fig)
        return None

    maneuver_end = float(config.slew_duration_s)
    ax.axvline(maneuver_end, color="gray", linestyle=":", alpha=0.7, linewidth=1.4)
    ax.set_title("Feedback Tracking Error vs Time", fontweight="bold")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Absolute Tracking Error (deg)")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right")

    # Remove legacy tracking response image to avoid confusion.
    legacy_path = os.path.abspath(os.path.join(out_dir, "mission_tracking_response.png"))
    if os.path.exists(legacy_path):
        try:
            os.remove(legacy_path)
        except OSError:
            pass

    plt.tight_layout()
    _ensure_dir(out_dir)
    plot_path = os.path.abspath(os.path.join(out_dir, "mission_tracking_error.png"))
    plt.savefig(plot_path, dpi=250, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return plot_path


def _plot_tracking_error_psd(
    pointing_data: Dict[str, Dict[str, object]],
    control_data: Dict[str, object],
    config: MissionConfig,
    out_dir: str,
) -> Optional[str]:
    """Plot high resolution PSD of feedback tracking error."""
    if not pointing_data:
        return None

    plt.rcParams.update({
        "font.size": 12,
        "axes.labelsize": 14,
        "axes.titlesize": 15,
        "legend.fontsize": 10,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
    })

    fig, ax = plt.subplots(1, 1, figsize=(16, 10))
    axis = _normalize_axis(config.rotation_axis)
    resonances_hz, antiresonances_hz = _detect_mode_lines_from_flexible_plant(control_data, config, fmax_hz=10.0)

    plotted = False
    psd_min_db = None
    psd_max_db = None

    for method in METHODS:
        method_data = pointing_data.get(method, {})
        if not method_data:
            continue

        max_time = None
        for controller in CONTROLLERS:
            data = method_data.get(controller)
            if not data:
                continue
            time = np.array(data.get("time", []), dtype=float)
            if len(time) == 0:
                continue
            max_time = float(time[-1]) if max_time is None else max(max_time, float(time[-1]))

        if max_time is None:
            continue

        settling_time = max(0.0, (max_time or config.slew_duration_s) - config.slew_duration_s)
        ref_data = _compute_torque_profile(config, method, settling_time=settling_time)
        ref_time = np.array(ref_data.get("time", []), dtype=float)
        ref_theta = np.array(ref_data.get("theta", []), dtype=float)
        if len(ref_time) == 0 or len(ref_theta) == 0:
            continue
        ref_time, aligned = _align_series(ref_time, ref_theta)
        ref_theta = aligned[0]
        if len(ref_time) == 0:
            continue
        ref_unwrapped = np.unwrap(ref_theta)

        for controller in CONTROLLERS:
            data = method_data.get(controller)
            if not data:
                continue
            run_mode = str(data.get("run_mode", "")).lower()
            if run_mode and run_mode != "combined":
                continue

            time = np.array(data.get("time", []), dtype=float)
            sigma = np.array(data.get("sigma", []), dtype=float)
            if len(time) == 0 or len(sigma) == 0:
                continue

            theta_actual = _sigma_to_angle(sigma, axis)
            time, aligned = _align_series(time, theta_actual)
            theta_actual = aligned[0]
            if len(time) == 0:
                continue

            actual_unwrapped = np.unwrap(theta_actual)
            ref_interp = np.interp(time, ref_time, ref_unwrapped, left=ref_unwrapped[0], right=ref_unwrapped[-1])
            err_rad = _wrap_angle_rad(ref_interp - actual_unwrapped)
            err_deg = np.degrees(err_rad)
            err_deg = signal.detrend(err_deg, type="linear")

            psd_freq, psd_vals = _compute_psd_high_resolution(time, err_deg)
            if len(psd_freq) == 0:
                continue

            mask = (psd_freq > 0) & (psd_freq <= 10.0) & np.isfinite(psd_vals) & (psd_vals > 0)
            if not np.any(mask):
                continue

            freq = psd_freq[mask]
            psd_db = 10.0 * np.log10(psd_vals[mask])
            psd_min_db = float(np.min(psd_db)) if psd_min_db is None else min(psd_min_db, float(np.min(psd_db)))
            psd_max_db = float(np.max(psd_db)) if psd_max_db is None else max(psd_max_db, float(np.max(psd_db)))

            ax.semilogx(
                freq,
                psd_db,
                color=_combo_color(method, controller),
                linewidth=1.8,
                alpha=0.95,
                label=_combo_label(method, controller),
            )
            plotted = True

    if not plotted:
        plt.close(fig)
        return None

    _draw_mode_lines_on_axis(ax, resonances_hz, antiresonances_hz)
    ax.set_title("Feedback Tracking Error PSD (High Resolution)", fontweight="bold")
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel(r"PSD (dB re deg$^2$/Hz)")
    ax.set_xlim(2e-2, 10.0)
    if psd_min_db is not None and psd_max_db is not None:
        span = max(psd_max_db - psd_min_db, 20.0)
        pad = max(4.0, 0.12 * span)
        ax.set_ylim(5.0 * np.floor((psd_min_db - pad) / 5.0), 5.0 * np.ceil((psd_max_db + pad) / 5.0))
    ax.xaxis.set_major_locator(LogLocator(base=10, numticks=12))
    ax.xaxis.set_minor_locator(LogLocator(base=10, subs=np.arange(2, 10) * 0.1, numticks=120))
    ax.grid(True, alpha=0.6, which="major", linestyle="-", linewidth=0.8, color="gray")
    ax.grid(True, alpha=0.3, which="minor", linestyle="-", linewidth=0.4, color="lightgray")
    ax.legend(loc="best", framealpha=0.95, fancybox=True, shadow=True)

    plt.tight_layout()
    _ensure_dir(out_dir)
    plot_path = os.path.abspath(os.path.join(out_dir, "mission_tracking_error_psd.png"))
    plt.savefig(plot_path, dpi=600, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return plot_path


def _plot_tracking_transfer(
    control_data: Dict[str, object],
    out_dir: str,
) -> Optional[str]:
    """Plot tracking transfer functions |E/R| and |Y/R|."""
    freqs = control_data.get("freqs")
    s_data = control_data.get("S")
    t_data = control_data.get("T")
    if freqs is None or s_data is None or t_data is None:
        return None

    plt.rcParams.update({
        "font.size": 10,
        "axes.labelsize": 11,
        "axes.titlesize": 12,
        "legend.fontsize": 9,
    })

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    ax_e, ax_y = axes

    for name in CONTROLLERS:
        s_mag_db = 20 * np.log10(np.abs(s_data[name]) + 1e-12)
        t_mag_db = 20 * np.log10(np.abs(t_data[name]) + 1e-12)
        ax_e.semilogx(freqs, s_mag_db, color=CONTROLLER_COLORS[name], linewidth=1.6,
                      label=CONTROLLER_LABELS[name])
        ax_y.semilogx(freqs, t_mag_db, color=CONTROLLER_COLORS[name], linewidth=1.6,
                      label=CONTROLLER_LABELS[name])

    ax_e.set_title("Tracking Error TF |E/R| = |S|", fontweight="bold")
    ax_y.set_title("Tracking Output TF |Y/R| = |T|", fontweight="bold")
    ax_e.set_xlabel("Frequency (Hz)")
    ax_y.set_xlabel("Frequency (Hz)")
    ax_e.set_ylabel("Magnitude (dB)")
    ax_y.set_ylabel("Magnitude (dB)")
    ax_e.grid(True, alpha=0.3, which="both")
    ax_y.grid(True, alpha=0.3, which="both")
    ax_e.legend(loc="best")
    ax_y.legend(loc="best")

    plt.tight_layout()
    _ensure_dir(out_dir)
    plot_path = os.path.abspath(os.path.join(out_dir, "mission_tracking_tf.png"))
    plt.savefig(plot_path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return plot_path


def _plot_disturbance_to_torque(
    control_data: Dict[str, object],
    out_dir: str,
) -> Optional[str]:
    """Plot disturbance torque to commanded torque transfer magnitude."""
    freqs = control_data.get("freqs")
    t_data = control_data.get("T_flex") or control_data.get("T")
    if freqs is None or t_data is None:
        return None

    plt.rcParams.update({
        "font.size": 10,
        "axes.labelsize": 11,
        "axes.titlesize": 12,
        "legend.fontsize": 9,
    })

    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    for name in CONTROLLERS:
        t_mag_db = 20 * np.log10(np.abs(t_data[name]) + 1e-12)
        ax.semilogx(freqs, t_mag_db, color=CONTROLLER_COLORS[name], linewidth=1.6,
                    label=CONTROLLER_LABELS[name])

    ax.set_title("Disturbance Torque \u2192 Commanded Torque (|T|)", fontweight="bold")
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Magnitude (dB)")
    ax.grid(True, alpha=0.3, which="both")
    ax.legend(loc="best")

    plt.tight_layout()
    _ensure_dir(out_dir)
    plot_path = os.path.abspath(os.path.join(out_dir, "mission_disturbance_to_torque.png"))
    plt.savefig(plot_path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return plot_path


def _plot_torque_psd_split(
    feedforward_torque: Dict[str, Dict[str, np.ndarray]],
    feedback_vibration: Dict[str, Dict[str, object]],
    control_data: Dict[str, object],
    config: MissionConfig,
    out_dir: str,
) -> Optional[str]:
    """Plot feedforward vs feedback torque PSDs in separate subplots."""
    if not feedforward_torque and not feedback_vibration:
        return None

    plt.rcParams.update({
        "font.size": 12,
        "axes.labelsize": 14,
        "axes.titlesize": 15,
        "legend.fontsize": 11,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
    })

    fig, (ax_ff, ax_fb) = plt.subplots(1, 2, figsize=(18, 6), sharex=True)
    resonances_hz, antiresonances_hz = _detect_mode_lines_from_flexible_plant(control_data, config, fmax_hz=10.0)

    def _build_psd_entry(
        time: np.ndarray,
        torque: np.ndarray,
        label: str,
        linestyle: str,
        color: str,
    ) -> Optional[Dict[str, object]]:
        """Compute high resolution PSD and package it for plotting."""
        if len(time) == 0 or len(torque) == 0:
            return None
        time, aligned = _align_series(time, torque)
        torque = aligned[0]
        if len(time) == 0:
            return None
        freq, psd = _compute_psd_high_resolution(time, torque)
        if len(freq) == 0:
            return None
        mask = (freq > 0) & (freq <= 10.0) & np.isfinite(psd) & (psd > 0)
        if not np.any(mask):
            return None
        freq = freq[mask]
        psd_db = 10.0 * np.log10(psd[mask])
        return {
            "freq": freq,
            "psd_db": psd_db,
            "label": label,
            "linestyle": linestyle,
            "color": color,
        }

    def _plot_psd_lines(ax, entries):
        """Draw PSD curves on the given axes from a list of entry dicts."""
        for entry in entries:
            color = entry.get("color", "#555555")
            ax.semilogx(
                entry["freq"],
                entry["psd_db"],
                color=color,
                linestyle=entry["linestyle"],
                linewidth=2.0,
                alpha=0.95,
                label=entry["label"],
            )

    def _set_psd_limits(ax, entries):
        """Set y axis limits from the 1st and 99th percentile of PSD values."""
        if not entries:
            return
        vals = np.concatenate([entry["psd_db"] for entry in entries if len(entry["psd_db"]) > 0])
        vals = vals[np.isfinite(vals)]
        if len(vals) == 0:
            return
        lo = float(np.percentile(vals, 1))
        hi = float(np.percentile(vals, 99))
        if hi <= lo:
            lo = float(np.min(vals))
            hi = float(np.max(vals))
        span = max(hi - lo, 10.0)
        pad = max(3.0, 0.12 * span)
        y_min = 5.0 * np.floor((lo - pad) / 5.0)
        y_max = 5.0 * np.ceil((hi + pad) / 5.0)
        ax.set_ylim(y_min, y_max)

    ff_entries = []
    for method in METHODS:
        # Prefer FF torque from combined runs so FF/FB PSD comparison uses the same mission run.
        combined_entry = None
        for controller in CONTROLLERS:
            key = f"{method}_{controller}"
            data = feedback_vibration.get(key)
            if not data:
                continue
            run_mode = str(data.get("run_mode", "")).lower()
            if run_mode and run_mode != "combined":
                continue
            combined_entry = _build_psd_entry(
                np.array(data.get("time", np.array([])), dtype=float),
                np.array(data.get("torque_ff", np.array([])), dtype=float),
                f"FF: {METHOD_LABELS.get(method, method)}",
                "-",
                METHOD_COLORS.get(method, "#555555"),
            )
            if combined_entry is not None:
                break

        if combined_entry is not None:
            ff_entries.append(combined_entry)
            continue

        # Fallback for older data where combined FF torque is unavailable.
        data = feedforward_torque.get(method)
        if not data:
            continue
        fallback_entry = _build_psd_entry(
            np.array(data.get("time", np.array([])), dtype=float),
            np.array(data.get("torque", np.array([])), dtype=float),
            f"FF: {METHOD_LABELS.get(method, method)}",
            "-",
            METHOD_COLORS.get(method, "#555555"),
        )
        if fallback_entry is not None:
            ff_entries.append(fallback_entry)

    fb_entries = []
    for method in METHODS:
        for controller in CONTROLLERS:
            key = f"{method}_{controller}"
            data = feedback_vibration.get(key)
            if not data:
                continue
            run_mode = str(data.get("run_mode", "")).lower()
            if run_mode and run_mode != "combined":
                continue
            fb_entry = _build_psd_entry(
                np.array(data.get("time", np.array([])), dtype=float),
                np.array(data.get("torque", np.array([])), dtype=float),
                _combo_label(method, controller),
                "-",
                _combo_color(method, controller),
            )
            if fb_entry is not None:
                fb_entries.append(fb_entry)

    if not ff_entries and not fb_entries:
        plt.close(fig)
        return None

    if ff_entries:
        _plot_psd_lines(ax_ff, ff_entries)
        _draw_mode_lines_on_axis(ax_ff, resonances_hz, antiresonances_hz)
        ax_ff.set_title("Feedforward Torque Command PSD", fontweight="bold")
        ax_ff.set_xlabel("Frequency (Hz)")
        ax_ff.set_ylabel(r"PSD (dB re N$^2$m$^2$/Hz)")
        _set_psd_limits(ax_ff, ff_entries)
        ax_ff.grid(True, alpha=0.6, which="major", linestyle="-", linewidth=0.8, color="gray")
        ax_ff.grid(True, alpha=0.3, which="minor", linestyle="-", linewidth=0.4, color="lightgray")
        ax_ff.legend(loc="best", framealpha=0.95, fontsize=10, fancybox=True, shadow=True)

    if fb_entries:
        _plot_psd_lines(ax_fb, fb_entries)
        _draw_mode_lines_on_axis(ax_fb, resonances_hz, antiresonances_hz)
        ax_fb.set_title("Feedback Torque Command PSD", fontweight="bold")
        ax_fb.set_xlabel("Frequency (Hz)")
        ax_fb.set_ylabel(r"PSD (dB re N$^2$m$^2$/Hz)")
        _set_psd_limits(ax_fb, fb_entries)
        ax_fb.grid(True, alpha=0.6, which="major", linestyle="-", linewidth=0.8, color="gray")
        ax_fb.grid(True, alpha=0.3, which="minor", linestyle="-", linewidth=0.4, color="lightgray")
        ax_fb.legend(loc="best", framealpha=0.95, fontsize=10, fancybox=True, shadow=True, ncol=1)

    # Keep low frequency content readable while retaining mission band up to 10 Hz.
    for ax in (ax_ff, ax_fb):
        ax.set_xlim(2e-2, 10.0)
        ax.xaxis.set_major_locator(LogLocator(base=10, numticks=12))
        ax.xaxis.set_minor_locator(LogLocator(base=10, subs=np.arange(2, 10) * 0.1, numticks=120))

    plt.tight_layout()
    _ensure_dir(out_dir)
    plot_path = os.path.abspath(os.path.join(out_dir, "mission_torque_psd_split.png"))
    plt.savefig(plot_path, dpi=500, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return plot_path


def _plot_torque_psd_coherence(
    feedforward_torque: Dict[str, Dict[str, np.ndarray]],
    feedback_vibration: Dict[str, Dict[str, object]],
    out_dir: str,
) -> Optional[str]:
    """Plot magnitude squared coherence between FF and FB torque commands."""
    if not feedforward_torque or not feedback_vibration:
        return None

    plt.rcParams.update({
        "font.size": 10,
        "axes.labelsize": 11,
        "axes.titlesize": 12,
        "legend.fontsize": 9,
    })

    fig, axes = plt.subplots(2, 2, figsize=(14, 10), sharex=True, sharey=True)
    axes = axes.flatten()
    plot_count = 0

    def _common_time_series(
        time_a: np.ndarray, sig_a: np.ndarray, time_b: np.ndarray, sig_b: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Resample two signals onto a common uniform time grid."""
        if len(time_a) == 0 or len(time_b) == 0:
            return np.array([]), np.array([]), np.array([])
        t0 = max(float(time_a[0]), float(time_b[0]))
        t1 = min(float(time_a[-1]), float(time_b[-1]))
        if t1 <= t0:
            return np.array([]), np.array([]), np.array([])
        dt_a = float(np.median(np.diff(time_a))) if len(time_a) > 1 else UNIFIED_SAMPLE_DT
        dt_b = float(np.median(np.diff(time_b))) if len(time_b) > 1 else UNIFIED_SAMPLE_DT
        dt_target = min(dt_a, dt_b, UNIFIED_SAMPLE_DT)
        if not np.isfinite(dt_target) or dt_target <= 0:
            dt_target = UNIFIED_SAMPLE_DT
        t_common = np.arange(t0, t1 + 0.5 * dt_target, dt_target)
        a_interp = np.interp(t_common, time_a, sig_a)
        b_interp = np.interp(t_common, time_b, sig_b)
        return t_common, a_interp, b_interp

    for method in METHODS:
        ff_data = feedforward_torque.get(method)
        if not ff_data:
            continue
        ff_time = ff_data.get("time", np.array([]))
        ff_torque = ff_data.get("torque", np.array([]))
        if len(ff_time) == 0 or len(ff_torque) == 0:
            continue

        for controller in CONTROLLERS:
            key = f"{method}_{controller}"
            data = feedback_vibration.get(key)
            if not data:
                continue
            run_mode = str(data.get("run_mode", "")).lower()
            if run_mode and run_mode != "combined":
                continue
            fb_time = data.get("time", np.array([]))
            fb_torque = data.get("torque", np.array([]))
            if len(fb_time) == 0 or len(fb_torque) == 0:
                continue

            time, ff_sig, fb_sig = _common_time_series(ff_time, ff_torque, fb_time, fb_torque)
            if len(time) == 0:
                continue

            params = _choose_psd_params(time, ff_sig)
            if not params:
                continue
            freq, coh = signal.coherence(
                ff_sig,
                fb_sig,
                fs=params["fs"],
                window=params["window"],
                nperseg=params["nperseg"],
                noverlap=params["noverlap"],
                detrend=params["detrend"],
            )
            mask = (freq >= 0) & (freq <= 10.0) & np.isfinite(coh)
            if not np.any(mask):
                continue

            ax = axes[plot_count]
            ax.plot(freq[mask], coh[mask], color="#1f77b4", linewidth=2.0, label="Coherence (FF vs FB)")
            ax.set_title(_combo_label(method, controller), fontweight="bold")
            ax.set_ylim(0.0, 1.0)
            ax.grid(True, alpha=0.3, which="both")
            ax.legend(loc="upper right", fontsize=8)
            plot_count += 1
            if plot_count >= len(axes):
                break
        if plot_count >= len(axes):
            break

    for extra_ax in axes[plot_count:]:
        extra_ax.axis("off")

    if plot_count == 0:
        plt.close(fig)
        return None

    fig.suptitle("Torque Coherence: Feedforward vs Feedback", fontweight="bold")
    fig.supxlabel("Frequency (Hz)")
    fig.supylabel("Magnitude-Squared Coherence")

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    _ensure_dir(out_dir)
    plot_path = os.path.abspath(os.path.join(out_dir, "mission_torque_psd_coherence.png"))
    plt.savefig(plot_path, dpi=400, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return plot_path


def _export_torque_psd_rms(
    feedforward_torque: Dict[str, Dict[str, np.ndarray]],
    feedback_vibration: Dict[str, Dict[str, object]],
    out_dir: str,
    fmin: float = 0.0,
    fmax: float = 10.0,
) -> Optional[str]:
    """Export band limited RMS torque from PSDs for feedforward, feedback, and total."""
    rows: List[List[str]] = []

    for method in METHODS:
        data = feedforward_torque.get(method)
        if not data:
            continue
        time = data.get("time", np.array([]))
        torque = data.get("torque", np.array([]))
        if len(time) == 0 or len(torque) == 0:
            continue
        time, aligned = _align_series(time, torque)
        torque = aligned[0]
        if len(time) == 0:
            continue
        freq, psd = _compute_psd(time, torque)
        rms = _compute_band_rms(freq, psd, fmin, fmax)
        rows.append(["ff", method, "", f"{fmin:.2f}-{fmax:.2f}", f"{rms:.6e}"])

    for method in METHODS:
        for controller in CONTROLLERS:
            key = f"{method}_{controller}"
            data = feedback_vibration.get(key)
            if not data:
                continue
            run_mode = str(data.get("run_mode", "")).lower()
            if run_mode and run_mode != "combined":
                continue
            time = data.get("time", np.array([]))
            fb_torque = data.get("torque", np.array([]))
            total_torque = data.get("torque_total")
            if total_torque is None or len(total_torque) == 0:
                total_torque = fb_torque
            if len(time) == 0 or len(fb_torque) == 0:
                continue
            time, aligned = _align_series(time, fb_torque, total_torque)
            fb_torque = aligned[0]
            total_torque = aligned[1]
            if len(time) == 0:
                continue
            freq_fb, psd_fb = _compute_psd(time, fb_torque)
            freq_total, psd_total = _compute_psd(time, total_torque)
            rms_fb = _compute_band_rms(freq_fb, psd_fb, fmin, fmax)
            rms_total = _compute_band_rms(freq_total, psd_total, fmin, fmax)
            rows.append(["fb", method, controller, f"{fmin:.2f}-{fmax:.2f}", f"{rms_fb:.6e}"])
            rows.append(["total", method, controller, f"{fmin:.2f}-{fmax:.2f}", f"{rms_total:.6e}"])

    if not rows:
        return None

    _ensure_dir(out_dir)
    path = os.path.abspath(os.path.join(out_dir, "torque_psd_rms.csv"))
    _write_csv(path, ["type", "method", "controller", "band_hz", "rms_torque_nm"], rows)
    print(f"Wrote torque PSD RMS CSV: {path}")
    return path


def _export_torque_command_metrics(
    feedforward_torque: Dict[str, Dict[str, np.ndarray]],
    feedback_vibration: Dict[str, Dict[str, object]],
    config: MissionConfig,
    out_dir: str,
    fmin: float = 0.0,
    fmax: float = 10.0,
) -> Optional[str]:
    """Export time domain actuator metrics for torque commands."""
    rows: List[List[str]] = []
    rw_limit = config.rw_max_torque_nm

    def _metrics_from_signal(time: np.ndarray, torque: np.ndarray) -> Tuple[float, float, float, float]:
        """Return peak, RMS, band limited RMS, and max rate for a torque signal."""
        if len(time) == 0 or len(torque) == 0:
            return float("nan"), float("nan"), float("nan"), float("nan")
        torque = np.array(torque, dtype=float)
        peak = float(np.max(np.abs(torque)))
        rms = float(np.sqrt(np.mean(torque**2)))
        dt = float(np.median(np.diff(time))) if len(time) > 1 else 0.0
        if dt > 0 and len(torque) > 1:
            rate = float(np.max(np.abs(np.diff(torque) / dt)))
        else:
            rate = float("nan")
        freq, psd = _compute_psd(time, torque)
        rms_band = _compute_band_rms(freq, psd, fmin, fmax)
        return peak, rms, rms_band, rate

    for method in METHODS:
        data = feedforward_torque.get(method)
        if not data:
            continue
        time = np.array(data.get("time", []), dtype=float)
        torque = np.array(data.get("torque", []), dtype=float)
        time, aligned = _align_series(time, torque)
        torque = aligned[0]
        peak, rms, rms_band, rate = _metrics_from_signal(time, torque)
        rows.append([
            "ff_body", method, "",
            f"{peak:.6e}", f"{rms:.6e}", f"{rms_band:.6e}", f"{rate:.6e}", "", ""
        ])

    for method in METHODS:
        for controller in CONTROLLERS:
            key = f"{method}_{controller}"
            data = feedback_vibration.get(key)
            if not data:
                continue
            run_mode = str(data.get("run_mode", "")).lower()
            if run_mode and run_mode != "combined":
                continue

            time = np.array(data.get("time", []), dtype=float)
            fb_torque = np.array(data.get("torque", []), dtype=float)
            total_torque = data.get("torque_total")
            if total_torque is None or len(total_torque) == 0:
                total_torque = fb_torque
            total_torque = np.array(total_torque, dtype=float)
            time, aligned = _align_series(time, fb_torque, total_torque)
            fb_torque = aligned[0]
            total_torque = aligned[1]

            fb_peak, fb_rms, fb_rms_band, fb_rate = _metrics_from_signal(time, fb_torque)
            rows.append([
                "fb_body", method, controller,
                f"{fb_peak:.6e}", f"{fb_rms:.6e}", f"{fb_rms_band:.6e}", f"{fb_rate:.6e}", "", ""
            ])

            tot_peak, tot_rms, tot_rms_band, tot_rate = _metrics_from_signal(time, total_torque)
            rows.append([
                "total_body", method, controller,
                f"{tot_peak:.6e}", f"{tot_rms:.6e}", f"{tot_rms_band:.6e}", f"{tot_rate:.6e}", "", ""
            ])

            rw_torque = data.get("rw_torque")
            if rw_torque is not None and len(rw_torque) > 0:
                rw_torque = np.array(rw_torque, dtype=float)
                if rw_torque.ndim == 1:
                    rw_env = np.abs(rw_torque)
                else:
                    rw_env = np.max(np.abs(rw_torque), axis=1)
                rw_peak, rw_rms, rw_rms_band, rw_rate = _metrics_from_signal(time, rw_env)
                sat_pct = ""
                if rw_limit is not None and np.isfinite(rw_limit) and rw_limit > 0:
                    sat_pct = f"{(rw_peak / rw_limit) * 100.0:.2f}"
                rows.append([
                    "rw_wheel", method, controller,
                    f"{rw_peak:.6e}", f"{rw_rms:.6e}", f"{rw_rms_band:.6e}", f"{rw_rate:.6e}",
                    f"{rw_peak:.6e}", sat_pct
                ])

    if not rows:
        return None

    _ensure_dir(out_dir)
    path = os.path.abspath(os.path.join(out_dir, "torque_command_metrics.csv"))
    _write_csv(
        path,
        [
            "type",
            "method",
            "controller",
            "peak_torque_nm",
            "rms_torque_nm",
            "rms_band_0_10hz_nm",
            "max_rate_nm_s",
            "peak_rw_torque_nm",
            "rw_saturation_percent",
        ],
        rows,
    )
    print(f"Wrote torque command metrics CSV: {path}")
    return path


def _format_margin_label(margins: Dict[str, float]) -> str:
    """Format stability margins for legend."""
    gm = margins.get("gain_margin_db", float("inf"))
    pm = margins.get("phase_margin_deg", float("inf"))
    gm_str = f"GM={gm:.1f}dB" if np.isfinite(gm) else "GM=inf"
    pm_str = f"PM={pm:.1f}deg" if np.isfinite(pm) else "PM=inf"
    return f"{gm_str}, {pm_str}"


def _plot_nyquist(
    control_data: Dict[str, object],
    out_dir: str,
) -> Optional[str]:
    """Plot Nyquist diagram."""
    l_data = control_data.get("L_flex") or control_data["L"]
    margins = control_data["margins"]

    plt.rcParams.update({
        "font.size": 10,
        "axes.labelsize": 11,
        "axes.titlesize": 12,
        "legend.fontsize": 9,
    })

    fig, (ax_full, ax_zoom) = plt.subplots(1, 2, figsize=(12, 6))

    line_styles = {"standard_pd": "-", "filtered_pd": "--"}
    nyq_traces = []

    for name in CONTROLLERS:
        l_resp = l_data[name]
        nyq = np.concatenate([l_resp, np.conjugate(l_resp[::-1])])
        nyq_traces.append(nyq)
        ls = line_styles.get(name, "-")
        label = f"{CONTROLLER_LABELS[name]} ({_format_margin_label(margins[name])})"
        ax_full.plot(nyq.real, nyq.imag, color=CONTROLLER_COLORS[name], label=label,
                     linewidth=1.5, linestyle=ls)
        ax_zoom.plot(nyq.real, nyq.imag, color=CONTROLLER_COLORS[name],
                     linewidth=1.5, linestyle=ls)

    # Critical point
    for ax in (ax_full, ax_zoom):
        ax.plot(-1, 0, marker="o", markersize=14, markerfacecolor="none",
                markeredgecolor="red", markeredgewidth=2.5)
        ax.plot(-1, 0, "rx", markersize=12, markeredgewidth=2.5,
                label="Critical Point (-1, 0)" if ax is ax_full else None)
    ax_zoom.annotate(
        "(-1, 0)",
        xy=(-1, 0),
        xytext=(-1.6, 0.8),
        textcoords="data",
        color="red",
        fontsize=10,
        arrowprops=dict(arrowstyle="->", color="red", linewidth=1.2),
    )

    ax_full.set_title("Nyquist Diagram", fontweight="bold")
    ax_full.set_xlabel("Real")
    ax_full.set_ylabel("Imaginary")
    ax_full.grid(True, alpha=0.3)
    ax_full.legend(loc="lower left")
    ax_full.set_aspect("equal")
    ax_full.axhline(0, color="black", linewidth=0.5)
    ax_full.axvline(0, color="black", linewidth=0.5)

    ax_zoom.set_title("Zoom Near (-1, 0)", fontweight="bold")
    ax_zoom.set_xlabel("Real")
    ax_zoom.set_ylabel("Imaginary")
    ax_zoom.grid(True, alpha=0.3)
    ax_zoom.set_aspect("equal")
    ax_zoom.axhline(0, color="black", linewidth=0.5)
    ax_zoom.axvline(0, color="black", linewidth=0.5)

    if nyq_traces:
        all_points = np.concatenate(nyq_traces)
        center = -1.0 + 0.0j
        near = all_points[np.abs(all_points - center) < 2.0]
        if len(near) > 0:
            real_span = float(np.max(np.abs(near.real + 1.0)))
            imag_span = float(np.max(np.abs(near.imag)))
            half_span = max(0.2, 1.2 * max(real_span, imag_span))
        else:
            half_span = 1.0
        ax_zoom.set_xlim([-1.0 - half_span, -1.0 + half_span])
        ax_zoom.set_ylim([-half_span, half_span])

    plt.tight_layout()
    _ensure_dir(out_dir)
    plot_path = os.path.abspath(os.path.join(out_dir, "mission_nyquist.png"))
    plt.savefig(plot_path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return plot_path


def _plot_disturbance_transfer(
    control_data: Dict[str, object],
    config: MissionConfig,
    out_dir: str,
) -> Optional[str]:
    """Plot disturbance torque to pointing error transfer."""
    freqs = control_data.get("freqs")
    plant = control_data.get("plant_flex_body")
    disturbance = control_data.get("disturbance_body")
    if freqs is None or plant is None or disturbance is None:
        return None

    plt.rcParams.update({
        "font.size": 10,
        "axes.labelsize": 11,
        "axes.titlesize": 12,
        "legend.fontsize": 9,
    })

    fig, ax = plt.subplots(1, 1, figsize=(10, 6))

    sigma_to_deg = 4.0 * 180.0 / np.pi
    open_mag_db = 20 * np.log10(np.abs(plant * sigma_to_deg) + 1e-12)
    ax.semilogx(freqs, open_mag_db, color="black", linewidth=1.8, label="Open-loop (body plant)")

    line_styles = {"standard_pd": "-"}
    for name in CONTROLLERS:
        if name not in disturbance:
            continue
        mag_db = 20 * np.log10(np.abs(disturbance[name] * sigma_to_deg) + 1e-12)
        ax.semilogx(
            freqs,
            mag_db,
            color=CONTROLLER_COLORS[name],
            linewidth=1.6,
            linestyle=line_styles.get(name, "-"),
            label=f"Closed-loop ({CONTROLLER_LABELS[name]})",
        )

    for f_mode in config.modal_freqs_hz:
        ax.axvline(f_mode, color="gray", linestyle=":", alpha=0.7)

    ax.set_title("Disturbance Torque to Pointing Error", fontweight="bold")
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Magnitude (dB, deg/Nm)")
    ax.grid(True, alpha=0.3, which="both")
    ax.legend(loc="lower left")
    ax.set_xlim([freqs[0], 10])

    plt.tight_layout()
    _ensure_dir(out_dir)
    plot_path = os.path.abspath(os.path.join(out_dir, "mission_disturbance_tf.png"))
    plt.savefig(plot_path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return plot_path


def _plot_noise_to_torque(
    control_data: Dict[str, object],
    out_dir: str,
) -> Optional[str]:
    """Plot measurement noise to commanded torque transfer: C / (1 + P*C)."""
    freqs = control_data.get("freqs")
    controller_resp = control_data.get("controller_resp")
    loop = control_data.get("L_flex") or control_data.get("L")
    if freqs is None or controller_resp is None or loop is None:
        return None

    plt.rcParams.update({
        "font.size": 10,
        "axes.labelsize": 11,
        "axes.titlesize": 12,
        "legend.fontsize": 9,
    })

    fig, ax = plt.subplots(1, 1, figsize=(10, 6))

    for name in CONTROLLERS:
        c_resp = controller_resp.get(name)
        l_resp = loop.get(name) if isinstance(loop, dict) else None
        if c_resp is None or l_resp is None:
            continue
        noise_to_tau = c_resp / (1.0 + l_resp)
        mag_db = 20 * np.log10(np.abs(noise_to_tau) + 1e-12)
        ax.semilogx(
            freqs,
            mag_db,
            color=CONTROLLER_COLORS.get(name, "#333333"),
            linewidth=1.6,
            linestyle="-",
            label=CONTROLLER_LABELS.get(name, name),
        )

    ax.set_title("Noise to Commanded Torque |C/(1+P*C)|", fontweight="bold")
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Magnitude (dB)")
    ax.grid(True, alpha=0.3, which="both")
    ax.legend(loc="lower left")
    ax.set_xlim([freqs[0], 10])

    plt.tight_layout()
    _ensure_dir(out_dir)
    plot_path = os.path.abspath(os.path.join(out_dir, "mission_noise_to_torque.png"))
    plt.savefig(plot_path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return plot_path


def _plot_noise_to_pointing(
    control_data: Dict[str, object],
    config: MissionConfig,
    out_dir: str,
) -> Optional[str]:
    """Plot rate gyro noise to pointing error transfer: P * C_omega / (1 + P*C)."""
    freqs = control_data.get("freqs")
    plant = control_data.get("plant_flex_body")
    loop = control_data.get("L_flex") or control_data.get("L")
    rate_resp = control_data.get("rate_path_resp")
    if freqs is None or plant is None or loop is None or rate_resp is None:
        return None

    plt.rcParams.update({
        "font.size": 10,
        "axes.labelsize": 11,
        "axes.titlesize": 12,
        "legend.fontsize": 9,
    })

    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    sigma_to_deg = 4.0 * 180.0 / np.pi
    line_styles = {"standard_pd": "-"}

    for name in CONTROLLERS:
        l_resp = loop.get(name) if isinstance(loop, dict) else None
        c_rate = rate_resp.get(name)
        if l_resp is None or c_rate is None:
            continue
        noise_to_point = plant * c_rate / (1.0 + l_resp)
        mag_db = 20 * np.log10(np.abs(noise_to_point * sigma_to_deg) + 1e-12)
        ax.semilogx(
            freqs,
            mag_db,
            color=CONTROLLER_COLORS.get(name, "#333333"),
            linewidth=1.6,
            linestyle=line_styles.get(name, "-"),
            label=CONTROLLER_LABELS.get(name, name),
        )

    for f_mode in config.modal_freqs_hz:
        ax.axvline(f_mode, color="gray", linestyle=":", alpha=0.7)

    ax.set_title("Rate Noise to Pointing Error |P*Cω/(1+P*C)|", fontweight="bold")
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Magnitude (dB, deg/(rad/s))")
    ax.grid(True, alpha=0.3, which="both")
    ax.legend(loc="lower left")
    ax.set_xlim([freqs[0], 10])

    plt.tight_layout()
    _ensure_dir(out_dir)
    plot_path = os.path.abspath(os.path.join(out_dir, "mission_noise_to_pointing.png"))
    plt.savefig(plot_path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return plot_path


def _plot_loop_components(
    control_data: Dict[str, object],
    config: MissionConfig,
    out_dir: str,
) -> List[str]:
    """Plot plant, controller, and loop transfer functions for key controllers."""
    freqs = control_data.get("freqs")
    plant = control_data.get("plant_flex_body")
    controller_resp = control_data.get("controller_resp", {})
    loop = control_data.get("L_flex") or control_data.get("L")

    if freqs is None or plant is None or loop is None or not controller_resp:
        return []

    plt.rcParams.update({
        "font.size": 10,
        "axes.labelsize": 11,
        "axes.titlesize": 12,
        "legend.fontsize": 9,
    })

    def _plot_one(ctrl_key: str, title: str, filename: str) -> Optional[str]:
        """Plot plant, controller, and loop magnitude for one controller."""
        c_resp = controller_resp.get(ctrl_key)
        if c_resp is None or ctrl_key not in loop:
            return None
        l_resp = loop[ctrl_key]

        fig, ax = plt.subplots(1, 1, figsize=(10, 6))
        plant_db = 20 * np.log10(np.abs(plant) + 1e-12)
        ctrl_db = 20 * np.log10(np.abs(c_resp) + 1e-12)
        loop_db = 20 * np.log10(np.abs(l_resp) + 1e-12)

        ax.semilogx(freqs, plant_db, color="black", linewidth=1.8, label="Plant |P|")
        ax.semilogx(freqs, ctrl_db, color="#1f77b4", linewidth=1.6, label="Controller |C|")
        ax.semilogx(freqs, loop_db, color="#d62728", linewidth=1.8, label="Loop |P*C|")

        for f_mode in config.modal_freqs_hz:
            ax.axvline(f_mode, color="gray", linestyle=":", alpha=0.6)

        ax.set_title(title, fontweight="bold")
        ax.set_xlabel("Frequency (Hz)")
        ax.set_ylabel("Magnitude (dB)")
        ax.grid(True, alpha=0.3, which="both")
        ax.legend(loc="best")
        ax.set_xlim([freqs[0], 10])

        plt.tight_layout()
        _ensure_dir(out_dir)
        plot_path = os.path.abspath(os.path.join(out_dir, filename))
        plt.savefig(plot_path, dpi=200, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        return plot_path

    plot_paths: List[str] = []

    # Loop components plot for the configured comparison controller.
    pd_plot = _plot_one(
        "standard_pd",
        "Loop Components: Standard PD",
        "mission_loop_components_standard_pd.png",
    )
    if pd_plot:
        plot_paths.append(pd_plot)

    return plot_paths


# ===========================================================================
# CSV Export Functions
# ===========================================================================


def _export_vibration_csv(
    feedforward_data: Dict[str, Dict[str, object]],
    feedback_data: Dict[str, Dict[str, object]],
    out_dir: str,
) -> None:
    """Export vibration data to CSV."""
    _ensure_dir(out_dir)

    # Feedforward vibration
    rows = []
    for method, data in feedforward_data.items():
        time = data.get("time", [])
        disp = data.get("displacement", [])
        for t, d in zip(time, disp):
            rows.append([f"{t:.6f}", method, f"{d:.6e}"])
    if rows:
        path = os.path.abspath(os.path.join(out_dir, "vibration_feedforward.csv"))
        _write_csv(path, ["time_s", "method", "displacement_m"], rows)
        print(f"Wrote feedforward vibration CSV: {path}")

    # Feedback vibration
    rows = []
    for _, data in feedback_data.items():
        time = data.get("time", [])
        disp = data.get("displacement", [])
        method = data.get("method")
        controller = data.get("controller")
        for t, d in zip(time, disp):
            rows.append([f"{t:.6f}", method or "", controller or "", f"{d:.6e}"])
    if rows:
        path = os.path.abspath(os.path.join(out_dir, "vibration_feedback.csv"))
        _write_csv(path, ["time_s", "method", "controller", "displacement_m"], rows)
        print(f"Wrote feedback vibration CSV: {path}")


def _export_pointing_error_csv(
    pointing_data: Dict[str, Dict[str, object]],
    out_dir: str,
    config: Optional[MissionConfig] = None,
) -> None:
    """Export pointing error data to CSV."""
    _ensure_dir(out_dir)
    rows = []
    for method, method_data in pointing_data.items():
        for controller, data in method_data.items():
            time = data.get("time", [])
            errors = _extract_pointing_error(data, config=config)
            if len(errors) == 0:
                continue
            time, aligned = _align_series(np.array(time, dtype=float), errors)
            errors = aligned[0]
            for t, e in zip(time, errors):
                rows.append([f"{t:.6f}", method, controller, f"{e:.6e}"])
    if rows:
        path = os.path.abspath(os.path.join(out_dir, "pointing_error.csv"))
        _write_csv(path, ["time_s", "method", "controller", "error_deg"], rows)
        print(f"Wrote pointing error CSV: {path}")


def _export_psd_csv(
    feedforward_data: Dict[str, Dict[str, object]],
    feedback_data: Dict[str, Dict[str, object]],
    out_dir: str,
    mission_psd_data: Optional[Dict[str, Dict[str, Dict[str, object]]]] = None,
) -> None:
    """Export PSD data to CSV."""
    _ensure_dir(out_dir)

    # Feedforward PSD
    rows = []
    for method, data in feedforward_data.items():
        psd_freq = data.get("psd_freq", [])
        psd_vals = data.get("psd", [])
        for f, p in zip(psd_freq, psd_vals):
            rows.append([f"{f:.6f}", method, f"{p:.6e}"])
    if rows:
        path = os.path.abspath(os.path.join(out_dir, "psd_feedforward.csv"))
        _write_csv(path, ["frequency_hz", "method", "psd_n2m2_per_hz"], rows)
        print(f"Wrote feedforward PSD CSV: {path}")

    # Feedback PSD
    rows = []
    for _, data in feedback_data.items():
        psd_freq = data.get("psd_freq", [])
        psd_vals = data.get("psd", [])
        method = data.get("method")
        controller = data.get("controller")
        for f, p in zip(psd_freq, psd_vals):
            rows.append([f"{f:.6f}", method or "", controller or "", f"{p:.6e}"])
    if rows:
        path = os.path.abspath(os.path.join(out_dir, "psd_feedback.csv"))
        _write_csv(path, ["frequency_hz", "method", "controller", "psd_n2m2_per_hz"], rows)
        print(f"Wrote feedback PSD CSV: {path}")

    # Mission PSD
    if mission_psd_data:
        rows = []
        for method in METHODS:
            if method not in mission_psd_data:
                continue
            for name in CONTROLLERS:
                if name not in mission_psd_data[method]:
                    continue
                psd_freq = mission_psd_data[method][name]["psd_freq"]
                psd_vals = mission_psd_data[method][name]["psd"]
                for f, p in zip(psd_freq, psd_vals):
                    rows.append([f"{f:.6f}", method, name, f"{p:.6e}"])
        if rows:
            path = os.path.abspath(os.path.join(out_dir, "psd_mission.csv"))
            _write_csv(path, ["frequency_hz", "method", "controller", "psd_disp_m2_per_hz"], rows)
            print(f"Wrote mission PSD CSV: {path}")


def _export_sensitivity_csv(control_data: Dict[str, object], out_dir: str) -> None:
    """Export sensitivity function data to CSV."""
    _ensure_dir(out_dir)
    freqs = control_data["freqs"]
    rows = []
    for i, f in enumerate(freqs):
        for name in CONTROLLERS:
            s_mag = 20 * np.log10(np.abs(control_data["S"][name][i]) + 1e-12)
            t_mag = 20 * np.log10(np.abs(control_data["T"][name][i]) + 1e-12)
            rows.append([f"{f:.6f}", name, f"{s_mag:.4f}", f"{t_mag:.4f}"])
    path = os.path.abspath(os.path.join(out_dir, "sensitivity_curves.csv"))
    _write_csv(path, ["frequency_hz", "controller", "S_mag_db", "T_mag_db"], rows)
    print(f"Wrote sensitivity curves CSV: {path}")

    if "S_flex" in control_data and "T_flex" in control_data:
        rows = []
        for i, f in enumerate(freqs):
            for name in CONTROLLERS:
                s_mag = 20 * np.log10(np.abs(control_data["S_flex"][name][i]) + 1e-12)
                t_mag = 20 * np.log10(np.abs(control_data["T_flex"][name][i]) + 1e-12)
                rows.append([f"{f:.6f}", name, f"{s_mag:.4f}", f"{t_mag:.4f}"])
        path = os.path.abspath(os.path.join(out_dir, "sensitivity_curves_flexible.csv"))
        _write_csv(path, ["frequency_hz", "controller", "S_mag_db", "T_mag_db"], rows)
        print(f"Wrote flexible sensitivity curves CSV: {path}")


def _export_nyquist_csv(control_data: Dict[str, object], out_dir: str) -> None:
    """Export Nyquist data to CSV."""
    _ensure_dir(out_dir)
    freqs = control_data["freqs"]
    l_data = control_data.get("L_flex") or control_data["L"]
    rows = []
    for i, f in enumerate(freqs):
        for name in CONTROLLERS:
            l_val = l_data[name][i]
            rows.append([f"{f:.6f}", name, f"{l_val.real:.6e}", f"{l_val.imag:.6e}"])
    path = os.path.abspath(os.path.join(out_dir, "nyquist_curves.csv"))
    _write_csv(path, ["frequency_hz", "controller", "L_real", "L_imag"], rows)
    print(f"Wrote nyquist curves CSV: {path}")


def _export_mission_summary_csv(
    ff_metrics: Dict[str, Dict[str, float]],
    ctrl_metrics: Dict[str, Dict[str, float]],
    pointing_metrics: Dict[str, Dict[str, float]],
    out_dir: str,
) -> None:
    """Export mission summary to CSV."""
    _ensure_dir(out_dir)
    rows = []

    # Feedforward metrics
    for method, m in ff_metrics.items():
        rows.append([
            "feedforward", method, "",
            f"{m.get('rms_torque_nm', 0):.4f}",
            f"{m.get('peak_torque_nm', 0):.4f}",
            f"{m.get('rms_vibration_mm', 0):.4f}",
        ])

    # Control metrics
    for name, m in ctrl_metrics.items():
        rows.append([
            "control", "", name,
            f"{m.get('gain_margin_db', 0):.2f}",
            f"{m.get('phase_margin_deg', 0):.2f}",
            "",
        ])

    # Pointing metrics
    for key, m in pointing_metrics.items():
        rows.append([
            "pointing", key, "",
            f"{m.get('rms_vibration_mm', 0):.4f}",
            f"{m.get('rms_pointing_error_deg', 0):.6f}",
            "",
        ])

    path = os.path.abspath(os.path.join(out_dir, "mission_summary.csv"))
    _write_csv(path, ["category", "method", "controller", "metric1", "metric2", "metric3"], rows)
    print(f"Wrote mission summary CSV: {path}")


# ===========================================================================
# Main Mission Simulation
# ===========================================================================


def run_mission_simulation(
    config: MissionConfig,
    out_dir: str = None,
    data_dir: Optional[str] = None,
    make_plots: bool = True,
    export_csv: bool = True,
    generate_pointing: bool = False,
) -> Dict[str, object]:
    """Run complete mission simulation analysis.

    Executes feedforward comparison, control stability analysis, and pointing
    error evaluation. Optionally generates plots and CSV exports for all
    computed metrics. Returns a dictionary containing feedforward, control,
    and pointing results along with saved plot paths.
    """
    script_dir = os.path.dirname(__file__)
    out_dir = out_dir or os.path.join(script_dir, "..", "output")
    data_dir = data_dir or os.path.join(script_dir, "..", "data", "trajectories")
    plots_dir = os.path.join(out_dir, "plots")
    metrics_dir = os.path.join(out_dir, "metrics")

    # Run analyses
    ff_metrics = run_feedforward_comparison(
        config,
        metrics_dir,
        make_plots=False,
        export_csv=export_csv,
        data_dir=data_dir,
        prefer_npz=True,
    )

    ctrl_data = _compute_control_analysis(config)
    ctrl_metrics = {}
    for name in CONTROLLERS:
        ctrl_metrics[name] = ctrl_data["margins"][name]

    pointing_metrics = run_pointing_summary(
        config,
        metrics_dir,
        data_dir=data_dir,
        make_plots=False,
        export_csv=export_csv,
        generate_pointing=generate_pointing,
    )

    # Build data structures for plotting
    feedforward_torque, feedforward_vibration = _collect_feedforward_data(
        config, data_dir=data_dir, prefer_npz=True
    )
    feedback_vibration = _collect_feedback_data(config, data_dir=data_dir, prefer_npz=True)

    # Mission vibration PSD data (combined FF + FB)
    mission_psd_data = _build_mission_psd_data(
        config, data_dir=data_dir, generate_if_missing=generate_pointing
    )

    # Pointing data
    pointing_data = _load_all_pointing_data(
        data_dir,
        config=config,
        generate_if_missing=generate_pointing,
    )

    # Export remaining CSVs
    if export_csv:
        _export_vibration_csv(feedforward_vibration, feedback_vibration, metrics_dir)
        _export_pointing_error_csv(pointing_data, metrics_dir, config=config)
        _export_psd_csv(feedforward_torque, feedback_vibration, metrics_dir, mission_psd_data)
        _export_sensitivity_csv(ctrl_data, metrics_dir)
        _export_nyquist_csv(ctrl_data, metrics_dir)
        _export_mission_summary_csv(ff_metrics, ctrl_metrics, pointing_metrics, metrics_dir)
        _export_torque_psd_rms(feedforward_torque, feedback_vibration, metrics_dir)
        _export_torque_command_metrics(feedforward_torque, feedback_vibration, config, metrics_dir)

    # Generate plots
    plot_paths: List[str] = []
    if make_plots:
        vibration_plot = _plot_vibration_comparison(
            feedforward_vibration, feedback_vibration, config, plots_dir
        )
        modal_acc_psd_plot = _plot_modal_acceleration_psd(feedback_vibration, config, plots_dir)
        sensitivity_plot = _plot_sensitivity_functions(ctrl_data, config, plots_dir)
        modal_plot = _plot_modal_excitation(ctrl_data, config, plots_dir)
        pointing_plot = _plot_pointing_error(pointing_data, plots_dir, config=config)
        psd_plot = _plot_psd_comparison(mission_psd_data, config, plots_dir)
        nyquist_plot = _plot_nyquist(ctrl_data, plots_dir)
        disturbance_plot = _plot_disturbance_transfer(ctrl_data, config, plots_dir)
        noise_torque_plot = _plot_noise_to_torque(ctrl_data, plots_dir)
        noise_pointing_plot = _plot_noise_to_pointing(ctrl_data, config, plots_dir)
        tracking_plot = _plot_tracking_response(pointing_data, config, plots_dir)
        tracking_psd_plot = _plot_tracking_error_psd(pointing_data, ctrl_data, config, plots_dir)
        tracking_tf_plot = _plot_tracking_transfer(ctrl_data, plots_dir)
        disturbance_to_torque_plot = _plot_disturbance_to_torque(ctrl_data, plots_dir)
        loop_component_plots = _plot_loop_components(ctrl_data, config, plots_dir)
        torque_cmd_plot = _plot_torque_command_time(feedback_vibration, plots_dir)
        torque_cmd_psd_plot = _plot_torque_command_psd(feedback_vibration, ctrl_data, config, plots_dir)
        torque_psd_split_plot = _plot_torque_psd_split(feedforward_torque, feedback_vibration, ctrl_data, config, plots_dir)
        torque_psd_coherence_plot = _plot_torque_psd_coherence(feedforward_torque, feedback_vibration, plots_dir)

        for plot in [
            vibration_plot,
            modal_acc_psd_plot,
            sensitivity_plot,
            modal_plot,
            pointing_plot,
            psd_plot,
            nyquist_plot,
            disturbance_plot,
            noise_torque_plot,
            noise_pointing_plot,
            tracking_plot,
            tracking_psd_plot,
            tracking_tf_plot,
            disturbance_to_torque_plot,
            torque_cmd_plot,
            torque_cmd_psd_plot,
            torque_psd_split_plot,
            torque_psd_coherence_plot,
        ]:
            if plot:
                plot_paths.append(plot)
                print(f"Saved plot: {plot}")

        if loop_component_plots:
            for plot in loop_component_plots:
                if plot:
                    plot_paths.append(plot)
                    print(f"Saved plot: {plot}")

        mirror_dir = plots_dir  # Output already goes to the right place
        for plot in plot_paths:
            _mirror_output(plot, mirror_dir)

    if make_plots:
        print(f"Plots are saved in: {os.path.abspath(plots_dir)}")
    if export_csv:
        print(f"CSV exports are saved in: {os.path.abspath(metrics_dir)}")

    return {
        "feedforward": ff_metrics,
        "control": ctrl_metrics,
        "pointing": pointing_metrics,
        "plots": plot_paths,
    }


# ===========================================================================
# CLI
# ===========================================================================


def main() -> None:
    """Parse command line arguments and run the mission simulation."""
    parser = argparse.ArgumentParser(description="Mission simulation analysis")
    parser.add_argument("--out-dir", default=None, help="Output directory (default: ../output)")
    parser.add_argument("--data-dir", default=None, help="Data directory for NPZ files")
    parser.add_argument("--no-plots", action="store_true", help="Skip plot generation")
    parser.add_argument("--no-csv", action="store_true", help="Skip CSV export")
    parser.add_argument("--generate-pointing", action="store_true",
                        help="Generate pointing data by running vizard_demo.py")
    args = parser.parse_args()

    config = default_config()
    run_mission_simulation(
        config,
        out_dir=args.out_dir,
        data_dir=args.data_dir,
        make_plots=not args.no_plots,
        export_csv=not args.no_csv,
        generate_pointing=args.generate_pointing,
    )


if __name__ == "__main__":
    main()

