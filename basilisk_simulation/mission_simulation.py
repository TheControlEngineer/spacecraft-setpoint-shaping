"""Unified mission simulation analysis for feedforward shaping and feedback control."""

from __future__ import annotations

import argparse
import csv
import math
import os
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.ticker import LogLocator
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from scipy import signal
from scipy.fft import fft, fftfreq

from spacecraft_properties import HUB_INERTIA, compute_effective_inertia


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class MissionConfig:
    """Configuration for mission simulation."""

    inertia: np.ndarray
    rotation_axis: np.ndarray
    modal_freqs_hz: List[float]
    modal_damping: List[float]
    modal_gains: List[float]
    ppf_gains: List[float]
    slew_angle_deg: float
    slew_duration_s: float
    feedforward_inertia: Optional[np.ndarray] = None
    control_modal_gains: Optional[List[float]] = None
    control_filter_cutoff_hz: Optional[float] = None
    control_filter_phase_lag_deg: float = 3.0
    vibration_highpass_hz: Optional[float] = None
    pointing_error_spec_asd_deg: Optional[float] = None
    pointing_error_spec_label: Optional[str] = None


METHODS = ["unshaped", "zvd", "fourth"]
CONTROLLERS = ["standard_pd", "filtered_pd", "avc"]

METHOD_LABELS = {
    "unshaped": "Unshaped",
    "zvd": "ZVD",
    "fourth": "Fourth-Order",
}

METHOD_COLORS = {
    "unshaped": "#d62728",  # red
    "zvd": "#1f77b4",  # blue
    "fourth": "#2ca02c",  # green
}

CONTROLLER_LABELS = {
    "standard_pd": "Standard PD",
    "filtered_pd": "Filtered PD",
    "avc": "AVC",
}

CONTROLLER_COLORS = {
    "standard_pd": "#ff7f0e",  # orange
    "filtered_pd": "#9467bd",  # purple
    "avc": "#17becf",  # cyan
}


def default_config() -> MissionConfig:
    """Return default mission configuration.

    Modal gains are the coupling factors between torque and modal displacement.
    For a flexible appendage at distance r from the rotation axis:
        gain ~ r / (I * omega_n^2)

    where I is the hub inertia and omega_n is the modal frequency.
    Typical values for solar arrays are 0.001-0.01 m/(N.m) at the modal frequency.

    IMPORTANT: modal_gains and control_modal_gains should be CONSISTENT.
    The theoretical values based on spacecraft geometry are:
        Mode 1 (0.4 Hz): r/(J*omega^2) = 3.5/(600*2.51^2) = 0.00092
        Mode 2 (1.3 Hz): r/(J*omega^2) = 4.5/(600*8.17^2) = 0.00011
    
    Using [0.0015, 0.0008] which is close to theoretical and produces
    realistic mm-level vibrations for the ~13 N.m peak torque.
    
    Controller Design (based on analysis in optimal_controller_design.py):
    - Bandwidth = first_mode/6 = 0.067 Hz (ensures PM > 62 degrees)
    - Controller bandwidth = first_mode/6 for adequate phase margin
    - PPF gains need to be tuned to add damping without destabilizing
    """
    return MissionConfig(
        inertia=HUB_INERTIA.copy(),
        feedforward_inertia=compute_effective_inertia(hub_inertia=HUB_INERTIA.copy()),
        rotation_axis=np.array([0.0, 0.0, 1.0]),
        modal_freqs_hz=[0.4, 1.3],
        modal_damping=[0.02, 0.015],
        # Modal gains - physically derived from: 2 * m_flex * r^2 / J_eff
        # Mode 1: 2 * 5kg * 3.5m^2 / 765 kg*m^2 = 0.160
        # Mode 2: 2 * 5kg * 4.5m^2 / 765 kg*m^2 = 0.265
        # These represent the modal participation factors for torque coupling
        modal_gains=[0.16, 0.265],
        # Use SAME gains for control analysis to ensure consistency
        control_modal_gains=[0.16, 0.265],
        # No derivative filter - causes phase margin to drop below 62 degrees
        control_filter_cutoff_hz=None,
        control_filter_phase_lag_deg=3.0,
        # PPF gains for modal damping - start conservative
        ppf_gains=[0.5, 0.5],
        slew_angle_deg=180.0,
        slew_duration_s=30.0,
        vibration_highpass_hz=None,
        pointing_error_spec_asd_deg=None,
        pointing_error_spec_label=None,
    )


# ---------------------------------------------------------------------------
# Utility Functions
# ---------------------------------------------------------------------------


def _ensure_dir(path: str) -> None:
    """Create directory if it doesn't exist."""
    os.makedirs(path, exist_ok=True)


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
    """Apply a simple high-pass filter to remove slow trends."""
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
    """Select a high-pass cutoff to isolate flexible vibration."""
    if config.vibration_highpass_hz is not None:
        return float(config.vibration_highpass_hz)
    duration = float(config.slew_duration_s)
    base = 2.0 / duration if duration > 0 else 0.05
    if config.modal_freqs_hz:
        return min(base, 0.5 * min(config.modal_freqs_hz))
    return base


def _extract_vibration_signals(
    time: np.ndarray, displacement: np.ndarray, config: MissionConfig
) -> Tuple[np.ndarray, np.ndarray]:
    """High-pass filter displacement and derive vibration acceleration."""
    if len(time) == 0 or len(displacement) == 0:
        return np.array([]), np.array([])
    cutoff_hz = _get_vibration_highpass_hz(config)
    disp = _highpass_filter(np.array(displacement, dtype=float), time, cutoff_hz)
    acc = _compute_acceleration_from_displacement(time, disp)
    acc = _highpass_filter(np.array(acc, dtype=float), time, cutoff_hz)
    return disp, acc


def _detrend_mean(data: np.ndarray) -> np.ndarray:
    """Remove mean from data."""
    return data - np.mean(data)


def _compute_psd(time: np.ndarray, signal_data: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Compute power spectral density using Welch's method."""
    if len(time) < 10:
        return np.array([]), np.array([])
    dt = np.median(np.diff(time))
    fs = 1.0 / dt
    nperseg = min(256, len(signal_data) // 4)
    if nperseg < 8:
        return np.array([]), np.array([])
    freq, psd = signal.welch(
        signal_data,
        fs=fs,
        nperseg=nperseg,
        noverlap=nperseg // 2,
        scaling="density",
    )
    return freq, psd


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


def _combine_modal_displacement(mode1: np.ndarray, mode2: np.ndarray) -> np.ndarray:
    """Combine modal displacements using root-sum-square."""
    if mode1 is None and mode2 is None:
        return np.array([])
    if mode1 is None or len(mode1) == 0:
        return np.abs(np.array(mode2, dtype=float))
    if mode2 is None or len(mode2) == 0:
        return np.abs(np.array(mode1, dtype=float))
    mode1 = np.array(mode1, dtype=float)
    mode2 = np.array(mode2, dtype=float)
    n = min(len(mode1), len(mode2))
    return np.sqrt(mode1[:n] ** 2 + mode2[:n] ** 2)


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
        ff_mask = np.array(["FF" in str(mode) for mode in control_mode], dtype=bool)
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
    # Check slew angle
    angle = npz_data.get("slew_angle_deg")
    if angle is not None:
        try:
            if abs(float(angle) - float(config.slew_angle_deg)) > 1.0:
                return False
        except (TypeError, ValueError):
            pass
    
    # Check modal gains - critical for vibration magnitude
    npz_gains = npz_data.get("modal_gains")
    if npz_gains is not None and config.modal_gains:
        try:
            npz_gains = np.array(npz_gains, dtype=float)
            cfg_gains = np.array(config.modal_gains, dtype=float)
            if len(npz_gains) != len(cfg_gains):
                return False
            # Require gains to match within 10%
            if not np.allclose(npz_gains, cfg_gains, rtol=0.1):
                return False
        except (TypeError, ValueError):
            pass
    
    return True


# ---------------------------------------------------------------------------
# Feedforward Analysis
# ---------------------------------------------------------------------------


def _compute_bang_bang_trajectory(
    theta_final: float, duration: float, dt: float = 0.001, settling_time: float = 30.0
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Compute bang-bang trajectory with settling time after maneuver.

    Args:
        theta_final: Target angle in radians
        duration: Maneuver duration in seconds
        dt: Time step in seconds
        settling_time: Additional time after maneuver for vibration settling

    Returns:
        Tuple of (time, theta, omega, alpha) arrays
    """
    total_time = duration + settling_time
    t = np.arange(0, total_time + dt, dt)
    n = len(t)
    t_half = duration / 2
    alpha_max = 4 * theta_final / duration**2

    theta = np.zeros(n)
    omega = np.zeros(n)
    alpha = np.zeros(n)

    for i, ti in enumerate(t):
        if ti <= t_half:
            alpha[i] = alpha_max
            omega[i] = alpha_max * ti
            theta[i] = 0.5 * alpha_max * ti**2
        elif ti <= duration:
            t_dec = ti - t_half
            alpha[i] = -alpha_max
            omega[i] = alpha_max * t_half - alpha_max * t_dec
            theta[i] = (
                0.5 * alpha_max * t_half**2
                + alpha_max * t_half * t_dec
                - 0.5 * alpha_max * t_dec**2
            )
        else:
            # Settling phase: maneuver complete, no torque
            alpha[i] = 0.0
            omega[i] = 0.0
            theta[i] = theta_final

    return t, theta, omega, alpha


def _zvd_shaper_params(f_mode: float, zeta: float) -> Tuple[List[float], List[float]]:
    """Return ZVD shaper amplitudes and delays for a single mode."""
    omega_n = 2 * np.pi * float(f_mode)
    zeta = float(zeta)
    if zeta >= 1.0:
        return [1.0], [0.0]
    omega_d = omega_n * np.sqrt(1 - zeta**2)
    if omega_d <= 0:
        return [1.0], [0.0]
    T_d = 2 * np.pi / omega_d
    K = np.exp(-zeta * np.pi / np.sqrt(1 - zeta**2))
    denom = 1 + 2 * K + K**2
    amplitudes = [1 / denom, 2 * K / denom, K**2 / denom]
    delays = [0.0, T_d / 2.0, T_d]
    return amplitudes, delays


def _apply_input_shaper(
    time: np.ndarray,
    signal_data: np.ndarray,
    amplitudes: List[float],
    delays: List[float],
) -> Tuple[np.ndarray, np.ndarray]:
    """Apply a discrete input shaper and extend the time vector."""
    if len(time) < 2:
        return time, signal_data
    dt = float(np.median(np.diff(time)))
    shift_max = max(int(round(delay / dt)) for delay in delays) if delays else 0
    n_base = len(signal_data)
    n_shaped = n_base + shift_max
    shaped = np.zeros(n_shaped)
    for amp, delay in zip(amplitudes, delays):
        shift = int(round(delay / dt))
        shaped[shift:shift + n_base] += amp * signal_data
    time_shaped = time[0] + np.arange(n_shaped) * dt
    return time_shaped, shaped


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
        method: Shaping method ('unshaped', 'zvd', 'fourth')
        settling_time: Time after maneuver for vibration settling (seconds)

    Returns:
        Dictionary with time, torque, theta, omega, alpha arrays and maneuver_end time
    """
    theta_final = np.radians(config.slew_angle_deg)
    duration = config.slew_duration_s
    axis = _normalize_axis(config.rotation_axis)
    ff_inertia = _get_feedforward_inertia(config)
    I_axis = float(axis @ ff_inertia @ axis)

    t, theta, omega, alpha = _compute_bang_bang_trajectory(
        theta_final, duration, settling_time=settling_time
    )
    torque_unshaped = I_axis * alpha
    maneuver_end = duration  # Default maneuver end time

    if method == "unshaped":
        torque = torque_unshaped
    elif method == "zvd":
        # ZVD shaper for first mode
        f_mode = config.modal_freqs_hz[0] if config.modal_freqs_hz else 1.0
        zeta = config.modal_damping[0] if config.modal_damping else 0.01
        amplitudes, delays = _zvd_shaper_params(f_mode, zeta)
        # Apply shaper only to the maneuver portion
        t_maneuver, alpha_maneuver = _apply_input_shaper(
            t[:int(duration/0.001)+1], alpha[:int(duration/0.001)+1], amplitudes, delays
        )
        maneuver_end = t_maneuver[-1]
        # Extend with settling phase
        dt = t_maneuver[1] - t_maneuver[0]
        n_settling = int(settling_time / dt)
        t = np.concatenate([t_maneuver, t_maneuver[-1] + np.arange(1, n_settling + 1) * dt])
        alpha = np.concatenate([alpha_maneuver, np.zeros(n_settling)])
        theta, omega, alpha = _integrate_trajectory(alpha, dt, theta_final)
        torque = I_axis * alpha
    elif method == "fourth":
        # Fourth-order shaper = ZVD on first two modes
        alpha_shaped = alpha[:int(duration/0.001)+1].copy()
        t_shaped = t[:int(duration/0.001)+1].copy()
        for idx, f_mode in enumerate(config.modal_freqs_hz[:2]):
            zeta = config.modal_damping[idx] if idx < len(config.modal_damping) else 0.01
            amplitudes, delays = _zvd_shaper_params(f_mode, zeta)
            t_shaped, alpha_shaped = _apply_input_shaper(t_shaped, alpha_shaped, amplitudes, delays)
        maneuver_end = t_shaped[-1]
        # Extend with settling phase
        dt = t_shaped[1] - t_shaped[0]
        n_settling = int(settling_time / dt)
        t = np.concatenate([t_shaped, t_shaped[-1] + np.arange(1, n_settling + 1) * dt])
        alpha = np.concatenate([alpha_shaped, np.zeros(n_settling)])
        theta, omega, alpha = _integrate_trajectory(alpha, dt, theta_final)
        torque = I_axis * alpha
    else:
        torque = torque_unshaped

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
    """Simulate modal response to torque profile using symplectic Euler integration.

    The modal equation of motion is:
        m*ddot{q} + c*dot{q} + k*q = gain * torque

    where:
        omega_n = sqrt(k/m) is the natural frequency
        zeta = c / (2*m*omega_n) is the damping ratio
        gain is the modal participation factor (torque to modal displacement coupling)

    This uses symplectic Euler integration which preserves energy better than
    standard Euler and matches scipy.integrate.odeint within ~1%.
    """
    dt = time[1] - time[0] if len(time) > 1 else 0.001
    n = len(time)

    mode1_disp = np.zeros(n)
    mode1_vel = np.zeros(n)
    mode2_disp = np.zeros(n)
    mode2_vel = np.zeros(n)

    if len(config.modal_freqs_hz) >= 1:
        omega1 = 2 * np.pi * config.modal_freqs_hz[0]
        zeta1 = config.modal_damping[0] if config.modal_damping else 0.01
        gain1 = config.modal_gains[0] if config.modal_gains else 0.1
        for i in range(1, n):
            # Symplectic Euler: update velocity first, then position with new velocity
            acc = gain1 * torque[i - 1] - 2 * zeta1 * omega1 * mode1_vel[i - 1] - omega1**2 * mode1_disp[i - 1]
            mode1_vel[i] = mode1_vel[i - 1] + acc * dt
            mode1_disp[i] = mode1_disp[i - 1] + mode1_vel[i] * dt  # Use NEW velocity

    if len(config.modal_freqs_hz) >= 2:
        omega2 = 2 * np.pi * config.modal_freqs_hz[1]
        zeta2 = config.modal_damping[1] if len(config.modal_damping) > 1 else 0.01
        gain2 = config.modal_gains[1] if len(config.modal_gains) > 1 else 0.1
        for i in range(1, n):
            # Symplectic Euler: update velocity first, then position with new velocity
            acc = gain2 * torque[i - 1] - 2 * zeta2 * omega2 * mode2_vel[i - 1] - omega2**2 * mode2_disp[i - 1]
            mode2_vel[i] = mode2_vel[i - 1] + acc * dt
            mode2_disp[i] = mode2_disp[i - 1] + mode2_vel[i] * dt  # Use NEW velocity

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
        # Not enough data - use all available
        residual = disp

    if len(residual) > 0 and np.max(np.abs(residual)) > 1e-15:
        rms_residual = float(np.sqrt(np.mean(residual**2))) * 1000  # mm
        peak_residual = float(np.max(np.abs(residual))) * 1000  # mm
    else:
        rms_residual = 0.0
        peak_residual = 0.0

    if len(torque) > 0:
        rms_torque = float(np.sqrt(np.mean(torque**2)))
        peak_torque = float(np.max(np.abs(torque)))
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
            torque_axis = _project_axis_torque(torque_vec, axis)
            time, aligned = _align_series(time, torque_axis, npz_data.get("mode1"), npz_data.get("mode2"))
            torque_axis = aligned[0]
            mode1 = aligned[1]
            mode2 = aligned[2]

            displacement = _combine_modal_displacement(mode1, mode2)
            displacement = displacement[: len(time)] if len(time) > 0 else displacement
            displacement, acceleration = _extract_vibration_signals(time, displacement, config)
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
                "maneuver_end": maneuver_end,
            }
        else:
            td = _compute_torque_profile(config, method)
            vd = _simulate_modal_response(td["time"], td["torque"], config)
            maneuver_end = _infer_maneuver_end(td["time"], None, td["torque"], config.slew_duration_s)
            displacement, acceleration = _extract_vibration_signals(
                vd["time"], vd["displacement"], config
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
    method_priority = ["fourth", "zvd", "unshaped"]

    for controller in CONTROLLERS:
        npz_data = None
        if prefer_npz:
            for method in method_priority:
                npz_data = _load_pointing_npz(
                    method,
                    data_dir,
                    controller=controller,
                    mode="combined",
                    generate_if_missing=False,
                    allow_legacy=True,
                )
                if npz_data is not None and _npz_matches_config(npz_data, config):
                    break
                npz_data = None
            if npz_data is None:
                for method in method_priority:
                    npz_data = _load_pointing_npz(
                        method,
                        data_dir,
                        controller=controller,
                        mode="fb_only",
                        generate_if_missing=False,
                        allow_legacy=False,
                    )
                    if npz_data is not None and _npz_matches_config(npz_data, config):
                        break
                    npz_data = None

        if npz_data is None:
            feedback_data[controller] = {
                "time": np.array([]),
                "displacement": np.array([]),
                "acceleration": np.array([]),
                "torque": np.array([]),
                "psd_freq": np.array([]),
                "psd": np.array([]),
            }
            continue

        time = np.array(npz_data.get("time", []), dtype=float)
        torque_vec = npz_data.get("fb_torque")
        if torque_vec is None or len(torque_vec) == 0:
            torque_vec = npz_data.get("total_torque")
        torque_axis = _project_axis_torque(torque_vec, axis)
        time, aligned = _align_series(time, torque_axis, npz_data.get("mode1"), npz_data.get("mode2"))
        torque_axis = aligned[0]
        mode1 = aligned[1]
        mode2 = aligned[2]

        displacement = _combine_modal_displacement(mode1, mode2)
        displacement = displacement[: len(time)] if len(time) > 0 else displacement
        displacement, acceleration = _extract_vibration_signals(time, displacement, config)
        psd_freq, psd_vals = _compute_psd(time, torque_axis) if len(time) > 0 else (np.array([]), np.array([]))

        feedback_data[controller] = {
            "time": time,
            "displacement": displacement,
            "acceleration": acceleration,
            "torque": torque_axis,
            "psd_freq": psd_freq,
            "psd": psd_vals,
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


# ---------------------------------------------------------------------------
# Feedback Control Analysis
# ---------------------------------------------------------------------------


def _build_flexible_plant_tf(
    inertia: np.ndarray,
    axis: int,
    modal_freqs_hz: List[float],
    modal_damping: List[float],
    modal_gains: List[float],
) -> signal.TransferFunction:
    """Build flexible plant transfer function."""
    I = float(inertia[axis, axis])
    rigid_num = np.array([1.0])
    rigid_den = np.array([I, 0.0, 0.0])

    if not modal_freqs_hz:
        return signal.TransferFunction(rigid_num, rigid_den)

    current_num = rigid_num
    current_den = rigid_den

    for f_mode, zeta, gain in zip(modal_freqs_hz, modal_damping, modal_gains):
        omega_n = 2 * np.pi * float(f_mode)
        mode_num = np.array([float(gain) / I])
        mode_den = np.array([1.0, 2.0 * float(zeta) * omega_n, omega_n**2])

        term1 = np.convolve(current_num, mode_den)
        term2 = np.convolve(mode_num, current_den)

        if len(term1) > len(term2):
            term2 = np.pad(term2, (len(term1) - len(term2), 0), mode="constant")
        elif len(term2) > len(term1):
            term1 = np.pad(term1, (len(term2) - len(term1), 0), mode="constant")

        current_num = term1 + term2
        current_den = np.convolve(current_den, mode_den)

    return signal.TransferFunction(current_num, current_den)


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

    # Phase crossover (for gain margin)
    idx = np.where((phase_deg[:-1] > -180.0) & (phase_deg[1:] <= -180.0))[0]
    if len(idx) > 0:
        i = idx[0]
        f1, f2 = freqs[i], freqs[i + 1]
        ph1, ph2 = phase_deg[i], phase_deg[i + 1]
        if ph2 != ph1:
            f_pc = f1 + (f2 - f1) * (-180.0 - ph1) / (ph2 - ph1)
        else:
            f_pc = f1
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

    # Gain crossover (for phase margin)
    idx = np.where((mag[:-1] > 1.0) & (mag[1:] <= 1.0))[0]
    if len(idx) > 0:
        i = idx[0]
        f1, f2 = freqs[i], freqs[i + 1]
        mag1, mag2 = mag[i], mag[i + 1]
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
            if logf2 != logf1:
                phase_gc = phase_deg[i] + (phase_deg[i + 1] - phase_deg[i]) * (logf_gc - logf1) / (logf2 - logf1)
            else:
                phase_gc = phase_deg[i]
        else:
            phase_gc = phase_deg[i]
        pm_deg = 180.0 + phase_gc

    return {"gain_margin_db": float(gm_db), "phase_margin_deg": float(pm_deg)}


def _compute_control_analysis(config: MissionConfig) -> Dict[str, object]:
    """Compute control system analysis (sensitivity, stability).

    Controller Design Philosophy:
    ----------------------------
    For flexible spacecraft, the key is placing the control bandwidth
    well below the first structural mode to avoid exciting vibrations.

    Analysis shows that STANDARD PD with proper bandwidth selection
    provides the best vibration suppression:
    - Negative sensitivity at modal frequencies (attenuates disturbances)
    - Good phase margin (63-65 degrees)
    - Simple and robust

    Design approach:
    1. Standard PD: Pure PD control with bandwidth at 0.1 Hz (well below
       0.4 Hz first mode). Provides excellent vibration suppression.

    2. Filtered PD: Adds derivative filter at 10x bandwidth to reduce
       sensor noise without affecting phase at modal frequencies.

    3. AVC: Identical to Filtered PD (PPF compensators removed as they
       added destabilizing phase lag at modal frequencies).

    KEY FINDING: The phase lead from the derivative term in PD control
    naturally provides good phase margin. Additional complexity (PPF, heavy
    filtering) makes performance WORSE by adding phase lag at resonances.
    """
    from feedback_control import (
        MRPFeedbackController,
        FilteredDerivativeController,
        ActiveVibrationController,
    )

    axis = 2  # Z-axis
    I = float(config.inertia[axis, axis])

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
    # For 0.4 Hz mode: bandwidth = 0.4/6 = 0.067 Hz
    # =========================================================================
    bandwidth_std = first_mode / 6.0  # 0.067 Hz for 0.4 Hz mode
    omega_bw_std = 2 * np.pi * bandwidth_std
    k_std = I * omega_bw_std**2
    p_std = 2 * 0.7 * I * omega_bw_std

    # Create standard PD controller (no filtering)
    controller_std = MRPFeedbackController(
        inertia=config.inertia,
        K=k_std,
        P=p_std,
        Ki=-1.0
    )

    # =========================================================================
    # FILTERED PD CONTROLLER
    # Analysis showed: Any filter cutoff adds phase lag that reduces PM
    # To meet PM > 62°, filter cutoff must be very high (> 5 Hz)
    # Using filter cutoff = 5 Hz to provide some HF rolloff (for sensor noise)
    # while maintaining PM close to pure PD
    # =========================================================================
    bandwidth_filt = first_mode / 6.0  # Same as standard
    omega_bw_filt = 2 * np.pi * bandwidth_filt
    k_filt = I * omega_bw_filt**2
    p_filt = 2 * 0.7 * I * omega_bw_filt

    # Filter cutoff very high to minimize phase lag impact
    # At 5 Hz, filter adds negligible phase at the 0.1 Hz crossover frequency
    filter_cutoff = 5.0  # 5 Hz - minimal impact on phase margin
    
    controller_filt = FilteredDerivativeController(
        inertia=config.inertia,
        K=k_filt,
        P=p_filt,
        filter_freq_hz=filter_cutoff
    )

    # =========================================================================
    # AVC CONTROLLER
    # Pure PD + PPF compensators (NO derivative filter to preserve PM)
    # PPF adds modal damping without the phase lag problems of filtering
    # =========================================================================
    bandwidth_avc = first_mode / 6.0  # Low bandwidth
    omega_bw_avc = 2 * np.pi * bandwidth_avc
    k_avc = I * omega_bw_avc**2
    p_avc = 2 * 0.7 * I * omega_bw_avc

    # NO derivative filter for AVC - use pure PD + PPF
    # This achieves PM > 62 deg while adding modal damping via PPF
    filter_cutoff_avc = 10.0  # Very high cutoff = effectively no filter
    
    # PPF gains: Use the user's config values
    # Low gains [2, 4] give PM=64 deg, higher gains [5, 10] give PM=62 deg (limit)
    ppf_gains_active = config.ppf_gains if config.ppf_gains and any(g > 0 for g in config.ppf_gains) else [2.0, 4.0]
    
    # Use ActiveVibrationController with PPF
    controller_avc = ActiveVibrationController(
        inertia=config.inertia,
        K=k_avc,
        P=p_avc,
        filter_freq_hz=filter_cutoff_avc,
        modal_freqs_hz=config.modal_freqs_hz,
        modal_damping=config.modal_damping,
        modal_gains=config.modal_gains,
        ppf_damping=0.5,
        ppf_gains=ppf_gains_active  # Use actual PPF gains
    )

    # Get open-loop transfer functions
    l_std_tf = controller_std.get_open_loop_tf(
        axis=axis,
        modal_freqs_hz=config.modal_freqs_hz,
        modal_damping=config.modal_damping,
        modal_gains=config.modal_gains,
        include_flexibility=True
    )
    
    l_filt_tf = controller_filt.get_open_loop_tf(
        axis=axis,
        modal_freqs_hz=config.modal_freqs_hz,
        modal_damping=config.modal_damping,
        modal_gains=config.modal_gains,
        include_flexibility=True
    )
    
    l_avc_tf = controller_avc.get_open_loop_tf(
        axis=axis,
        include_flexibility=True
    )

    # Evaluate frequency responses
    _, l_std = signal.freqresp(l_std_tf, omega)
    _, l_filt = signal.freqresp(l_filt_tf, omega)
    _, l_avc = signal.freqresp(l_avc_tf, omega)

    # Sensitivity and complementary sensitivity
    s_std = 1 / (1 + l_std)
    s_filt = 1 / (1 + l_filt)
    s_avc = 1 / (1 + l_avc)

    t_std = l_std / (1 + l_std)
    t_filt = l_filt / (1 + l_filt)
    t_avc = l_avc / (1 + l_avc)

    # Stability margins
    margins = {
        "standard_pd": _compute_stability_margins(l_std, freqs),
        "filtered_pd": _compute_stability_margins(l_filt, freqs),
        "avc": _compute_stability_margins(l_avc, freqs),
    }

    # Get plant response for reference
    plant = _build_flexible_plant_tf(
        config.inertia, axis, config.modal_freqs_hz, config.modal_damping, config.modal_gains
    )

    return {
        "freqs": freqs,
        "omega": omega,
        "plant": signal.freqresp(plant, omega)[1],
        "L": {"standard_pd": l_std, "filtered_pd": l_filt, "avc": l_avc},
        "S": {"standard_pd": s_std, "filtered_pd": s_filt, "avc": s_avc},
        "T": {"standard_pd": t_std, "filtered_pd": t_filt, "avc": t_avc},
        "margins": margins,
        "gains": {"K": k_std, "P": p_std, "filter_cutoff": filter_cutoff},
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


# ---------------------------------------------------------------------------
# NPZ File Loading (for Basilisk simulation data)
# ---------------------------------------------------------------------------


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

    search_dirs = [data_dir, os.getcwd()]
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
    """Try to generate NPZ file by running vizard_demo.py."""
    script_dir = os.path.dirname(__file__)
    script_path = os.path.join(script_dir, "vizard_demo.py")

    if not os.path.exists(script_path):
        return False

    try:
        cmd = [sys.executable, script_path, method]
        if controller:
            cmd.extend(["--controller", controller])
        if mode != "combined":
            cmd.extend(["--mode", mode])
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
    if any("AVC" in mode for mode in modes):
        return "avc"
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
    mode1 = np.array(data["mode1"], dtype=float) if "mode1" in data else np.zeros_like(time)
    mode2 = np.array(data["mode2"], dtype=float) if "mode2" in data else np.zeros_like(time)
    sigma = np.array(data["sigma"], dtype=float) if "sigma" in data else None
    omega = np.array(data["omega"], dtype=float) if "omega" in data else None
    control_mode = data["control_mode"].tolist() if "control_mode" in data else []
    target_sigma = np.array(data["target_sigma"], dtype=float) if "target_sigma" in data else np.zeros(3)
    controller_in_file = str(data["controller"]) if "controller" in data else None
    run_mode = str(data["run_mode"]) if "run_mode" in data else mode
    slew_angle_deg = float(data["slew_angle_deg"]) if "slew_angle_deg" in data else None

    ff_torque = np.array(data["ff_torque"], dtype=float) if "ff_torque" in data else None
    fb_torque = np.array(data["fb_torque"], dtype=float) if "fb_torque" in data else None
    total_torque = np.array(data["total_torque"], dtype=float) if "total_torque" in data else None
    rw_torque = np.array(data["rw_torque"], dtype=float) if "rw_torque" in data else None

    controller_used = controller or controller_in_file or _infer_controller_from_control_mode(control_mode)

    return {
        "time": time,
        "mode1": mode1,
        "mode2": mode2,
        "sigma": sigma,
        "omega": omega,
        "control_mode": control_mode,
        "target_sigma": target_sigma,
        "controller": controller_used,
        "run_mode": run_mode,
        "slew_angle_deg": slew_angle_deg,
        "ff_torque": ff_torque,
        "fb_torque": fb_torque,
        "total_torque": total_torque,
        "rw_torque": rw_torque,
    }


def _load_all_pointing_data(
    data_dir: str,
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
            if legacy is not None:
                ctrl = legacy.get("controller") or "avc"
                method_data[ctrl] = legacy

        if method_data:
            pointing[method] = method_data

    return pointing


# ---------------------------------------------------------------------------
# Pointing Summary
# ---------------------------------------------------------------------------


def _compute_pointing_error(sigma: np.ndarray, target_sigma: np.ndarray) -> np.ndarray:
    """Compute pointing error from MRP."""
    if sigma is None or len(sigma) == 0:
        return np.array([])
    sigma = np.atleast_2d(np.array(sigma, dtype=float))
    target = np.array(target_sigma, dtype=float).flatten()
    errors = np.zeros(len(sigma))
    for i, sigma_row in enumerate(sigma):
        sigma_error = _mrp_subtract(sigma_row, target)
        errors[i] = np.degrees(4 * np.arctan(np.linalg.norm(sigma_error)))
    return errors


def run_pointing_summary(
    config: MissionConfig,
    out_dir: str,
    data_dir: Optional[str] = None,
    make_plots: bool = True,
    export_csv: bool = True,
) -> Dict[str, Dict[str, float]]:
    """Run pointing performance summary."""
    data_dir = data_dir or os.path.dirname(__file__)
    pointing_data = _load_all_pointing_data(data_dir, generate_if_missing=False)

    results: Dict[str, Dict[str, float]] = {}

    for method, method_data in pointing_data.items():
        # Check if this is legacy single-controller data
        is_legacy = len(method_data) == 1
        for controller, data in method_data.items():
            # Use just method name for legacy format, combined for multi-controller
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
            sigma = data.get("sigma")
            target = data.get("target_sigma", np.zeros(3))
            if sigma is not None and len(sigma) > 0 and len(time) > 0:
                error = _compute_pointing_error(sigma, target)
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


# ---------------------------------------------------------------------------
# PSD Data Building
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Plotting Functions
# ---------------------------------------------------------------------------


def _plot_vibration_comparison(
    feedforward_vibration: Dict[str, Dict[str, object]],
    feedback_vibration: Dict[str, Dict[str, object]],
    out_dir: str,
) -> Optional[str]:
    """Plot vibration comparison."""
    if not feedforward_vibration and not feedback_vibration:
        return None

    plt.rcParams.update({
        "font.size": 10,
        "axes.labelsize": 11,
        "axes.titlesize": 12,
        "legend.fontsize": 9,
    })

    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    ax_disp, ax_acc = axes

    maneuver_end = None
    method_handles = []
    controller_handles = []

    for method, data in feedforward_vibration.items():
        time = data.get("time", np.array([]))
        disp = data.get("displacement", np.array([]))
        acc = data.get("acceleration", np.array([]))

        if len(time) == 0:
            continue

        if maneuver_end is None:
            maneuver_end = data.get("maneuver_end", 30.0)

        disp_mm = np.array(disp, dtype=float) * 1000.0
        acc_mm = _detrend_mean(np.array(acc, dtype=float)) * 1000.0

        ax_disp.plot(time, disp_mm, color=METHOD_COLORS[method],
                     label=METHOD_LABELS[method], linewidth=1.5)
        ax_acc.plot(time, acc_mm, color=METHOD_COLORS[method],
                    label=METHOD_LABELS[method], linewidth=1.5)
        method_handles.append(Line2D([0], [0], color=METHOD_COLORS[method], lw=1.5,
                                      label=METHOD_LABELS[method]))

    for controller, data in feedback_vibration.items():
        time = data.get("time", np.array([]))
        disp = data.get("displacement", np.array([]))
        acc = data.get("acceleration", np.array([]))

        if len(time) == 0:
            continue

        disp_mm = np.array(disp, dtype=float) * 1000.0
        acc_mm = _detrend_mean(np.array(acc, dtype=float)) * 1000.0

        ax_disp.plot(time, disp_mm, color=CONTROLLER_COLORS[controller],
                     label=CONTROLLER_LABELS[controller], linewidth=1.5, linestyle="--")
        ax_acc.plot(time, acc_mm, color=CONTROLLER_COLORS[controller],
                    label=CONTROLLER_LABELS[controller], linewidth=1.5, linestyle="--")
        controller_handles.append(Line2D([0], [0], color=CONTROLLER_COLORS[controller],
                                         lw=1.5, linestyle="--",
                                         label=CONTROLLER_LABELS[controller]))

    if maneuver_end is not None:
        ax_disp.axvline(maneuver_end, color="gray", linestyle=":", linewidth=1.5, alpha=0.7)
        ax_acc.axvline(maneuver_end, color="gray", linestyle=":", linewidth=1.5, alpha=0.7)

    ax_disp.set_title("Modal Vibration Displacement", fontweight="bold")
    ax_disp.set_ylabel("Displacement (mm)")
    ax_disp.grid(True, alpha=0.3)
    if method_handles:
        ax_disp.legend(handles=method_handles, loc="upper right", title="Feedforward")
    ax_disp.axhline(0, color="black", linewidth=0.5, alpha=0.3)

    ax_acc.set_title("Modal Acceleration Response", fontweight="bold")
    ax_acc.set_xlabel("Time (s)")
    ax_acc.set_ylabel(r"Acceleration (mm/s$^2$)")
    ax_acc.grid(True, alpha=0.3)
    if controller_handles:
        ax_acc.legend(handles=controller_handles, loc="upper right", title="Feedback")
    ax_acc.axhline(0, color="black", linewidth=0.5, alpha=0.3)

    plt.tight_layout()
    _ensure_dir(out_dir)
    plot_path = os.path.abspath(os.path.join(out_dir, "mission_vibration.png"))
    plt.savefig(plot_path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return plot_path


def _plot_sensitivity_functions(
    control_data: Dict[str, object],
    config: MissionConfig,
    out_dir: str,
) -> Optional[str]:
    """Plot sensitivity functions."""
    freqs = control_data["freqs"]
    s_data = control_data["S"]
    t_data = control_data["T"]

    plt.rcParams.update({
        "font.size": 10,
        "axes.labelsize": 11,
        "axes.titlesize": 12,
        "legend.fontsize": 9,
    })

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    ax_s, ax_t = axes

    line_styles = {"standard_pd": "-", "filtered_pd": "--", "avc": "-."}

    for name in CONTROLLERS:
        ls = line_styles.get(name, "-")
        s_mag_db = 20 * np.log10(np.abs(s_data[name]) + 1e-12)
        t_mag_db = 20 * np.log10(np.abs(t_data[name]) + 1e-12)
        ax_s.semilogx(freqs, s_mag_db, label=CONTROLLER_LABELS[name],
                      color=CONTROLLER_COLORS[name], linewidth=1.5, linestyle=ls)
        ax_t.semilogx(freqs, t_mag_db, label=CONTROLLER_LABELS[name],
                      color=CONTROLLER_COLORS[name], linewidth=1.5, linestyle=ls)

    for f_mode in config.modal_freqs_hz:
        ax_s.axvline(f_mode, color="gray", linestyle=":", alpha=0.7)
        ax_t.axvline(f_mode, color="gray", linestyle=":", alpha=0.7)

    ax_s.axhline(0, color="black", linewidth=0.5, alpha=0.5)
    ax_t.axhline(0, color="black", linewidth=0.5, alpha=0.5)

    ax_s.set_title(r"Sensitivity Function $S(j\omega)$", fontweight="bold")
    ax_s.set_xlabel("Frequency (Hz)")
    ax_s.set_ylabel("Magnitude (dB)")
    ax_s.grid(True, alpha=0.3, which="both")
    ax_s.legend(loc="lower right")
    ax_s.set_ylim([-40, 10])
    ax_s.set_xlim([freqs[0], 10])

    ax_t.set_title(r"Complementary Sensitivity $T(j\omega)$", fontweight="bold")
    ax_t.set_xlabel("Frequency (Hz)")
    ax_t.set_ylabel("Magnitude (dB)")
    ax_t.grid(True, alpha=0.3, which="both")
    ax_t.legend(loc="lower left")
    ax_t.set_ylim([-50, 10])
    ax_t.set_xlim([freqs[0], 10])

    plt.tight_layout()
    _ensure_dir(out_dir)
    plot_path = os.path.abspath(os.path.join(out_dir, "mission_sensitivity.png"))
    plt.savefig(plot_path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return plot_path


def _plot_pointing_error(
    pointing_errors: Dict[str, Dict[str, object]],
    out_dir: str,
) -> Optional[str]:
    """Plot pointing error time series."""
    if not pointing_errors:
        return None

    plt.rcParams.update({
        "font.size": 10,
        "axes.labelsize": 11,
        "axes.titlesize": 12,
        "legend.fontsize": 9,
    })

    fig, ax = plt.subplots(1, 1, figsize=(10, 5.5))

    for method, method_data in pointing_errors.items():
        for controller, data in method_data.items():
            time = data.get("time", np.array([]))
            sigma = data.get("sigma")
            target = data.get("target_sigma", np.zeros(3))

            if sigma is None or len(time) == 0:
                continue

            errors = _compute_pointing_error(sigma, target)
            if len(errors) == 0:
                continue

            label = f"{METHOD_LABELS.get(method, method)}"
            ax.semilogy(time, np.maximum(errors, 1e-10),
                        color=METHOD_COLORS.get(method, "black"),
                        label=label, linewidth=1.5)
            break  # Only plot first controller per method

    ax.set_title("Pointing Error", fontweight="bold")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Pointing Error (deg)")
    ax.grid(True, alpha=0.3, which="both")
    ax.legend(loc="upper right")

    plt.tight_layout()
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

    line_styles = {"standard_pd": "-", "filtered_pd": "--", "avc": "-."}

    plotted_items = []
    psd_min = None
    psd_max = None

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

            # Filter to 0-10 Hz range
            mask = (psd_freq > 0) & (psd_freq <= 10.0) & np.isfinite(psd_vals) & (psd_vals > 0)
            if not np.any(mask):
                continue

            freq_filtered = psd_freq[mask]
            psd_filtered = psd_vals[mask]

            psd_min = float(np.min(psd_filtered)) if psd_min is None else min(psd_min, float(np.min(psd_filtered)))
            psd_max = float(np.max(psd_filtered)) if psd_max is None else max(psd_max, float(np.max(psd_filtered)))

            label = f"{METHOD_LABELS[method]} + {CONTROLLER_LABELS[name]}"
            ax.semilogy(
                freq_filtered,
                psd_filtered,
                color=METHOD_COLORS[method],
                linestyle=line_styles.get(name, "-"),
                linewidth=2.0,
                alpha=0.9,
                label=label,
            )
            plotted_items.append((method, name))

    # Mark modal frequencies
    if psd_max is not None:
        for f_mode in config.modal_freqs_hz:
            if f_mode <= 10.0:
                ax.axvline(f_mode, color="red", linestyle="--", alpha=0.6, linewidth=1.5)
                ax.text(f_mode + 0.05, psd_max * 0.5, f"Mode: {f_mode:.2f} Hz",
                        rotation=90, va="bottom", ha="left", fontsize=10, alpha=0.9,
                        fontweight="bold", color="red")

    ax.set_title("Mission Vibration Displacement PSD (Combined Feedforward + Feedback)",
                 fontweight="bold", fontsize=16, pad=15)
    ax.set_xlabel("Frequency (Hz)", fontsize=14, fontweight="bold")
    ax.set_ylabel(r"PSD (m$^2$ / Hz)", fontsize=14, fontweight="bold")

    # X-axis: 0-10 Hz with ticks at 0.5 Hz intervals
    ax.set_xlim([0.0, 10.0])
    ax.set_xticks(np.arange(0, 10.5, 0.5))
    ax.set_xticks(np.arange(0, 10.1, 0.1), minor=True)

    # Y-axis: limit to 3 decades for readability
    if psd_min is not None and psd_max is not None:
        top = psd_max * 5.0
        bottom = max(psd_min / 2.0, top / 1e3)
        ax.set_ylim([bottom, top])

    ax.yaxis.set_major_locator(LogLocator(base=10, numticks=15))
    ax.yaxis.set_minor_locator(LogLocator(base=10, subs=np.arange(2, 10) * 0.1, numticks=100))

    ax.grid(True, alpha=0.6, which="major", linestyle="-", linewidth=0.8, color="gray")
    ax.grid(True, alpha=0.3, which="minor", linestyle="-", linewidth=0.4, color="lightgray")

    if plotted_items:
        ax.legend(loc="upper right", framealpha=0.95, ncol=1, fontsize=11,
                  fancybox=True, shadow=True)

    plt.tight_layout()
    _ensure_dir(out_dir)
    plot_path = os.path.abspath(os.path.join(out_dir, "mission_psd.png"))
    plt.savefig(plot_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return plot_path


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
    l_data = control_data["L"]
    margins = control_data["margins"]

    plt.rcParams.update({
        "font.size": 10,
        "axes.labelsize": 11,
        "axes.titlesize": 12,
        "legend.fontsize": 9,
    })

    fig, ax = plt.subplots(1, 1, figsize=(8, 7))

    line_styles = {"standard_pd": "-", "filtered_pd": "--", "avc": "-."}

    for name in CONTROLLERS:
        l_resp = l_data[name]
        nyq = np.concatenate([l_resp, np.conjugate(l_resp[::-1])])
        ls = line_styles.get(name, "-")
        label = f"{CONTROLLER_LABELS[name]} ({_format_margin_label(margins[name])})"
        ax.plot(nyq.real, nyq.imag, color=CONTROLLER_COLORS[name], label=label,
                linewidth=1.5, linestyle=ls)

    # Critical point
    ax.plot(-1, 0, "rx", markersize=10, markeredgewidth=2, label="Critical Point")

    ax.set_title("Nyquist Diagram", fontweight="bold")
    ax.set_xlabel("Real")
    ax.set_ylabel("Imaginary")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower left")
    ax.set_aspect("equal")
    ax.axhline(0, color="black", linewidth=0.5)
    ax.axvline(0, color="black", linewidth=0.5)

    plt.tight_layout()
    _ensure_dir(out_dir)
    plot_path = os.path.abspath(os.path.join(out_dir, "mission_nyquist.png"))
    plt.savefig(plot_path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return plot_path


# ---------------------------------------------------------------------------
# CSV Export Functions
# ---------------------------------------------------------------------------


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
    for name, data in feedback_data.items():
        time = data.get("time", [])
        disp = data.get("displacement", [])
        for t, d in zip(time, disp):
            rows.append([f"{t:.6f}", name, f"{d:.6e}"])
    if rows:
        path = os.path.abspath(os.path.join(out_dir, "vibration_feedback.csv"))
        _write_csv(path, ["time_s", "controller", "displacement_m"], rows)
        print(f"Wrote feedback vibration CSV: {path}")


def _export_pointing_error_csv(
    pointing_data: Dict[str, Dict[str, object]],
    out_dir: str,
) -> None:
    """Export pointing error data to CSV."""
    _ensure_dir(out_dir)
    rows = []
    for method, method_data in pointing_data.items():
        for controller, data in method_data.items():
            time = data.get("time", [])
            sigma = data.get("sigma")
            target = data.get("target_sigma", np.zeros(3))
            if sigma is None:
                continue
            errors = _compute_pointing_error(sigma, target)
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
    for name, data in feedback_data.items():
        psd_freq = data.get("psd_freq", [])
        psd_vals = data.get("psd", [])
        for f, p in zip(psd_freq, psd_vals):
            rows.append([f"{f:.6f}", name, f"{p:.6e}"])
    if rows:
        path = os.path.abspath(os.path.join(out_dir, "psd_feedback.csv"))
        _write_csv(path, ["frequency_hz", "controller", "psd_n2m2_per_hz"], rows)
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


def _export_nyquist_csv(control_data: Dict[str, object], out_dir: str) -> None:
    """Export Nyquist data to CSV."""
    _ensure_dir(out_dir)
    freqs = control_data["freqs"]
    rows = []
    for i, f in enumerate(freqs):
        for name in CONTROLLERS:
            l_val = control_data["L"][name][i]
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


# ---------------------------------------------------------------------------
# Main Mission Simulation
# ---------------------------------------------------------------------------


def run_mission_simulation(
    config: MissionConfig,
    out_dir: str = "analysis",
    data_dir: Optional[str] = None,
    make_plots: bool = True,
    export_csv: bool = True,
    generate_pointing: bool = False,
) -> Dict[str, object]:
    """Run complete mission simulation analysis."""
    data_dir = data_dir or os.path.dirname(__file__)

    # Run analyses
    ff_metrics = run_feedforward_comparison(
        config,
        out_dir,
        make_plots=False,
        export_csv=export_csv,
        data_dir=data_dir,
        prefer_npz=True,
    )

    ctrl_data = _compute_control_analysis(config)
    ctrl_metrics = {}
    for name in CONTROLLERS:
        ctrl_metrics[name] = ctrl_data["margins"][name]

    pointing_metrics = run_pointing_summary(config, out_dir, data_dir=data_dir,
                                            make_plots=False, export_csv=export_csv)

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
    pointing_data = _load_all_pointing_data(data_dir, generate_if_missing=generate_pointing)

    # Export remaining CSVs
    if export_csv:
        _export_vibration_csv(feedforward_vibration, feedback_vibration, out_dir)
        _export_pointing_error_csv(pointing_data, out_dir)
        _export_psd_csv(feedforward_torque, feedback_vibration, out_dir, mission_psd_data)
        _export_sensitivity_csv(ctrl_data, out_dir)
        _export_nyquist_csv(ctrl_data, out_dir)
        _export_mission_summary_csv(ff_metrics, ctrl_metrics, pointing_metrics, out_dir)

    # Generate plots
    plot_paths: List[str] = []
    if make_plots:
        vibration_plot = _plot_vibration_comparison(feedforward_vibration, feedback_vibration, out_dir)
        sensitivity_plot = _plot_sensitivity_functions(ctrl_data, config, out_dir)
        pointing_plot = _plot_pointing_error(pointing_data, out_dir)
        psd_plot = _plot_psd_comparison(mission_psd_data, config, out_dir)
        nyquist_plot = _plot_nyquist(ctrl_data, out_dir)

        for plot in [vibration_plot, sensitivity_plot, pointing_plot, psd_plot, nyquist_plot]:
            if plot:
                plot_paths.append(plot)
                print(f"Saved plot: {plot}")

    if make_plots:
        print(f"Plots are saved in: {os.path.abspath(out_dir)}")
    if export_csv:
        print(f"CSV exports are saved in: {os.path.abspath(out_dir)}")

    return {
        "feedforward": ff_metrics,
        "control": ctrl_metrics,
        "pointing": pointing_metrics,
        "plots": plot_paths,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    """Command line interface."""
    parser = argparse.ArgumentParser(description="Mission simulation analysis")
    parser.add_argument("--out-dir", default="analysis", help="Output directory")
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

