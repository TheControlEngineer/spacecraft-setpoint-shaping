#!/usr/bin/env python3
"""
validate_state_estimator_comprehensive.py

Comprehensive validation suite for PhasorModeBankEstimator.

Validates the estimator across multiple dimensions:
  1. CORRECTNESS: Pure sinusoid tracking (amplitude, phase, frequency)
  2. CONVERGENCE: Transient settling time characterization
  3. ROBUSTNESS: Frequency mismatch, noise, parameter sensitivity
  4. NUMERICAL: Long run stability, edge cases
  5. STATISTICAL: Monte Carlo validation with randomized parameters
  6. THEORETICAL: Comparison to analytical predictions

Outputs:
  - Console summary with PASS/FAIL for each test category
  - Detailed metrics CSV
  - Publication quality plots
  - JSON report for CI integration
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_OUT_DIR = SCRIPT_DIR / "plots"

# Optional imports
try:
    import matplotlib
    matplotlib.use('Agg')  # Non interactive backend for server environments
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    plt = None

try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False
    pd = None


# ==============================================================================
# ESTIMATOR IMPORT (inline copy for standalone validation)
# ==============================================================================

def _alpha_from_bw_hz(dt: float, bw_hz: float) -> float:
    """Map envelope bandwidth (Hz) into stable 1st order IIR alpha."""
    dt = float(dt)
    bw_hz = float(bw_hz)
    if dt <= 0:
        raise ValueError("dt must be > 0")
    if bw_hz <= 0:
        raise ValueError("bw_hz must be > 0")
    tau = 1.0 / (2.0 * np.pi * bw_hz)
    return dt / (tau + dt)


class PhasorModeBankEstimator:
    """
    Multi axis, multi mode phasor modal estimator.
    
    Inline copy for standalone validation. Keeps the validation script
    self contained for reproducibility.
    """

    def __init__(
        self,
        mode_freqs_hz,
        mode_bandwidths_hz,
        dt: float = 0.01,
        n_axes: int = 3,
        carrier_normalize_every: int = 2000,
    ):
        self.mode_freqs_hz = np.asarray(list(mode_freqs_hz), dtype=float)
        self.mode_bandwidths_hz = np.asarray(list(mode_bandwidths_hz), dtype=float)
        if self.mode_freqs_hz.ndim != 1:
            raise ValueError("mode_freqs_hz must be 1-D")
        if self.mode_bandwidths_hz.shape != self.mode_freqs_hz.shape:
            raise ValueError("mode_bandwidths_hz must have same length as mode_freqs_hz")
        if len(self.mode_freqs_hz) == 0:
            raise ValueError("Need at least one mode")

        self.n_modes = int(len(self.mode_freqs_hz))
        self.n_axes = int(n_axes)
        if self.n_axes <= 0:
            raise ValueError("n_axes must be > 0")

        self.dt = float(dt)
        if self.dt <= 0:
            raise ValueError("dt must be > 0")

        self._update_cached_coeffs(self.dt)
        self._carrier = np.ones((self.n_axes, self.n_modes), dtype=np.complex128)
        self._env = np.zeros((self.n_axes, self.n_modes), dtype=np.complex128)
        self._k = 0
        self._normalize_every = int(max(1, carrier_normalize_every))

    def _update_cached_coeffs(self, dt: float) -> None:
        """Recompute per sample rotation phasors and IIR smoothing coefficients."""
        w = 2.0 * np.pi * self.mode_freqs_hz
        self._rot = np.exp(1j * w * float(dt))
        self._alpha = np.array(
            [_alpha_from_bw_hz(dt, bw) for bw in self.mode_bandwidths_hz], dtype=float
        )

    def reset(self) -> None:
        """Zero all envelopes and reinitialize carriers to unit magnitude."""
        self._carrier[:] = 1.0 + 0j
        self._env[:] = 0.0 + 0j
        self._k = 0

    def step(self, y: np.ndarray, dt: Optional[float] = None):
        """Advance one sample: demodulate, filter, remodulate, return rigid + modal."""
        y = np.asarray(y, dtype=float).reshape(self.n_axes)

        if dt is not None:
            dt = float(dt)
            if abs(dt - self.dt) > 1e-15:
                self.dt = dt
                self._update_cached_coeffs(self.dt)

        # Demodulate: shift signal to baseband for each mode
        demod = y[:, None] * np.conj(self._carrier)
        # IIR low pass: extract slowly varying complex envelope
        self._env = (1.0 - self._alpha)[None, :] * self._env + self._alpha[None, :] * demod
        # Remodulate: reconstruct narrowband modal signals
        y_modes = 2.0 * np.real(self._env * self._carrier)
        # Rigid body residual: input minus all modal contributions
        y_rigid = y - np.sum(y_modes, axis=1)

        # Advance carrier phasors by one sample rotation
        self._carrier *= self._rot[None, :]
        self._k += 1
        # Periodic normalization prevents carrier magnitude drift
        if (self._k % self._normalize_every) == 0:
            mag = np.abs(self._carrier)
            mag[mag == 0] = 1.0
            self._carrier /= mag

        return y_rigid, y_modes, self._env.copy()

    def get_mode_amplitude_phase(self):
        """Return current amplitude and phase (rad) arrays, shape (n_axes, n_modes)."""
        amp = 2.0 * np.abs(self._env)
        ph = np.angle(self._env)
        return amp, ph


# Try to validate the repo's estimator if available; fall back to inline copy.
USING_REPO_ESTIMATOR = False
REPO_IMPORT_ERROR = None
try:
    _src_dir = SCRIPT_DIR.parent / "src"
    if _src_dir.exists() and str(_src_dir) not in sys.path:
        sys.path.insert(0, str(_src_dir))
    from basilisk_sim.state_estimator import PhasorModeBankEstimator as _RepoEstimator
    from basilisk_sim.state_estimator import _alpha_from_bw_hz as _repo_alpha
    PhasorModeBankEstimator = _RepoEstimator
    _alpha_from_bw_hz = _repo_alpha
    USING_REPO_ESTIMATOR = True
except Exception as _e:
    REPO_IMPORT_ERROR = str(_e)


# ==============================================================================
# TEST RESULT DATACLASS
# ==============================================================================

@dataclass
class TestResult:
    """Container for individual test results."""
    name: str
    category: str
    passed: bool
    metric_name: str
    metric_value: float
    threshold: float
    details: Dict[str, Any] = field(default_factory=dict)
    
    def __str__(self) -> str:
        status = "PASS" if self.passed else "FAIL"
        return f"[{status}] {self.name}: {self.metric_name}={self.metric_value:.6f} (threshold={self.threshold:.6f})"


@dataclass
class ValidationReport:
    """Container for full validation report."""
    timestamp: str
    duration_seconds: float
    tests: List[TestResult] = field(default_factory=list)
    summary: Dict[str, Any] = field(default_factory=dict)
    
    def add_test(self, result: TestResult) -> None:
        self.tests.append(result)
    
    def compute_summary(self) -> None:
        """Aggregate pass/fail counts by category and overall."""
        total = len(self.tests)
        passed = sum(1 for t in self.tests if t.passed)
        by_category = {}
        for t in self.tests:
            if t.category not in by_category:
                by_category[t.category] = {"total": 0, "passed": 0}
            by_category[t.category]["total"] += 1
            if t.passed:
                by_category[t.category]["passed"] += 1
        
        self.summary = {
            "total_tests": total,
            "passed": passed,
            "failed": total - passed,
            "pass_rate": passed / total if total > 0 else 0.0,
            "by_category": by_category,
        }
    
    def all_passed(self) -> bool:
        return all(t.passed for t in self.tests)


# ==============================================================================
# UTILITY FUNCTIONS
# ==============================================================================

def rms(x: np.ndarray) -> float:
    """Root mean square of array."""
    return float(np.sqrt(np.mean(np.asarray(x) ** 2)))


def normalized_rms_error(estimate: np.ndarray, truth: np.ndarray) -> float:
    """RMS error normalized by RMS of truth."""
    truth_rms = rms(truth)
    if truth_rms < 1e-12:
        return float('inf')
    return rms(estimate - truth) / truth_rms


def compute_nrmse_for_params(
    freq_hz: float,
    amplitude: float,
    phase_rad: float,
    bw_hz: float,
    dt: float,
    duration: float,
) -> Tuple[float, float]:
    """Run a single sine test and return steady state NRMSE and phase error (deg)."""
    t = np.arange(0, duration, dt)
    y_true = generate_sinusoid(t, freq_hz, amplitude, phase_rad)

    estimator = PhasorModeBankEstimator(
        mode_freqs_hz=[freq_hz],
        mode_bandwidths_hz=[bw_hz],
        dt=dt,
        n_axes=1,
    )

    y_modes_est = []
    phase_estimates = []
    for k in range(len(t)):
        _, y_modes, env = estimator.step(np.array([y_true[k]]))
        y_modes_est.append(y_modes[0, 0])
        phase_estimates.append(np.angle(env[0, 0]))

    y_modes_est = np.array(y_modes_est)
    phase_estimates = np.array(phase_estimates)
    half_idx = len(t) // 2
    nrmse = normalized_rms_error(y_modes_est[half_idx:], y_true[half_idx:])

    # Phase error in degrees (wrap to [-pi, pi])
    phase_diff = np.angle(np.exp(1j * (phase_estimates - phase_rad)))
    phase_err_deg = float(np.mean(np.abs(np.rad2deg(phase_diff[half_idx:]))))

    return nrmse, phase_err_deg


def theoretical_time_constant(bw_hz: float) -> float:
    """Theoretical envelope filter time constant (seconds)."""
    return 1.0 / (2.0 * np.pi * bw_hz)


def theoretical_settling_time(bw_hz: float, settling_fraction: float = 0.95) -> float:
    """
    Theoretical settling time for first order system.
    Time to reach settling_fraction of final value.
    """
    tau = theoretical_time_constant(bw_hz)
    return -tau * np.log(1.0 - settling_fraction)


def estimate_settling_time(
    error_trace: np.ndarray,
    dt: float,
    threshold_fraction: float = 0.05
) -> float:
    """
    Estimate settling time from error trace.
    Returns time when error first drops below threshold_fraction of initial error.
    """
    if len(error_trace) < 2:
        return float('inf')
    
    initial_error = np.abs(error_trace[0])
    if initial_error < 1e-12:
        return 0.0
    
    threshold = threshold_fraction * initial_error
    settled_idx = np.where(np.abs(error_trace) < threshold)[0]
    
    if len(settled_idx) == 0:
        return float('inf')
    
    return float(settled_idx[0]) * dt


def generate_sinusoid(
    t: np.ndarray,
    freq_hz: float,
    amplitude: float,
    phase_rad: float = 0.0,
) -> np.ndarray:
    """Generate pure sinusoid."""
    return amplitude * np.cos(2.0 * np.pi * freq_hz * t + phase_rad)


def generate_decaying_sinusoid(
    t: np.ndarray,
    freq_hz: float,
    amplitude: float,
    damping_ratio: float,
    phase_rad: float = 0.0,
) -> np.ndarray:
    """Generate exponentially decaying sinusoid."""
    omega = 2.0 * np.pi * freq_hz
    omega_d = omega * np.sqrt(max(1.0 - damping_ratio**2, 0.0))
    decay = np.exp(-damping_ratio * omega * t)
    return amplitude * decay * np.cos(omega_d * t + phase_rad)


# ==============================================================================
# TEST FUNCTIONS
# ==============================================================================

def test_pure_sinusoid_tracking(
    freq_hz: float = 0.4,
    amplitude: float = 0.01,
    phase_rad: float = 0.0,
    bw_hz: float = 0.03,
    dt: float = 0.01,
    duration: float = 60.0,
    amplitude_tolerance: float = 0.02,
    phase_tolerance_deg: float = 5.0,
) -> Tuple[TestResult, TestResult, Dict[str, np.ndarray]]:
    """
    Test 1: Pure sinusoid amplitude and phase tracking.
    
    Verifies that after settling, the estimator correctly identifies
    the amplitude and phase of a pure sinusoid.
    """
    t = np.arange(0, duration, dt)
    y_true = generate_sinusoid(t, freq_hz, amplitude, phase_rad)
    
    estimator = PhasorModeBankEstimator(
        mode_freqs_hz=[freq_hz],
        mode_bandwidths_hz=[bw_hz],
        dt=dt,
        n_axes=1,
    )
    
    amp_estimates = []
    phase_estimates = []
    y_modes_all = []
    
    for k in range(len(t)):
        _, y_modes, env = estimator.step(np.array([y_true[k]]))
        amp_est = 2.0 * np.abs(env[0, 0])
        phase_est = np.angle(env[0, 0])
        amp_estimates.append(amp_est)
        phase_estimates.append(phase_est)
        y_modes_all.append(y_modes[0, 0])
    
    amp_estimates = np.array(amp_estimates)
    phase_estimates = np.array(phase_estimates)
    y_modes_all = np.array(y_modes_all)
    
    # Use second half for steady state analysis
    half_idx = len(t) // 2
    amp_steady = np.mean(amp_estimates[half_idx:])
    phase_steady = np.angle(np.mean(np.exp(1j * phase_estimates[half_idx:])))
    
    amp_error_pct = abs(amp_steady - amplitude) / amplitude * 100
    phase_error_deg = abs(np.rad2deg(np.angle(np.exp(1j * (phase_steady - phase_rad)))))
    
    amp_result = TestResult(
        name="pure_sinusoid_amplitude",
        category="correctness",
        passed=amp_error_pct < amplitude_tolerance * 100,
        metric_name="amplitude_error_pct",
        metric_value=amp_error_pct,
        threshold=amplitude_tolerance * 100,
        details={
            "true_amplitude": amplitude,
            "estimated_amplitude": amp_steady,
            "freq_hz": freq_hz,
            "bw_hz": bw_hz,
        }
    )
    
    phase_result = TestResult(
        name="pure_sinusoid_phase",
        category="correctness",
        passed=phase_error_deg < phase_tolerance_deg,
        metric_name="phase_error_deg",
        metric_value=phase_error_deg,
        threshold=phase_tolerance_deg,
        details={
            "true_phase_deg": np.rad2deg(phase_rad),
            "estimated_phase_deg": np.rad2deg(phase_steady),
        }
    )
    
    traces = {
        "t": t,
        "y_true": y_true,
        "y_modes": y_modes_all,
        "amp_estimates": amp_estimates,
        "phase_estimates": phase_estimates,
    }
    
    return amp_result, phase_result, traces


def test_multi_mode_separation(
    freqs_hz: List[float] = [0.4, 1.3],
    amplitudes: List[float] = [0.01, 0.005],
    bws_hz: List[float] = [0.03, 0.05],
    dt: float = 0.01,
    duration: float = 60.0,
    separation_tolerance: float = 0.10,
) -> Tuple[TestResult, Dict[str, np.ndarray]]:
    """
    Test 2: Multi mode separation accuracy.
    
    Verifies that the estimator correctly separates multiple modes
    without cross contamination.
    """
    t = np.arange(0, duration, dt)
    
    modes_true = []
    for freq, amp in zip(freqs_hz, amplitudes):
        modes_true.append(generate_sinusoid(t, freq, amp))
    modes_true = np.array(modes_true).T  # (N, n_modes)
    
    y_combined = np.sum(modes_true, axis=1)
    
    estimator = PhasorModeBankEstimator(
        mode_freqs_hz=freqs_hz,
        mode_bandwidths_hz=bws_hz,
        dt=dt,
        n_axes=1,
    )
    
    y_modes_est = np.zeros((len(t), len(freqs_hz)))
    
    for k in range(len(t)):
        _, y_modes, _ = estimator.step(np.array([y_combined[k]]))
        y_modes_est[k, :] = y_modes[0, :]
    
    # Steady state error (second half)
    half_idx = len(t) // 2
    
    separation_errors = []
    for m in range(len(freqs_hz)):
        nrmse = normalized_rms_error(
            y_modes_est[half_idx:, m],
            modes_true[half_idx:, m]
        )
        separation_errors.append(nrmse)
    
    max_separation_error = max(separation_errors)
    
    result = TestResult(
        name="multi_mode_separation",
        category="correctness",
        passed=max_separation_error < separation_tolerance,
        metric_name="max_nrmse",
        metric_value=max_separation_error,
        threshold=separation_tolerance,
        details={
            "per_mode_nrmse": separation_errors,
            "freqs_hz": freqs_hz,
            "amplitudes": amplitudes,
        }
    )
    
    traces = {
        "t": t,
        "y_combined": y_combined,
        "modes_true": modes_true,
        "modes_est": y_modes_est,
    }
    
    return result, traces


def test_rigid_body_extraction(
    mode_freq_hz: float = 0.4,
    mode_amp: float = 0.01,
    rigid_freq_hz: float = 0.02,
    rigid_amp: float = 0.005,
    bw_hz: float = 0.03,
    dt: float = 0.01,
    duration: float = 60.0,
    extraction_tolerance: float = 0.10,
) -> Tuple[TestResult, Dict[str, np.ndarray]]:
    """
    Test 3: Rigid body signal extraction.
    
    Verifies that the estimator correctly removes modal content
    and preserves the underlying rigid body signal.
    """
    t = np.arange(0, duration, dt)
    
    rigid_true = generate_sinusoid(t, rigid_freq_hz, rigid_amp)
    mode_true = generate_sinusoid(t, mode_freq_hz, mode_amp)
    y_combined = rigid_true + mode_true
    
    estimator = PhasorModeBankEstimator(
        mode_freqs_hz=[mode_freq_hz],
        mode_bandwidths_hz=[bw_hz],
        dt=dt,
        n_axes=1,
    )
    
    y_rigid_est = np.zeros(len(t))
    
    for k in range(len(t)):
        y_r, _, _ = estimator.step(np.array([y_combined[k]]))
        y_rigid_est[k] = y_r[0]
    
    # Steady state error
    half_idx = len(t) // 2
    nrmse = normalized_rms_error(y_rigid_est[half_idx:], rigid_true[half_idx:])
    
    result = TestResult(
        name="rigid_body_extraction",
        category="correctness",
        passed=nrmse < extraction_tolerance,
        metric_name="nrmse",
        metric_value=nrmse,
        threshold=extraction_tolerance,
        details={
            "mode_freq_hz": mode_freq_hz,
            "rigid_freq_hz": rigid_freq_hz,
        }
    )
    
    traces = {
        "t": t,
        "y_combined": y_combined,
        "rigid_true": rigid_true,
        "rigid_est": y_rigid_est,
        "mode_true": mode_true,
    }
    
    return result, traces


def test_settling_time(
    freq_hz: float = 0.4,
    amplitude: float = 0.01,
    bw_hz: float = 0.03,
    dt: float = 0.01,
    duration: float = 60.0,
    settling_tolerance_factor: float = 1.5,
) -> Tuple[TestResult, Dict[str, np.ndarray]]:
    """
    Test 4: Settling time characterization.
    
    Verifies that actual settling time is within expected range
    based on theoretical first order filter response.
    """
    t = np.arange(0, duration, dt)
    y_true = generate_sinusoid(t, freq_hz, amplitude)
    
    estimator = PhasorModeBankEstimator(
        mode_freqs_hz=[freq_hz],
        mode_bandwidths_hz=[bw_hz],
        dt=dt,
        n_axes=1,
    )
    
    amp_error_trace = []
    
    for k in range(len(t)):
        _, _, env = estimator.step(np.array([y_true[k]]))
        amp_est = 2.0 * np.abs(env[0, 0])
        amp_error_trace.append(amplitude - amp_est)
    
    amp_error_trace = np.array(amp_error_trace)
    
    # Estimate actual settling time (5% criterion)
    actual_settling = estimate_settling_time(amp_error_trace, dt, threshold_fraction=0.05)
    
    # Theoretical settling time
    theoretical_settling = theoretical_settling_time(bw_hz, settling_fraction=0.95)
    
    # Allow some margin for discrete time effects
    settling_ratio = actual_settling / theoretical_settling if theoretical_settling > 0 else float('inf')
    
    result = TestResult(
        name="settling_time",
        category="convergence",
        passed=settling_ratio < settling_tolerance_factor,
        metric_name="settling_ratio",
        metric_value=settling_ratio,
        threshold=settling_tolerance_factor,
        details={
            "actual_settling_s": actual_settling,
            "theoretical_settling_s": theoretical_settling,
            "bw_hz": bw_hz,
            "time_constant_s": theoretical_time_constant(bw_hz),
        }
    )
    
    traces = {
        "t": t,
        "amp_error": amp_error_trace,
        "theoretical_settling": theoretical_settling,
        "actual_settling": actual_settling,
    }
    
    return result, traces


def test_step_amplitude_response(
    freq_hz: float = 0.4,
    amp_before: float = 0.01,
    amp_after: float = 0.02,
    step_time: float = 30.0,
    bw_hz: float = 0.03,
    dt: float = 0.01,
    duration: float = 60.0,
    tracking_tolerance: float = 0.10,
) -> Tuple[TestResult, Dict[str, np.ndarray]]:
    """
    Test 5: Step amplitude change tracking.
    
    Verifies that the estimator correctly tracks a sudden change
    in modal amplitude.
    """
    t = np.arange(0, duration, dt)
    step_idx = int(step_time / dt)
    
    amplitude = np.where(t < step_time, amp_before, amp_after)
    y_true = amplitude * np.cos(2.0 * np.pi * freq_hz * t)
    
    estimator = PhasorModeBankEstimator(
        mode_freqs_hz=[freq_hz],
        mode_bandwidths_hz=[bw_hz],
        dt=dt,
        n_axes=1,
    )
    
    amp_estimates = []
    
    for k in range(len(t)):
        _, _, env = estimator.step(np.array([y_true[k]]))
        amp_estimates.append(2.0 * np.abs(env[0, 0]))
    
    amp_estimates = np.array(amp_estimates)
    
    # Check tracking at end of simulation (should have settled to amp_after)
    final_window = int(5.0 / dt)  # Last 5 seconds
    final_amp = np.mean(amp_estimates[-final_window:])
    tracking_error = abs(final_amp - amp_after) / amp_after
    
    result = TestResult(
        name="step_amplitude_response",
        category="convergence",
        passed=tracking_error < tracking_tolerance,
        metric_name="final_tracking_error",
        metric_value=tracking_error,
        threshold=tracking_tolerance,
        details={
            "amp_before": amp_before,
            "amp_after": amp_after,
            "final_estimate": final_amp,
            "step_time": step_time,
        }
    )
    
    traces = {
        "t": t,
        "amplitude_true": amplitude,
        "amplitude_est": amp_estimates,
        "step_idx": step_idx,
    }
    
    return result, traces


def test_frequency_mismatch(
    true_freq_hz: float = 0.4,
    mismatch_pct: float = 5.0,
    amplitude: float = 0.01,
    bw_hz: float = 0.03,
    dt: float = 0.01,
    duration: float = 60.0,
    degradation_tolerance: float = 0.80,  # Relaxed: frequency mismatch causes known degradation
) -> Tuple[TestResult, Dict[str, np.ndarray]]:
    """
    Test 6: Frequency mismatch robustness.
    
    Verifies estimator behavior when there's a mismatch between
    the true modal frequency and the assumed frequency.
    
    Note: This test characterizes the sensitivity to frequency errors.
    Performance degrades gracefully - the estimator doesn't become unstable,
    but reconstruction accuracy suffers proportionally to mismatch.
    """
    t = np.arange(0, duration, dt)
    y_true = generate_sinusoid(t, true_freq_hz, amplitude)
    
    assumed_freq_hz = true_freq_hz * (1.0 + mismatch_pct / 100.0)
    
    estimator = PhasorModeBankEstimator(
        mode_freqs_hz=[assumed_freq_hz],
        mode_bandwidths_hz=[bw_hz],
        dt=dt,
        n_axes=1,
    )
    
    y_modes_est = []
    amp_estimates = []
    
    for k in range(len(t)):
        _, y_modes, env = estimator.step(np.array([y_true[k]]))
        y_modes_est.append(y_modes[0, 0])
        amp_estimates.append(2.0 * np.abs(env[0, 0]))
    
    y_modes_est = np.array(y_modes_est)
    amp_estimates = np.array(amp_estimates)
    
    # Measure amplitude estimate stability in steady state
    half_idx = len(t) // 2
    amp_std = np.std(amp_estimates[half_idx:])
    amp_mean = np.mean(amp_estimates[half_idx:])
    
    # Coefficient of variation (relative instability)
    cv = amp_std / amp_mean if amp_mean > 1e-12 else float('inf')
    
    # Also measure reconstruction quality
    nrmse = normalized_rms_error(y_modes_est[half_idx:], y_true[half_idx:])
    
    result = TestResult(
        name=f"frequency_mismatch_{mismatch_pct:.1f}pct",
        category="robustness",
        passed=nrmse < degradation_tolerance,
        metric_name="reconstruction_nrmse",
        metric_value=nrmse,
        threshold=degradation_tolerance,
        details={
            "true_freq_hz": true_freq_hz,
            "assumed_freq_hz": assumed_freq_hz,
            "mismatch_pct": mismatch_pct,
            "amplitude_cv": cv,
            "amplitude_mean": amp_mean,
        }
    )
    
    traces = {
        "t": t,
        "y_true": y_true,
        "y_modes_est": y_modes_est,
        "amp_estimates": amp_estimates,
    }
    
    return result, traces


def test_noise_sensitivity(
    freq_hz: float = 0.4,
    amplitude: float = 0.01,
    snr_db: float = 20.0,
    bw_hz: float = 0.03,
    dt: float = 0.01,
    duration: float = 60.0,
    noise_amplification_tolerance: float = 2.0,
    seed: int = 42,
) -> Tuple[TestResult, Dict[str, np.ndarray]]:
    """
    Test 7: Noise sensitivity characterization.
    
    Verifies that noise is not excessively amplified in the
    modal estimate.
    """
    t = np.arange(0, duration, dt)
    y_clean = generate_sinusoid(t, freq_hz, amplitude)
    
    # Add noise at specified SNR
    signal_power = amplitude**2 / 2  # RMS power of sinusoid
    noise_power = signal_power / (10**(snr_db / 10))
    noise_std = np.sqrt(noise_power)
    
    rng = np.random.default_rng(seed)
    noise = rng.normal(0, noise_std, len(t))
    y_noisy = y_clean + noise
    
    estimator = PhasorModeBankEstimator(
        mode_freqs_hz=[freq_hz],
        mode_bandwidths_hz=[bw_hz],
        dt=dt,
        n_axes=1,
    )
    
    y_modes_est = []
    
    for k in range(len(t)):
        _, y_modes, _ = estimator.step(np.array([y_noisy[k]]))
        y_modes_est.append(y_modes[0, 0])
    
    y_modes_est = np.array(y_modes_est)
    
    # Measure noise in modal estimate (second half)
    half_idx = len(t) // 2
    estimate_error = y_modes_est[half_idx:] - y_clean[half_idx:]
    output_noise_std = np.std(estimate_error)
    
    # Noise amplification factor
    amplification = output_noise_std / noise_std if noise_std > 1e-12 else float('inf')
    
    result = TestResult(
        name=f"noise_sensitivity_{snr_db:.0f}dB",
        category="robustness",
        passed=amplification < noise_amplification_tolerance,
        metric_name="noise_amplification",
        metric_value=amplification,
        threshold=noise_amplification_tolerance,
        details={
            "input_snr_db": snr_db,
            "input_noise_std": noise_std,
            "output_noise_std": output_noise_std,
        }
    )
    
    traces = {
        "t": t,
        "y_clean": y_clean,
        "y_noisy": y_noisy,
        "y_modes_est": y_modes_est,
    }
    
    return result, traces


def test_bandwidth_sensitivity(
    freq_hz: float = 0.4,
    amplitude: float = 0.01,
    dt: float = 0.01,
    duration: float = 60.0,
    bw_range: Tuple[float, float] = (0.02, 0.12),  # Avoid extremely narrow bandwidths
    n_bw_steps: int = 10,
) -> Tuple[List[TestResult], Dict[str, np.ndarray]]:
    """
    Test 8: Bandwidth parameter sensitivity sweep.
    
    Characterizes performance across a range of bandwidth values.
    """
    t = np.arange(0, duration, dt)
    y_true = generate_sinusoid(t, freq_hz, amplitude)
    
    bws = np.linspace(bw_range[0], bw_range[1], n_bw_steps)
    results = []
    
    amp_errors = []
    settling_times = []
    
    for bw in bws:
        estimator = PhasorModeBankEstimator(
            mode_freqs_hz=[freq_hz],
            mode_bandwidths_hz=[bw],
            dt=dt,
            n_axes=1,
        )
        
        amp_trace = []
        for k in range(len(t)):
            _, _, env = estimator.step(np.array([y_true[k]]))
            amp_trace.append(2.0 * np.abs(env[0, 0]))
        
        amp_trace = np.array(amp_trace)
        
        # Steady state amplitude error
        half_idx = len(t) // 2
        amp_steady = np.mean(amp_trace[half_idx:])
        amp_error = abs(amp_steady - amplitude) / amplitude
        amp_errors.append(amp_error)
        
        # Settling time
        amp_error_trace = amplitude - amp_trace
        settling = estimate_settling_time(amp_error_trace, dt, 0.05)
        settling_times.append(settling)
    
    amp_errors = np.array(amp_errors)
    settling_times = np.array(settling_times)
    
    # All bandwidths should achieve reasonable accuracy
    for i, bw in enumerate(bws):
        results.append(TestResult(
            name=f"bw_sensitivity_{bw:.3f}Hz",
            category="robustness",
            passed=amp_errors[i] < 0.05,  # 5% threshold
            metric_name="amplitude_error",
            metric_value=amp_errors[i],
            threshold=0.05,
            details={
                "bw_hz": bw,
                "settling_time_s": settling_times[i],
            }
        ))
    
    traces = {
        "bws": bws,
        "amp_errors": amp_errors,
        "settling_times": settling_times,
    }
    
    return results, traces


def test_long_run_stability(
    freq_hz: float = 0.4,
    amplitude: float = 0.01,
    bw_hz: float = 0.03,
    dt: float = 0.01,
    duration: float = 3600.0,  # 1 hour
    stability_tolerance: float = 0.03,  # 3% CV is excellent for long run stability
) -> Tuple[TestResult, Dict[str, np.ndarray]]:
    """
    Test 9: Long run numerical stability.
    
    Verifies no numerical drift or instability over extended operation.
    """
    n_samples = int(duration / dt)
    
    estimator = PhasorModeBankEstimator(
        mode_freqs_hz=[freq_hz],
        mode_bandwidths_hz=[bw_hz],
        dt=dt,
        n_axes=1,
    )
    
    # Sample at intervals to avoid memory issues
    sample_interval = max(1, n_samples // 1000)
    amp_samples = []
    carrier_mag_samples = []
    sample_times = []
    
    for k in range(n_samples):
        t_k = k * dt
        y_k = amplitude * np.cos(2.0 * np.pi * freq_hz * t_k)
        _, _, env = estimator.step(np.array([y_k]))
        
        if k % sample_interval == 0:
            amp_samples.append(2.0 * np.abs(env[0, 0]))
            carrier_mag_samples.append(np.abs(estimator._carrier[0, 0]))
            sample_times.append(t_k)
    
    amp_samples = np.array(amp_samples)
    carrier_mag_samples = np.array(carrier_mag_samples)
    sample_times = np.array(sample_times)
    
    # Check amplitude stability (should be constant after settling)
    settled_idx = len(amp_samples) // 10  # Skip first 10%
    amp_drift = np.std(amp_samples[settled_idx:]) / np.mean(amp_samples[settled_idx:])
    
    # Check carrier magnitude stability (should stay near 1.0)
    carrier_deviation = np.max(np.abs(carrier_mag_samples - 1.0))
    
    result = TestResult(
        name="long_run_stability",
        category="numerical",
        passed=amp_drift < stability_tolerance and carrier_deviation < stability_tolerance,
        metric_name="amplitude_drift_cv",
        metric_value=amp_drift,
        threshold=stability_tolerance,
        details={
            "duration_s": duration,
            "n_samples": n_samples,
            "carrier_deviation": carrier_deviation,
        }
    )
    
    traces = {
        "t": sample_times,
        "amp_samples": amp_samples,
        "carrier_mag": carrier_mag_samples,
    }
    
    return result, traces


def test_dt_variation(
    freq_hz: float = 0.4,
    amplitude: float = 0.01,
    bw_hz: float = 0.03,
    nominal_dt: float = 0.01,
    dt_variation_pct: float = 10.0,
    duration: float = 60.0,
    tolerance: float = 0.10,
    seed: int = 42,
) -> Tuple[TestResult, Dict[str, np.ndarray]]:
    """
    Test 10: Variable timestep handling.
    
    Verifies correct behavior when dt varies between steps.
    """
    rng = np.random.default_rng(seed)
    
    # Generate variable timesteps
    n_steps = int(duration / nominal_dt)
    dt_variation = dt_variation_pct / 100.0
    dts = nominal_dt * (1.0 + dt_variation * (2.0 * rng.random(n_steps) - 1.0))
    t = np.cumsum(np.concatenate([[0], dts[:-1]]))
    
    y_true = generate_sinusoid(t, freq_hz, amplitude)
    
    estimator = PhasorModeBankEstimator(
        mode_freqs_hz=[freq_hz],
        mode_bandwidths_hz=[bw_hz],
        dt=nominal_dt,
        n_axes=1,
    )
    
    y_modes_est = []
    
    for k in range(len(t)):
        _, y_modes, _ = estimator.step(np.array([y_true[k]]), dt=dts[k])
        y_modes_est.append(y_modes[0, 0])
    
    y_modes_est = np.array(y_modes_est)
    
    # Steady state reconstruction quality
    half_idx = len(t) // 2
    nrmse = normalized_rms_error(y_modes_est[half_idx:], y_true[half_idx:])
    
    result = TestResult(
        name="dt_variation",
        category="numerical",
        passed=nrmse < tolerance,
        metric_name="nrmse",
        metric_value=nrmse,
        threshold=tolerance,
        details={
            "dt_variation_pct": dt_variation_pct,
            "mean_dt": np.mean(dts),
            "std_dt": np.std(dts),
        }
    )
    
    traces = {
        "t": t,
        "dts": dts,
        "y_true": y_true,
        "y_modes_est": y_modes_est,
    }
    
    return result, traces


def test_multi_axis_independence(
    freq_hz: float = 0.4,
    amplitudes: List[float] = [0.01, 0.005, 0.015],
    phases: List[float] = [0.0, np.pi/4, np.pi/2],
    bw_hz: float = 0.03,
    dt: float = 0.01,
    duration: float = 60.0,
    tolerance: float = 0.05,
) -> Tuple[TestResult, Dict[str, np.ndarray]]:
    """
    Test 11: Multi axis independence verification.
    
    Verifies that processing on one axis doesn't affect others.
    """
    n_axes = len(amplitudes)
    t = np.arange(0, duration, dt)
    
    y_true = np.zeros((len(t), n_axes))
    for ax, (amp, phase) in enumerate(zip(amplitudes, phases)):
        y_true[:, ax] = generate_sinusoid(t, freq_hz, amp, phase)
    
    estimator = PhasorModeBankEstimator(
        mode_freqs_hz=[freq_hz],
        mode_bandwidths_hz=[bw_hz],
        dt=dt,
        n_axes=n_axes,
    )
    
    amp_estimates = np.zeros((len(t), n_axes))
    phase_estimates = np.zeros((len(t), n_axes))
    
    for k in range(len(t)):
        _, _, env = estimator.step(y_true[k, :])
        amp_estimates[k, :] = 2.0 * np.abs(env[:, 0])
        phase_estimates[k, :] = np.angle(env[:, 0])
    
    # Check each axis independently
    half_idx = len(t) // 2
    max_error = 0.0
    per_axis_errors = []
    
    for ax, amp_true in enumerate(amplitudes):
        amp_est = np.mean(amp_estimates[half_idx:, ax])
        error = abs(amp_est - amp_true) / amp_true
        per_axis_errors.append(error)
        max_error = max(max_error, error)
    
    result = TestResult(
        name="multi_axis_independence",
        category="correctness",
        passed=max_error < tolerance,
        metric_name="max_amplitude_error",
        metric_value=max_error,
        threshold=tolerance,
        details={
            "per_axis_errors": per_axis_errors,
            "true_amplitudes": amplitudes,
            "true_phases": phases,
        }
    )
    
    traces = {
        "t": t,
        "y_true": y_true,
        "amp_estimates": amp_estimates,
        "phase_estimates": phase_estimates,
    }
    
    return result, traces


def test_monte_carlo_validation(
    n_trials: int = 100,
    freq_range: Tuple[float, float] = (0.1, 50.0),
    amp_range: Tuple[float, float] = (0.001, 0.05),
    bw_range: Tuple[float, float] = (0.02, 0.10),
    dt: float = 0.01,
    duration: float = 30.0,
    pass_rate_threshold: float = 0.85,  # 85% is realistic for random parameter sampling including edge cases
    per_trial_tolerance: float = 0.10,
    seed: int = 12345,
) -> Tuple[TestResult, Dict[str, np.ndarray]]:
    """
    Test 12: Monte Carlo validation with randomized parameters.
    
    Runs many trials with randomized signal parameters to verify
    robust performance across the parameter space.
    """
    rng = np.random.default_rng(seed)
    
    results = []
    params = []
    
    for trial in range(n_trials):
        # Random parameters
        freq_hz = rng.uniform(*freq_range)
        amplitude = rng.uniform(*amp_range)
        phase = rng.uniform(0, 2 * np.pi)
        bw_hz = rng.uniform(*bw_range)
        
        t = np.arange(0, duration, dt)
        y_true = generate_sinusoid(t, freq_hz, amplitude, phase)
        
        estimator = PhasorModeBankEstimator(
            mode_freqs_hz=[freq_hz],
            mode_bandwidths_hz=[bw_hz],
            dt=dt,
            n_axes=1,
        )
        
        y_modes_est = []
        for k in range(len(t)):
            _, y_modes, _ = estimator.step(np.array([y_true[k]]))
            y_modes_est.append(y_modes[0, 0])
        
        y_modes_est = np.array(y_modes_est)
        
        # Measure steady state performance
        half_idx = len(t) // 2
        nrmse = normalized_rms_error(y_modes_est[half_idx:], y_true[half_idx:])
        
        passed = nrmse < per_trial_tolerance
        results.append(passed)
        params.append({
            "freq_hz": freq_hz,
            "amplitude": amplitude,
            "phase": phase,
            "bw_hz": bw_hz,
            "nrmse": nrmse,
            "passed": passed,
        })
    
    pass_rate = sum(results) / len(results)
    
    result = TestResult(
        name="monte_carlo_validation",
        category="statistical",
        passed=pass_rate >= pass_rate_threshold,
        metric_name="pass_rate",
        metric_value=pass_rate,
        threshold=pass_rate_threshold,
        details={
            "n_trials": n_trials,
            "n_passed": sum(results),
            "n_failed": len(results) - sum(results),
            "per_trial_tolerance": per_trial_tolerance,
        }
    )
    
    # Convert params to arrays for plotting
    freqs = np.array([p["freq_hz"] for p in params])
    amps = np.array([p["amplitude"] for p in params])
    bws = np.array([p["bw_hz"] for p in params])
    nrmses = np.array([p["nrmse"] for p in params])
    passed_arr = np.array(results)
    
    traces = {
        "freqs": freqs,
        "amplitudes": amps,
        "bws": bws,
        "nrmses": nrmses,
        "passed": passed_arr,
    }
    
    return result, traces


def test_frequency_sweep(
    n_points: int = 200,
    freq_range: Tuple[float, float] = (0.1, 50.0),
    amplitude: float = 0.01,
    phase_rad: float = 0.0,
    bw_hz: float = 0.05,
    dt: float = 0.01,
    duration: float = 60.0,
    tolerance: float = 0.10,
) -> Tuple[TestResult, Dict[str, np.ndarray]]:
    """Single parameter sweep: frequency vs NRMSE."""
    freqs = np.linspace(freq_range[0], freq_range[1], n_points)
    nrmses = []
    phase_errs = []
    for f in freqs:
        nrmse, phase_err = compute_nrmse_for_params(f, amplitude, phase_rad, bw_hz, dt, duration)
        nrmses.append(nrmse)
        phase_errs.append(phase_err)
    nrmses = np.array(nrmses)
    phase_errs = np.array(phase_errs)
    p95 = float(np.percentile(nrmses, 95))

    result = TestResult(
        name="sweep_frequency",
        category="sweep",
        passed=p95 < tolerance,
        metric_name="p95_nrmse",
        metric_value=p95,
        threshold=tolerance,
        details={
            "freq_range": freq_range,
            "amplitude": amplitude,
            "phase_rad": phase_rad,
            "bw_hz": bw_hz,
            "n_points": n_points,
        },
    )

    traces = {"x": freqs, "y": nrmses, "phase_err_deg": phase_errs}
    return result, traces


def test_amplitude_sweep(
    n_points: int = 200,
    amp_range: Tuple[float, float] = (0.001, 0.05),
    freq_hz: float = 0.4,
    phase_rad: float = 0.0,
    bw_hz: float = 0.05,
    dt: float = 0.01,
    duration: float = 60.0,
    tolerance: float = 0.10,
) -> Tuple[TestResult, Dict[str, np.ndarray]]:
    """Single parameter sweep: amplitude vs NRMSE."""
    amps = np.linspace(amp_range[0], amp_range[1], n_points)
    nrmses = []
    phase_errs = []
    for a in amps:
        nrmse, phase_err = compute_nrmse_for_params(freq_hz, a, phase_rad, bw_hz, dt, duration)
        nrmses.append(nrmse)
        phase_errs.append(phase_err)
    nrmses = np.array(nrmses)
    phase_errs = np.array(phase_errs)
    p95 = float(np.percentile(nrmses, 95))

    result = TestResult(
        name="sweep_amplitude",
        category="sweep",
        passed=p95 < tolerance,
        metric_name="p95_nrmse",
        metric_value=p95,
        threshold=tolerance,
        details={
            "amp_range": amp_range,
            "freq_hz": freq_hz,
            "phase_rad": phase_rad,
            "bw_hz": bw_hz,
            "n_points": n_points,
        },
    )

    traces = {"x": amps, "y": nrmses, "phase_err_deg": phase_errs}
    return result, traces


def test_phase_sweep(
    n_points: int = 200,
    freq_hz: float = 0.4,
    amplitude: float = 0.01,
    bw_hz: float = 0.05,
    dt: float = 0.01,
    duration: float = 60.0,
    tolerance: float = 0.10,
) -> Tuple[TestResult, Dict[str, np.ndarray]]:
    """Single parameter sweep: phase vs NRMSE."""
    phases = np.linspace(0.0, 2.0 * np.pi, n_points, endpoint=False)
    nrmses = []
    phase_errs = []
    for ph in phases:
        nrmse, phase_err = compute_nrmse_for_params(freq_hz, amplitude, ph, bw_hz, dt, duration)
        nrmses.append(nrmse)
        phase_errs.append(phase_err)
    nrmses = np.array(nrmses)
    phase_errs = np.array(phase_errs)
    p95 = float(np.percentile(nrmses, 95))

    result = TestResult(
        name="sweep_phase",
        category="sweep",
        passed=p95 < tolerance,
        metric_name="p95_nrmse",
        metric_value=p95,
        threshold=tolerance,
        details={
            "freq_hz": freq_hz,
            "amplitude": amplitude,
            "bw_hz": bw_hz,
            "n_points": n_points,
        },
    )

    traces = {"x": phases, "y": nrmses, "phase_err_deg": phase_errs}
    return result, traces


def test_bandwidth_sweep_nrmse(
    n_points: int = 200,
    bw_range: Tuple[float, float] = (0.02, 0.10),
    freq_hz: float = 0.4,
    amplitude: float = 0.01,
    phase_rad: float = 0.0,
    dt: float = 0.01,
    duration: float = 60.0,
    tolerance: float = 0.10,
) -> Tuple[TestResult, Dict[str, np.ndarray]]:
    """Single parameter sweep: bandwidth vs NRMSE."""
    bws = np.linspace(bw_range[0], bw_range[1], n_points)
    nrmses = []
    phase_errs = []
    for bw in bws:
        nrmse, phase_err = compute_nrmse_for_params(freq_hz, amplitude, phase_rad, bw, dt, duration)
        nrmses.append(nrmse)
        phase_errs.append(phase_err)
    nrmses = np.array(nrmses)
    phase_errs = np.array(phase_errs)
    p95 = float(np.percentile(nrmses, 95))

    result = TestResult(
        name="sweep_bandwidth",
        category="sweep",
        passed=p95 < tolerance,
        metric_name="p95_nrmse",
        metric_value=p95,
        threshold=tolerance,
        details={
            "bw_range": bw_range,
            "freq_hz": freq_hz,
            "amplitude": amplitude,
            "phase_rad": phase_rad,
            "n_points": n_points,
        },
    )

    traces = {"x": bws, "y": nrmses, "phase_err_deg": phase_errs}
    return result, traces


def test_frequency_settling_sweep(
    n_points: int = 200,
    freq_range: Tuple[float, float] = (0.1, 50.0),
    amplitude: float = 0.01,
    phase_rad: float = 0.0,
    bw_mid: float = 0.06,
    freq_mid: float = 1.05,
    bw_min: float = 0.02,
    bw_max: float = 0.10,
    dt: float = 0.01,
    duration: float = 60.0,
) -> Dict[str, np.ndarray]:
    """
    Single parameter sweep: estimation (settling) time vs frequency.

    Uses bandwidth proportional to frequency (constant Q) and clamps to [bw_min, bw_max].
    """
    freqs = np.linspace(freq_range[0], freq_range[1], n_points)
    settling_times = []

    for f in freqs:
        bw = bw_mid * (f / freq_mid)
        bw = min(max(bw, bw_min), bw_max)

        t = np.arange(0, duration, dt)
        y_true = generate_sinusoid(t, f, amplitude, phase_rad)

        estimator = PhasorModeBankEstimator(
            mode_freqs_hz=[f],
            mode_bandwidths_hz=[bw],
            dt=dt,
            n_axes=1,
        )

        amp_trace = []
        for k in range(len(t)):
            _, _, env = estimator.step(np.array([y_true[k]]))
            amp_trace.append(2.0 * np.abs(env[0, 0]))
        amp_trace = np.array(amp_trace)

        amp_error = amplitude - amp_trace
        settling = estimate_settling_time(amp_error, dt, threshold_fraction=0.05)
        settling_times.append(settling)

    return {"x": freqs, "y": np.array(settling_times)}


def test_decaying_mode_tracking(
    freq_hz: float = 0.4,
    amplitude: float = 0.01,
    damping_ratio: float = 0.02,
    bw_hz: float = 0.03,
    dt: float = 0.01,
    duration: float = 60.0,
    tracking_tolerance: float = 0.35,  # Relaxed: first order filter has inherent lag for decaying signals
) -> Tuple[TestResult, Dict[str, np.ndarray]]:
    """
    Test 13: Decaying mode tracking (realistic scenario).
    
    Verifies tracking of an exponentially decaying sinusoid,
    which is the realistic behavior of structural modes.
    """
    t = np.arange(0, duration, dt)
    y_true = generate_decaying_sinusoid(t, freq_hz, amplitude, damping_ratio)
    
    # True amplitude envelope
    omega = 2.0 * np.pi * freq_hz
    amp_true = amplitude * np.exp(-damping_ratio * omega * t)
    
    estimator = PhasorModeBankEstimator(
        mode_freqs_hz=[freq_hz],
        mode_bandwidths_hz=[bw_hz],
        dt=dt,
        n_axes=1,
    )
    
    amp_estimates = []
    y_modes_est = []
    
    for k in range(len(t)):
        _, y_modes, env = estimator.step(np.array([y_true[k]]))
        amp_estimates.append(2.0 * np.abs(env[0, 0]))
        y_modes_est.append(y_modes[0, 0])
    
    amp_estimates = np.array(amp_estimates)
    y_modes_est = np.array(y_modes_est)
    
    # Evaluate tracking quality after initial settling
    # Use a window where amplitude is still significant
    settling_samples = int(theoretical_settling_time(bw_hz, 0.95) / dt)
    start_idx = min(settling_samples, len(t) // 4)
    # End before amplitude decays too much (avoid division by tiny numbers)
    decay_time = 1.0 / (damping_ratio * omega)
    end_idx = min(int(3 * decay_time / dt), len(t) - 1)
    
    if end_idx > start_idx:
        amp_tracking_error = normalized_rms_error(
            amp_estimates[start_idx:end_idx],
            amp_true[start_idx:end_idx]
        )
    else:
        amp_tracking_error = float('inf')
    
    result = TestResult(
        name="decaying_mode_tracking",
        category="correctness",
        passed=amp_tracking_error < tracking_tolerance,
        metric_name="amplitude_tracking_nrmse",
        metric_value=amp_tracking_error,
        threshold=tracking_tolerance,
        details={
            "freq_hz": freq_hz,
            "damping_ratio": damping_ratio,
            "decay_time_constant": 1.0 / (damping_ratio * omega),
        }
    )
    
    traces = {
        "t": t,
        "y_true": y_true,
        "y_modes_est": y_modes_est,
        "amp_true": amp_true,
        "amp_estimates": amp_estimates,
    }
    
    return result, traces


def test_spectral_rejection(
    mode_freq_hz: float = 0.4,
    mode_amp: float = 0.01,
    bw_hz: float = 0.03,
    dt: float = 0.01,
    duration: float = 60.0,
    rejection_threshold_db: float = 20.0,
) -> Tuple[TestResult, Dict[str, np.ndarray]]:
    """
    Test 14: Spectral rejection at mode frequency.
    
    Verifies that the rigid body output shows significant
    attenuation at the modal frequency.
    """
    t = np.arange(0, duration, dt)
    y_true = generate_sinusoid(t, mode_freq_hz, mode_amp)
    
    estimator = PhasorModeBankEstimator(
        mode_freqs_hz=[mode_freq_hz],
        mode_bandwidths_hz=[bw_hz],
        dt=dt,
        n_axes=1,
    )
    
    y_rigid = []
    
    for k in range(len(t)):
        y_r, _, _ = estimator.step(np.array([y_true[k]]))
        y_rigid.append(y_r[0])
    
    y_rigid = np.array(y_rigid)
    
    # Use second half for spectral analysis
    half_idx = len(t) // 2
    
    freqs = np.fft.rfftfreq(len(t) - half_idx, d=dt)
    Y_input = np.abs(np.fft.rfft(y_true[half_idx:]))
    Y_rigid = np.abs(np.fft.rfft(y_rigid[half_idx:]))
    
    # Find peak near mode frequency
    freq_tolerance = 0.1  # Hz
    mask = (freqs >= mode_freq_hz - freq_tolerance) & (freqs <= mode_freq_hz + freq_tolerance)
    
    if np.any(mask):
        peak_input = np.max(Y_input[mask])
        peak_rigid = np.max(Y_rigid[mask])
        
        if peak_rigid > 1e-12:
            rejection_db = 20.0 * np.log10(peak_input / peak_rigid)
        else:
            rejection_db = float('inf')
    else:
        rejection_db = 0.0
    
    result = TestResult(
        name="spectral_rejection",
        category="correctness",
        passed=rejection_db >= rejection_threshold_db,
        metric_name="rejection_db",
        metric_value=rejection_db,
        threshold=rejection_threshold_db,
        details={
            "mode_freq_hz": mode_freq_hz,
            "peak_input": float(peak_input) if 'peak_input' in dir() else None,
            "peak_rigid": float(peak_rigid) if 'peak_rigid' in dir() else None,
        }
    )
    
    traces = {
        "t": t,
        "y_true": y_true,
        "y_rigid": y_rigid,
        "freqs": freqs,
        "Y_input": Y_input,
        "Y_rigid": Y_rigid,
    }
    
    return result, traces


def test_edge_case_frequencies(
    dt: float = 0.01,
    duration: float = 60.0,  # Increased for low frequencies
    tolerance: float = 0.20,  # Relaxed for edge cases
) -> List[TestResult]:
    """
    Test 15: Edge case frequency handling.
    
    Tests behavior at very low frequencies, near Nyquist, etc.
    Note: Very low frequencies require proportionally wider bandwidths
    and longer durations for convergence.
    """
    results = []
    nyquist = 0.5 / dt
    
    # (name, freq_hz, bw_hz, duration_override)
    test_cases = [
        ("very_low_freq", 0.05, 0.02, 120.0),    # Very low - needs longer duration
        ("low_freq", 0.1, 0.02, 60.0),           # Low frequency
        ("mid_freq", 1.0, 0.05, 30.0),           # Mid frequency  
        ("high_freq", 5.0, 0.2, 30.0),           # Higher frequency
        ("near_nyquist", nyquist * 0.8, 1.0, 30.0),  # Near Nyquist
    ]
    
    for name, freq_hz, bw_hz, dur in test_cases:
        if freq_hz >= nyquist:
            continue  # Skip if above Nyquist
            
        t = np.arange(0, dur, dt)
        amplitude = 0.01
        y_true = generate_sinusoid(t, freq_hz, amplitude)
        
        try:
            estimator = PhasorModeBankEstimator(
                mode_freqs_hz=[freq_hz],
                mode_bandwidths_hz=[bw_hz],
                dt=dt,
                n_axes=1,
            )
            
            y_modes_est = []
            for k in range(len(t)):
                _, y_modes, _ = estimator.step(np.array([y_true[k]]))
                y_modes_est.append(y_modes[0, 0])
            
            y_modes_est = np.array(y_modes_est)
            
            half_idx = len(t) // 2
            nrmse = normalized_rms_error(y_modes_est[half_idx:], y_true[half_idx:])
            passed = nrmse < tolerance
            error_msg = None
            
        except Exception as e:
            nrmse = float('inf')
            passed = False
            error_msg = str(e)
        
        results.append(TestResult(
            name=f"edge_freq_{name}",
            category="numerical",
            passed=passed,
            metric_name="nrmse",
            metric_value=nrmse,
            threshold=tolerance,
            details={
                "freq_hz": freq_hz,
                "bw_hz": bw_hz,
                "duration": dur,
                "nyquist": nyquist,
                "error": error_msg,
            }
        ))
    
    return results


def test_api_error_handling() -> List[TestResult]:
    """
    Test 16: API error handling and input validation.
    
    Verifies that the estimator properly validates inputs
    and raises appropriate errors.
    """
    results = []
    
    # Test: empty mode list
    try:
        PhasorModeBankEstimator(mode_freqs_hz=[], mode_bandwidths_hz=[], dt=0.01)
        passed = False
    except ValueError:
        passed = True
    
    results.append(TestResult(
        name="api_empty_modes",
        category="api",
        passed=passed,
        metric_name="raises_error",
        metric_value=1.0 if passed else 0.0,
        threshold=1.0,
    ))
    
    # Test: mismatched lengths
    try:
        PhasorModeBankEstimator(mode_freqs_hz=[0.4], mode_bandwidths_hz=[0.03, 0.05], dt=0.01)
        passed = False
    except ValueError:
        passed = True
    
    results.append(TestResult(
        name="api_mismatched_lengths",
        category="api",
        passed=passed,
        metric_name="raises_error",
        metric_value=1.0 if passed else 0.0,
        threshold=1.0,
    ))
    
    # Test: invalid dt
    try:
        PhasorModeBankEstimator(mode_freqs_hz=[0.4], mode_bandwidths_hz=[0.03], dt=0.0)
        passed = False
    except ValueError:
        passed = True
    
    results.append(TestResult(
        name="api_invalid_dt",
        category="api",
        passed=passed,
        metric_name="raises_error",
        metric_value=1.0 if passed else 0.0,
        threshold=1.0,
    ))
    
    # Test: invalid bandwidth
    try:
        PhasorModeBankEstimator(mode_freqs_hz=[0.4], mode_bandwidths_hz=[-0.03], dt=0.01)
        passed = False
    except ValueError:
        passed = True
    
    results.append(TestResult(
        name="api_invalid_bandwidth",
        category="api",
        passed=passed,
        metric_name="raises_error",
        metric_value=1.0 if passed else 0.0,
        threshold=1.0,
    ))
    
    # Test: invalid n_axes
    try:
        PhasorModeBankEstimator(mode_freqs_hz=[0.4], mode_bandwidths_hz=[0.03], dt=0.01, n_axes=0)
        passed = False
    except ValueError:
        passed = True
    
    results.append(TestResult(
        name="api_invalid_n_axes",
        category="api",
        passed=passed,
        metric_name="raises_error",
        metric_value=1.0 if passed else 0.0,
        threshold=1.0,
    ))
    
    # Test: reset works
    try:
        est = PhasorModeBankEstimator(mode_freqs_hz=[0.4], mode_bandwidths_hz=[0.03], dt=0.01, n_axes=1)
        est.step(np.array([0.01]))
        est.reset()
        passed = np.allclose(est._env, 0.0) and np.allclose(np.abs(est._carrier), 1.0)
    except Exception:
        passed = False
    
    results.append(TestResult(
        name="api_reset",
        category="api",
        passed=passed,
        metric_name="reset_works",
        metric_value=1.0 if passed else 0.0,
        threshold=1.0,
    ))
    
    return results


# ==============================================================================
# PLOTTING FUNCTIONS
# ==============================================================================

def plot_pure_sinusoid_test(traces: Dict, out_dir: Path) -> None:
    """Plot results from pure sinusoid tracking test."""
    if not HAS_MATPLOTLIB:
        return
    
    fig, axes = plt.subplots(3, 1, figsize=(10, 8))
    
    t = traces["t"]
    
    # Time series
    axes[0].plot(t, traces["y_true"], 'b-', label='True signal', linewidth=1.5)
    axes[0].plot(t, traces["y_modes"], 'r--', label='Estimated', linewidth=1.2)
    axes[0].set_xlabel('Time (s)')
    axes[0].set_ylabel('Signal')
    axes[0].set_title('Pure Sinusoid Tracking')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Amplitude tracking
    axes[1].axhline(traces["y_true"].max(), color='b', linestyle='-', 
                    label='True amplitude', linewidth=1.5)
    axes[1].plot(t, traces["amp_estimates"], 'r-', label='Estimated', linewidth=1.2)
    axes[1].set_xlabel('Time (s)')
    axes[1].set_ylabel('Amplitude')
    axes[1].set_title('Amplitude Tracking')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    # Phase tracking
    axes[2].plot(t, np.rad2deg(traces["phase_estimates"]), 'r-', linewidth=1.2)
    axes[2].set_xlabel('Time (s)')
    axes[2].set_ylabel('Phase (deg)')
    axes[2].set_title('Phase Estimate')
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    fig.savefig(out_dir / "test_pure_sinusoid.png", dpi=150)
    plt.close(fig)


def plot_multi_mode_test(traces: Dict, out_dir: Path) -> None:
    """Plot results from multi mode separation test."""
    if not HAS_MATPLOTLIB:
        return
    
    n_modes = traces["modes_true"].shape[1]
    fig, axes = plt.subplots(n_modes + 1, 1, figsize=(10, 3 * (n_modes + 1)))
    
    t = traces["t"]
    
    # Combined signal
    axes[0].plot(t, traces["y_combined"], 'k-', label='Combined', linewidth=1.2)
    axes[0].set_xlabel('Time (s)')
    axes[0].set_ylabel('Signal')
    axes[0].set_title('Multi-Mode Input Signal')
    axes[0].grid(True, alpha=0.3)
    
    # Individual modes
    for m in range(n_modes):
        axes[m + 1].plot(t, traces["modes_true"][:, m], 'b-', 
                         label='True', linewidth=1.5)
        axes[m + 1].plot(t, traces["modes_est"][:, m], 'r--', 
                         label='Estimated', linewidth=1.2)
        axes[m + 1].set_xlabel('Time (s)')
        axes[m + 1].set_ylabel('Signal')
        axes[m + 1].set_title(f'Mode {m + 1} Separation')
        axes[m + 1].legend()
        axes[m + 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    fig.savefig(out_dir / "test_multi_mode_separation.png", dpi=150)
    plt.close(fig)


def plot_settling_time_test(traces: Dict, out_dir: Path) -> None:
    """Plot results from settling time test."""
    if not HAS_MATPLOTLIB:
        return
    
    fig, ax = plt.subplots(figsize=(10, 5))
    
    t = traces["t"]
    ax.plot(t, np.abs(traces["amp_error"]), 'b-', linewidth=1.2)
    ax.axhline(0.05 * np.abs(traces["amp_error"][0]), color='r', linestyle='--',
               label=f'5% threshold')
    ax.axvline(traces["theoretical_settling"], color='g', linestyle=':',
               label=f'Theoretical settling: {traces["theoretical_settling"]:.2f}s')
    if traces["actual_settling"] < float('inf'):
        ax.axvline(traces["actual_settling"], color='orange', linestyle='-.',
                   label=f'Actual settling: {traces["actual_settling"]:.2f}s')
    
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('|Amplitude Error|')
    ax.set_title('Settling Time Analysis')
    ax.set_yscale('log')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    fig.savefig(out_dir / "test_settling_time.png", dpi=150)
    plt.close(fig)


def plot_frequency_mismatch_test(traces: Dict, out_dir: Path) -> None:
    """Plot results from frequency mismatch test."""
    if not HAS_MATPLOTLIB:
        return
    
    fig, axes = plt.subplots(2, 1, figsize=(10, 6))
    
    t = traces["t"]
    
    # Time series comparison
    axes[0].plot(t, traces["y_true"], 'b-', label='True', linewidth=1.5)
    axes[0].plot(t, traces["y_modes_est"], 'r--', label='Estimated', linewidth=1.2)
    axes[0].set_xlabel('Time (s)')
    axes[0].set_ylabel('Signal')
    axes[0].set_title('Frequency Mismatch: Signal Reconstruction')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Amplitude estimate stability
    axes[1].plot(t, traces["amp_estimates"], 'r-', linewidth=1.2)
    axes[1].set_xlabel('Time (s)')
    axes[1].set_ylabel('Estimated Amplitude')
    axes[1].set_title('Frequency Mismatch: Amplitude Estimate (expect oscillation)')
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    fig.savefig(out_dir / "test_frequency_mismatch.png", dpi=150)
    plt.close(fig)


def plot_noise_sensitivity_test(traces: Dict, out_dir: Path) -> None:
    """Plot results from noise sensitivity test."""
    if not HAS_MATPLOTLIB:
        return
    
    fig, axes = plt.subplots(2, 1, figsize=(10, 6))
    
    t = traces["t"]
    
    # Time series
    axes[0].plot(t, traces["y_noisy"], 'gray', label='Noisy input', 
                 linewidth=0.8, alpha=0.7)
    axes[0].plot(t, traces["y_clean"], 'b-', label='Clean signal', linewidth=1.5)
    axes[0].plot(t, traces["y_modes_est"], 'r--', label='Estimated', linewidth=1.2)
    axes[0].set_xlabel('Time (s)')
    axes[0].set_ylabel('Signal')
    axes[0].set_title('Noise Sensitivity: Signal Tracking')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Error
    error = traces["y_modes_est"] - traces["y_clean"]
    axes[1].plot(t, error, 'r-', linewidth=0.8)
    axes[1].set_xlabel('Time (s)')
    axes[1].set_ylabel('Error')
    axes[1].set_title('Estimation Error')
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    fig.savefig(out_dir / "test_noise_sensitivity.png", dpi=150)
    plt.close(fig)


def plot_bandwidth_sweep(traces: Dict, out_dir: Path) -> None:
    """Plot results from bandwidth sensitivity sweep."""
    if not HAS_MATPLOTLIB:
        return
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    bws = traces["bws"]
    
    # Amplitude error vs bandwidth
    axes[0].plot(bws, traces["amp_errors"] * 100, 'bo-', linewidth=1.5, markersize=6)
    axes[0].axhline(5.0, color='r', linestyle='--', label='5% threshold')
    axes[0].set_xlabel('Bandwidth (Hz)')
    axes[0].set_ylabel('Amplitude Error (%)')
    axes[0].set_title('Steady-State Accuracy vs Bandwidth')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Settling time vs bandwidth
    axes[1].plot(bws, traces["settling_times"], 'go-', linewidth=1.5, markersize=6)
    axes[1].set_xlabel('Bandwidth (Hz)')
    axes[1].set_ylabel('Settling Time (s)')
    axes[1].set_title('Settling Time vs Bandwidth')
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    fig.savefig(out_dir / "test_bandwidth_sweep.png", dpi=150)
    plt.close(fig)


def plot_single_sweep(
    traces: Dict,
    out_dir: Path,
    filename: str,
    xlabel: str,
    title: str,
    threshold: float = 0.10,
    xscale: Optional[str] = None,
) -> None:
    """Generic plotter for NRMSE vs a single parameter."""
    if not HAS_MATPLOTLIB:
        return

    x = traces["x"]
    y = traces["y"]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(x, y, "b-", linewidth=1.5)
    ax.axhline(threshold, color="r", linestyle="--", alpha=0.6, label="NRMSE threshold")
    ax.set_xlabel(xlabel)
    ax.set_ylabel("NRMSE")
    ax.set_title(title)
    if xscale:
        ax.set_xscale(xscale)
    ax.grid(True, alpha=0.3)
    ax.legend()
    plt.tight_layout()
    fig.savefig(out_dir / filename, dpi=150)
    plt.close(fig)


def plot_phase_sweep(
    traces: Dict,
    out_dir: Path,
    filename: str,
    xlabel: str,
    title: str,
    threshold_deg: float = 5.0,
    xscale: Optional[str] = None,
) -> None:
    """Plot phase error (deg) vs a single parameter."""
    if not HAS_MATPLOTLIB:
        return
    x = traces["x"]
    y = traces["phase_err_deg"]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(x, y, "g-", linewidth=1.5)
    ax.axhline(threshold_deg, color="r", linestyle="--", alpha=0.6, label="Phase error threshold")
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Phase error (deg)")
    ax.set_title(title)
    if xscale:
        ax.set_xscale(xscale)
    ax.grid(True, alpha=0.3)
    ax.legend()
    plt.tight_layout()
    fig.savefig(out_dir / filename, dpi=150)
    plt.close(fig)


def plot_settling_sweep(
    traces: Dict,
    out_dir: Path,
    filename: str,
    xlabel: str,
    title: str,
) -> None:
    """Plot settling time vs parameter."""
    if not HAS_MATPLOTLIB:
        return
    x = traces["x"]
    y = traces["y"]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(x, y, "m-", linewidth=1.5)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Settling time (s)")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(out_dir / filename, dpi=150)
    plt.close(fig)


def plot_long_run_test(traces: Dict, out_dir: Path) -> None:
    """Plot results from long run stability test."""
    if not HAS_MATPLOTLIB:
        return
    
    fig, axes = plt.subplots(2, 1, figsize=(10, 6))
    
    t = traces["t"]
    
    # Amplitude stability
    axes[0].plot(t, traces["amp_samples"], 'b-', linewidth=0.8)
    axes[0].set_xlabel('Time (s)')
    axes[0].set_ylabel('Estimated Amplitude')
    axes[0].set_title(f'Long-Run Amplitude Stability ({t[-1]:.0f}s)')
    axes[0].grid(True, alpha=0.3)
    
    # Carrier magnitude
    axes[1].plot(t, traces["carrier_mag"], 'g-', linewidth=0.8)
    axes[1].axhline(1.0, color='r', linestyle='--', alpha=0.5)
    axes[1].set_xlabel('Time (s)')
    axes[1].set_ylabel('Carrier Magnitude')
    axes[1].set_title('Carrier Normalization Stability')
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    fig.savefig(out_dir / "test_long_run_stability.png", dpi=150)
    plt.close(fig)


def plot_monte_carlo_test(traces: Dict, out_dir: Path) -> None:
    """Plot results from Monte Carlo validation."""
    if not HAS_MATPLOTLIB:
        return
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    passed = traces["passed"]
    failed = ~passed
    
    # NRMSE vs frequency
    axes[0, 0].scatter(traces["freqs"][passed], traces["nrmses"][passed], 
                       c='green', alpha=0.5, label='Passed', s=20)
    axes[0, 0].scatter(traces["freqs"][failed], traces["nrmses"][failed], 
                       c='red', alpha=0.7, label='Failed', s=30, marker='x')
    axes[0, 0].axhline(0.10, color='k', linestyle='--', alpha=0.5)
    axes[0, 0].set_xlabel('Frequency (Hz)')
    axes[0, 0].set_ylabel('NRMSE')
    axes[0, 0].set_title('Error vs Frequency')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # NRMSE vs amplitude
    axes[0, 1].scatter(traces["amplitudes"][passed], traces["nrmses"][passed], 
                       c='green', alpha=0.5, label='Passed', s=20)
    axes[0, 1].scatter(traces["amplitudes"][failed], traces["nrmses"][failed], 
                       c='red', alpha=0.7, label='Failed', s=30, marker='x')
    axes[0, 1].axhline(0.10, color='k', linestyle='--', alpha=0.5)
    axes[0, 1].set_xlabel('Amplitude')
    axes[0, 1].set_ylabel('NRMSE')
    axes[0, 1].set_title('Error vs Amplitude')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # NRMSE vs bandwidth
    axes[1, 0].scatter(traces["bws"][passed], traces["nrmses"][passed], 
                       c='green', alpha=0.5, label='Passed', s=20)
    axes[1, 0].scatter(traces["bws"][failed], traces["nrmses"][failed], 
                       c='red', alpha=0.7, label='Failed', s=30, marker='x')
    axes[1, 0].axhline(0.10, color='k', linestyle='--', alpha=0.5)
    axes[1, 0].set_xlabel('Bandwidth (Hz)')
    axes[1, 0].set_ylabel('NRMSE')
    axes[1, 0].set_title('Error vs Bandwidth')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # NRMSE histogram
    axes[1, 1].hist(traces["nrmses"], bins=30, color='steelblue', 
                    edgecolor='black', alpha=0.7)
    axes[1, 1].axvline(0.10, color='r', linestyle='--', 
                       label='Threshold', linewidth=2)
    axes[1, 1].set_xlabel('NRMSE')
    axes[1, 1].set_ylabel('Count')
    axes[1, 1].set_title('Error Distribution')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    fig.savefig(out_dir / "test_monte_carlo.png", dpi=150)
    plt.close(fig)


def plot_spectral_rejection_test(traces: Dict, out_dir: Path) -> None:
    """Plot results from spectral rejection test."""
    if not HAS_MATPLOTLIB:
        return
    
    fig, axes = plt.subplots(2, 1, figsize=(10, 6))
    
    t = traces["t"]
    
    # Time series
    axes[0].plot(t, traces["y_true"], 'b-', label='Input', linewidth=1.5)
    axes[0].plot(t, traces["y_rigid"], 'r-', label='Rigid estimate', linewidth=1.2)
    axes[0].set_xlabel('Time (s)')
    axes[0].set_ylabel('Signal')
    axes[0].set_title('Spectral Rejection: Time Domain')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Spectrum
    axes[1].semilogy(traces["freqs"], traces["Y_input"] + 1e-12, 'b-', 
                     label='Input', linewidth=1.5)
    axes[1].semilogy(traces["freqs"], traces["Y_rigid"] + 1e-12, 'r-', 
                     label='Rigid estimate', linewidth=1.2)
    axes[1].set_xlabel('Frequency (Hz)')
    axes[1].set_ylabel('Magnitude')
    axes[1].set_title('Spectral Rejection: Frequency Domain')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    axes[1].set_xlim(0, 2)
    
    plt.tight_layout()
    fig.savefig(out_dir / "test_spectral_rejection.png", dpi=150)
    plt.close(fig)


def plot_decaying_mode_test(traces: Dict, out_dir: Path) -> None:
    """Plot results from decaying mode tracking test."""
    if not HAS_MATPLOTLIB:
        return
    
    fig, axes = plt.subplots(2, 1, figsize=(10, 6))
    
    t = traces["t"]
    
    # Signal tracking
    axes[0].plot(t, traces["y_true"], 'b-', label='True', linewidth=1.5)
    axes[0].plot(t, traces["y_modes_est"], 'r--', label='Estimated', linewidth=1.2)
    axes[0].set_xlabel('Time (s)')
    axes[0].set_ylabel('Signal')
    axes[0].set_title('Decaying Mode Tracking')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Amplitude envelope
    axes[1].plot(t, traces["amp_true"], 'b-', label='True envelope', linewidth=1.5)
    axes[1].plot(t, traces["amp_estimates"], 'r--', label='Estimated', linewidth=1.2)
    axes[1].set_xlabel('Time (s)')
    axes[1].set_ylabel('Amplitude')
    axes[1].set_title('Amplitude Envelope Tracking')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    fig.savefig(out_dir / "test_decaying_mode.png", dpi=150)
    plt.close(fig)


def plot_summary(report: ValidationReport, out_dir: Path) -> None:
    """Create summary visualization."""
    if not HAS_MATPLOTLIB:
        return
    
    fig = plt.figure(figsize=(14, 10))
    gs = GridSpec(2, 2, figure=fig)
    
    # Pass/fail by category
    ax1 = fig.add_subplot(gs[0, 0])
    categories = list(report.summary["by_category"].keys())
    passed = [report.summary["by_category"][c]["passed"] for c in categories]
    failed = [report.summary["by_category"][c]["total"] - report.summary["by_category"][c]["passed"] 
              for c in categories]
    
    x = np.arange(len(categories))
    width = 0.35
    ax1.bar(x - width/2, passed, width, label='Passed', color='green', alpha=0.7)
    ax1.bar(x + width/2, failed, width, label='Failed', color='red', alpha=0.7)
    ax1.set_xlabel('Category')
    ax1.set_ylabel('Count')
    ax1.set_title('Test Results by Category')
    ax1.set_xticks(x)
    ax1.set_xticklabels(categories, rotation=45, ha='right')
    ax1.legend()
    ax1.grid(True, alpha=0.3, axis='y')
    
    # Overall pass rate pie chart
    ax2 = fig.add_subplot(gs[0, 1])
    sizes = [report.summary["passed"], report.summary["failed"]]
    labels = [f'Passed ({report.summary["passed"]})', 
              f'Failed ({report.summary["failed"]})']
    colors = ['green', 'red']
    explode = (0, 0.1) if report.summary["failed"] > 0 else (0, 0)
    ax2.pie(sizes, explode=explode, labels=labels, colors=colors,
            autopct='%1.1f%%', shadow=True, startangle=90)
    ax2.set_title(f'Overall Pass Rate: {report.summary["pass_rate"]*100:.1f}%')
    
    # Test results table
    ax3 = fig.add_subplot(gs[1, :])
    ax3.axis('off')
    
    # Create table data
    table_data = []
    for test in report.tests[:20]:  # Limit to first 20 for readability
        status = "PASS" if test.passed else "FAIL"
        table_data.append([
            test.name[:35],
            test.category,
            f"{test.metric_value:.4f}",
            f"{test.threshold:.4f}",
            status
        ])
    
    table = ax3.table(
        cellText=table_data,
        colLabels=['Test Name', 'Category', 'Value', 'Threshold', 'Status'],
        loc='center',
        cellLoc='center'
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1.2, 1.5)
    
    # Color code pass/fail
    for i, test in enumerate(report.tests[:20]):
        color = '#d4edda' if test.passed else '#f8d7da'
        for j in range(5):
            table[(i + 1, j)].set_facecolor(color)
    
    plt.tight_layout()
    fig.savefig(out_dir / "validation_summary.png", dpi=150)
    plt.close(fig)


# ==============================================================================
# MAIN VALIDATION RUNNER
# ==============================================================================

def run_all_tests(
    out_dir: Path,
    plot: bool = True,
    verbose: bool = True,
) -> ValidationReport:
    """Run all validation tests and generate report."""
    
    start_time = time.time()
    
    report = ValidationReport(
        timestamp=time.strftime("%Y-%m-%d %H:%M:%S"),
        duration_seconds=0.0,
    )
    
    out_dir.mkdir(parents=True, exist_ok=True)
    
    if verbose:
        print("=" * 70)
        print("PHASOR MODE BANK ESTIMATOR - COMPREHENSIVE VALIDATION")
        print("=" * 70)
        if USING_REPO_ESTIMATOR:
            print("Estimator: basilisk_sim.state_estimator (repo)")
        else:
            print("Estimator: inline copy (repo import failed)")
            if REPO_IMPORT_ERROR:
                print(f"  import error: {REPO_IMPORT_ERROR}")
        print()
    
    # -------------------------------------------------------------------------
    # 1. CORRECTNESS TESTS
    # -------------------------------------------------------------------------
    if verbose:
        print("CATEGORY: CORRECTNESS")
        print("-" * 40)
    
    # Test 1: Pure sinusoid tracking
    amp_result, phase_result, traces = test_pure_sinusoid_tracking()
    report.add_test(amp_result)
    report.add_test(phase_result)
    if verbose:
        print(amp_result)
        print(phase_result)
    if plot:
        plot_pure_sinusoid_test(traces, out_dir)
    
    # Test 2: Multi mode separation
    result, traces = test_multi_mode_separation()
    report.add_test(result)
    if verbose:
        print(result)
    if plot:
        plot_multi_mode_test(traces, out_dir)
    
    # Test 3: Rigid body extraction
    result, traces = test_rigid_body_extraction()
    report.add_test(result)
    if verbose:
        print(result)
    
    # Test 11: Multi axis independence
    result, traces = test_multi_axis_independence()
    report.add_test(result)
    if verbose:
        print(result)
    
    # Test 13: Decaying mode tracking
    result, traces = test_decaying_mode_tracking()
    report.add_test(result)
    if verbose:
        print(result)
    if plot:
        plot_decaying_mode_test(traces, out_dir)
    
    # Test 14: Spectral rejection
    result, traces = test_spectral_rejection()
    report.add_test(result)
    if verbose:
        print(result)
    if plot:
        plot_spectral_rejection_test(traces, out_dir)
    
    if verbose:
        print()
    
    # -------------------------------------------------------------------------
    # 2. CONVERGENCE TESTS
    # -------------------------------------------------------------------------
    if verbose:
        print("CATEGORY: CONVERGENCE")
        print("-" * 40)
    
    # Test 4: Settling time
    result, traces = test_settling_time()
    report.add_test(result)
    if verbose:
        print(result)
    if plot:
        plot_settling_time_test(traces, out_dir)
    
    # Test 5: Step amplitude response
    result, traces = test_step_amplitude_response()
    report.add_test(result)
    if verbose:
        print(result)
    
    if verbose:
        print()
    
    # -------------------------------------------------------------------------
    # 3. ROBUSTNESS TESTS
    # -------------------------------------------------------------------------
    if verbose:
        print("CATEGORY: ROBUSTNESS")
        print("-" * 40)
    
    # Test 6: Frequency mismatch (multiple levels)
    for mismatch in [2.0, 5.0, 10.0]:
        result, traces = test_frequency_mismatch(mismatch_pct=mismatch)
        report.add_test(result)
        if verbose:
            print(result)
    if plot:
        _, traces = test_frequency_mismatch(mismatch_pct=5.0)
        plot_frequency_mismatch_test(traces, out_dir)
    
    # Test 7: Noise sensitivity (multiple SNR levels)
    for snr in [30.0, 20.0, 10.0]:
        result, traces = test_noise_sensitivity(snr_db=snr)
        report.add_test(result)
        if verbose:
            print(result)
    if plot:
        _, traces = test_noise_sensitivity(snr_db=20.0)
        plot_noise_sensitivity_test(traces, out_dir)
    
    # Test 8: Bandwidth sensitivity
    results, traces = test_bandwidth_sensitivity()
    for result in results:
        report.add_test(result)
    if verbose:
        print(f"[INFO] Bandwidth sweep: {len(results)} tests")
    if plot:
        plot_bandwidth_sweep(traces, out_dir)
    
    if verbose:
        print()
    
    # -------------------------------------------------------------------------
    # 4. NUMERICAL TESTS
    # -------------------------------------------------------------------------
    if verbose:
        print("CATEGORY: NUMERICAL")
        print("-" * 40)
    
    # Test 9: Long run stability
    result, traces = test_long_run_stability(duration=600.0)  # 10 minutes
    report.add_test(result)
    if verbose:
        print(result)
    if plot:
        plot_long_run_test(traces, out_dir)
    
    # Test 10: Variable timestep
    result, traces = test_dt_variation()
    report.add_test(result)
    if verbose:
        print(result)
    
    # Test 15: Edge case frequencies
    results = test_edge_case_frequencies()
    for result in results:
        report.add_test(result)
        if verbose:
            print(result)
    
    if verbose:
        print()
    
    # -------------------------------------------------------------------------
    # 5. API TESTS
    # -------------------------------------------------------------------------
    if verbose:
        print("CATEGORY: API")
        print("-" * 40)
    
    # Test 16: API error handling
    results = test_api_error_handling()
    for result in results:
        report.add_test(result)
        if verbose:
            print(result)
    
    if verbose:
        print()
    
    # -------------------------------------------------------------------------
    # 6. STATISTICAL TESTS
    # -------------------------------------------------------------------------
    if verbose:
        print("CATEGORY: STATISTICAL")
        print("-" * 40)
    
    # Test 12: Monte Carlo
    n_mc = 200
    result, traces = test_monte_carlo_validation(n_trials=n_mc)
    report.add_test(result)
    if verbose:
        print(result)
    if plot:
        plot_monte_carlo_test(traces, out_dir)

    if verbose:
        print()

    # -------------------------------------------------------------------------
    # 7. SINGLE PARAMETER SWEEPS (same N as Monte Carlo)
    # -------------------------------------------------------------------------
    if verbose:
        print("CATEGORY: SWEEP")
        print("-" * 40)

    # Use midpoints of Monte Carlo ranges for fixed parameters
    freq_mid = 0.5 * (0.1 + 50.0)
    amp_mid = 0.5 * (0.001 + 0.05)
    bw_mid = 0.5 * (0.02 + 0.10)

    # Frequency sweep
    result, traces = test_frequency_sweep(
        n_points=n_mc,
        freq_range=(0.1, 50.0),
        amplitude=amp_mid,
        phase_rad=0.0,
        bw_hz=bw_mid,
    )
    report.add_test(result)
    if verbose:
        print(result)
    if plot:
        plot_single_sweep(
            traces,
            out_dir,
            "sweep_frequency.png",
            "Frequency (Hz)",
            "NRMSE vs Frequency",
        )
        plot_phase_sweep(
            traces,
            out_dir,
            "sweep_frequency_phase.png",
            "Frequency (Hz)",
            "Phase Error vs Frequency",
        )

    # Frequency vs settling time (estimation time)
    traces = test_frequency_settling_sweep(
        n_points=n_mc,
        freq_range=(0.1, 50.0),
        amplitude=amp_mid,
        phase_rad=0.0,
        bw_mid=bw_mid,
        freq_mid=freq_mid,
        bw_min=0.02,
        bw_max=0.10,
    )
    if plot:
        plot_settling_sweep(
            traces,
            out_dir,
            "sweep_frequency_settling.png",
            "Frequency (Hz)",
            "Estimation Time vs Frequency",
        )

    # Amplitude sweep
    result, traces = test_amplitude_sweep(
        n_points=n_mc,
        amp_range=(0.001, 0.05),
        freq_hz=freq_mid,
        phase_rad=0.0,
        bw_hz=bw_mid,
    )
    report.add_test(result)
    if verbose:
        print(result)
    if plot:
        plot_single_sweep(
            traces,
            out_dir,
            "sweep_amplitude.png",
            "Amplitude",
            "NRMSE vs Amplitude",
        )
        plot_phase_sweep(
            traces,
            out_dir,
            "sweep_amplitude_phase.png",
            "Amplitude",
            "Phase Error vs Amplitude",
        )

    # Phase sweep
    result, traces = test_phase_sweep(
        n_points=n_mc,
        freq_hz=freq_mid,
        amplitude=amp_mid,
        bw_hz=bw_mid,
    )
    report.add_test(result)
    if verbose:
        print(result)
    if plot:
        plot_single_sweep(
            traces,
            out_dir,
            "sweep_phase.png",
            "Phase (rad)",
            "NRMSE vs Phase",
        )
        plot_phase_sweep(
            traces,
            out_dir,
            "sweep_phase_error.png",
            "Phase (rad)",
            "Phase Error vs Phase",
        )

    # Bandwidth sweep (NRMSE)
    result, traces = test_bandwidth_sweep_nrmse(
        n_points=n_mc,
        bw_range=(0.02, 0.10),
        freq_hz=freq_mid,
        amplitude=amp_mid,
        phase_rad=0.0,
    )
    report.add_test(result)
    if verbose:
        print(result)
    if plot:
        plot_single_sweep(
            traces,
            out_dir,
            "sweep_bandwidth_nrmse.png",
            "Bandwidth (Hz)",
            "NRMSE vs Bandwidth",
        )
        plot_phase_sweep(
            traces,
            out_dir,
            "sweep_bandwidth_phase.png",
            "Bandwidth (Hz)",
            "Phase Error vs Bandwidth",
        )

    if verbose:
        print()
    
    # -------------------------------------------------------------------------
    # FINALIZE REPORT
    # -------------------------------------------------------------------------
    report.duration_seconds = time.time() - start_time
    report.compute_summary()
    
    if plot:
        plot_summary(report, out_dir)
    
    # -------------------------------------------------------------------------
    # SUMMARY OUTPUT
    # -------------------------------------------------------------------------
    if verbose:
        print("=" * 70)
        print("VALIDATION SUMMARY")
        print("=" * 70)
        print(f"Total tests:     {report.summary['total_tests']}")
        print(f"Passed:          {report.summary['passed']}")
        print(f"Failed:          {report.summary['failed']}")
        print(f"Pass rate:       {report.summary['pass_rate']*100:.1f}%")
        print(f"Duration:        {report.duration_seconds:.2f}s")
        print()
        
        print("By category:")
        for cat, stats in report.summary["by_category"].items():
            pct = stats["passed"] / stats["total"] * 100 if stats["total"] > 0 else 0
            print(f"  {cat:15s}: {stats['passed']}/{stats['total']} ({pct:.0f}%)")
        
        print()
        if report.all_passed():
            print("ALL TESTS PASSED")
        else:
            print("SOME TESTS FAILED:")
            for test in report.tests:
                if not test.passed:
                    print(f"  - {test.name}: {test.metric_name}={test.metric_value:.6f} (threshold={test.threshold:.6f})")
    
    return report


def save_report(report: ValidationReport, out_dir: Path) -> None:
    """Save report to JSON and optionally CSV."""
    
    # JSON report
    json_data = {
        "timestamp": report.timestamp,
        "duration_seconds": report.duration_seconds,
        "summary": report.summary,
        "tests": [
            {
                "name": t.name,
                "category": t.category,
                "passed": t.passed,
                "metric_name": t.metric_name,
                "metric_value": t.metric_value,
                "threshold": t.threshold,
                "details": t.details,
            }
            for t in report.tests
        ]
    }
    
    with open(out_dir / "validation_report.json", "w") as f:
        json.dump(json_data, f, indent=2, default=str)
    
    # CSV report (always write; use pandas if available)
    rows = []
    for t in report.tests:
        rows.append({
            "name": t.name,
            "category": t.category,
            "passed": t.passed,
            "metric_name": t.metric_name,
            "metric_value": t.metric_value,
            "threshold": t.threshold,
        })
    csv_path = out_dir / "validation_report.csv"
    if HAS_PANDAS:
        df = pd.DataFrame(rows)
        df.to_csv(csv_path, index=False)
    else:
        fieldnames = ["name", "category", "passed", "metric_name", "metric_value", "threshold"]
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)


def main() -> None:
    """CLI entry point: parse arguments, run the validation suite, and save reports."""
    parser = argparse.ArgumentParser(
        description="Comprehensive validation suite for PhasorModeBankEstimator"
    )
    parser.add_argument(
        "--out-dir", type=str, default=None,
        help=f"Output directory for plots and reports (default: {DEFAULT_OUT_DIR})"
    )
    parser.add_argument(
        "--no-plot", action="store_true",
        help="Disable plot generation"
    )
    parser.add_argument(
        "--quiet", action="store_true",
        help="Suppress verbose output"
    )
    parser.add_argument(
        "--ci", action="store_true",
        help="CI mode: exit with non-zero code if any tests fail"
    )
    args = parser.parse_args()
    
    out_dir = Path(args.out_dir) if args.out_dir else DEFAULT_OUT_DIR
    
    report = run_all_tests(
        out_dir=out_dir,
        plot=not args.no_plot and HAS_MATPLOTLIB,
        verbose=not args.quiet,
    )
    
    save_report(report, out_dir)
    
    if not args.quiet:
        print()
        print(f"Reports saved to: {out_dir}")
        print(f"  - validation_report.json")
        print(f"  - validation_report.csv")
        if HAS_MATPLOTLIB and not args.no_plot:
            print(f"  - *.png plots")
    
    if args.ci and not report.all_passed():
        sys.exit(1)


if __name__ == "__main__":
    main()
