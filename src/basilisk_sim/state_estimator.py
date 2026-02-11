"""
state_estimator.py

Frequency-domain tunable, low-compute state estimator for narrowband flexible modes.

This module implements a "phasor / lock-in" style modal estimator:
- You pick one or more target mode frequencies (Hz).
- For each frequency, the estimator keeps a complex envelope state (amplitude/phase)
  and reconstructs that narrowband component.
- Subtracting the reconstructed component from the measurement yields a
  "rigid-body" estimate with those modes removed (a notch-like effect),
  but you ALSO get a true *state* per mode (complex envelope) for monitoring/adaptation.

Why this fits your repo:
- Your spacecraft model explicitly calls out two dominant array modes at 0.4 Hz and 1.3 Hz,
  and your shaping/controllers already tune in frequency space.
- Your simulation and feedforward tooling standardize on dt=0.01 (100 Hz).

Core math (per mode):
  u[k]      = y[k] * exp(-j*ω*k*dt)        # complex demodulation
  a[k+1]    = (1-α)*a[k] + α*u[k]          # complex 1st order LPF (envelope)
  y_hat[k]  = 2*Re{ a[k] * exp(+j*ω*k*dt)} # reconstruct narrowband component

α is tuned directly by a bandwidth parameter (Hz) or time constant.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional, Tuple

import numpy as np


def _alpha_from_bw_hz(dt: float, bw_hz: float) -> float:
    """
    Map a desired envelope bandwidth (Hz) into a stable 1st-order IIR alpha.

    We use the same "bilinear-ish" form you already use elsewhere:
        alpha = dt / (tau + dt),   tau = 1/(2π*bw)

    bw_hz ~ bandwidth of the envelope tracking (smaller = narrower band / higher Q).
    """
    dt = float(dt)
    bw_hz = float(bw_hz)
    if dt <= 0:
        raise ValueError("dt must be > 0")
    if bw_hz <= 0:
        raise ValueError("bw_hz must be > 0")

    tau = 1.0 / (2.0 * np.pi * bw_hz)
    return dt / (tau + dt)


@dataclass
class ModeState:
    """Convenience container for one mode's state for one axis."""
    freq_hz: float
    bw_hz: float
    env: complex          # complex envelope (≈ (A/2) * e^{j phase})
    phase: complex        # unit magnitude carrier exp(j*ωt)


class PhasorModeBankEstimator:
    """
    Multi-axis, multi-mode phasor modal estimator.

    Inputs:
      - y: measurement vector (n_axes,)
      - dt: optional step size; if omitted uses the dt passed at construction

    Outputs:
      - y_rigid: measurement with estimated modal components removed (n_axes,)
      - y_modes: estimated modal components per axis per mode (n_axes, n_modes)
      - env: complex envelopes per axis per mode (n_axes, n_modes)

    Notes:
      - For best results, y should contain the signal where the flex shows up
        (often gyro rate or attitude error).
      - The estimator is intentionally "model-light": it doesn't need accurate
        modal masses/gains; it only needs approximate modal frequencies.
    """

    def __init__(
        self,
        mode_freqs_hz: Iterable[float],
        mode_bandwidths_hz: Iterable[float],
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

        # Per mode parameters (cached for nominal dt)
        self._update_cached_coeffs(self.dt)

        # State: complex carrier phase and envelope per axis/mode
        self._carrier = np.ones((self.n_axes, self.n_modes), dtype=np.complex128)
        self._env = np.zeros((self.n_axes, self.n_modes), dtype=np.complex128)

        self._k = 0
        self._normalize_every = int(max(1, carrier_normalize_every))

    def _update_cached_coeffs(self, dt: float) -> None:
        """(Re)compute alpha and one-step carrier rotation for a given dt."""
        w = 2.0 * np.pi * self.mode_freqs_hz
        self._rot = np.exp(1j * w * float(dt))  # (n_modes,)
        self._alpha = np.array([_alpha_from_bw_hz(dt, bw) for bw in self.mode_bandwidths_hz], dtype=float)

    def reset(self) -> None:
        self._carrier[:] = 1.0 + 0j
        self._env[:] = 0.0 + 0j
        self._k = 0

    def step(
        self,
        y: np.ndarray,
        dt: Optional[float] = None
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Step the estimator forward one sample.

        Args:
            y: shape (n_axes,) array of the measured signal
            dt: optional dt override (if your sim dt isn't perfectly constant)

        Returns:
            (y_rigid, y_modes, env)
        """
        y = np.asarray(y, dtype=float).reshape(self.n_axes)

        # Optional dt update: still cheap, but avoid if dt constant.
        if dt is not None:
            dt = float(dt)
            if abs(dt - self.dt) > 1e-15:
                self.dt = dt
                self._update_cached_coeffs(self.dt)

        # Demodulate to DC: y * e^{ jωt}
        demod = y[:, None] * np.conj(self._carrier)  # (axes, modes) complex

        # Envelope LPF: env less than (1 α) env + α demod
        self._env = (1.0 - self._alpha)[None, :] * self._env + self._alpha[None, :] * demod

        # Reconstruct narrowband component: y_hat = 2*Re{ env * e^{jωt} }
        y_modes = 2.0 * np.real(self._env * self._carrier)  # (axes, modes)

        # Remove the estimated modes
        y_rigid = y - np.sum(y_modes, axis=1)

        # Advance carrier phase: e^{jω(t+dt)} = e^{jωt} * e^{jωdt}
        self._carrier *= self._rot[None, :]

        # Renormalize occasionally to control numerical drift
        self._k += 1
        if (self._k % self._normalize_every) == 0:
            mag = np.abs(self._carrier)
            mag[mag == 0] = 1.0
            self._carrier /= mag

        return y_rigid, y_modes, self._env

    def get_mode_amplitude_phase(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Return amplitude and phase estimates per axis/mode.

        If y ≈ A cos(ωt + φ), the envelope tends to env ≈ (A/2) e^{jφ}.

        Returns:
            amplitude: (axes, modes)  -> estimated sinusoid amplitude A
            phase:     (axes, modes)  -> phase φ in radians
        """
        amp = 2.0 * np.abs(self._env)
        ph = np.angle(self._env)
        return amp, ph
