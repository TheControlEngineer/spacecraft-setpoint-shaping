#!/usr/bin/env python3
"""
Temporary plot: flexible plant Bode (gain + wrapped phase).
Uses the same flexible plant construction as feedback_control.py.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
import matplotlib.pyplot as plt
from scipy import signal

# Ensure src/ is on sys.path so basilisk_sim imports work from any cwd
SCRIPT_DIR = Path(__file__).resolve().parent
SRC_DIR = SCRIPT_DIR.parent / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from basilisk_sim.feedback_control import _build_flexible_plant_tf
from basilisk_sim.spacecraft_properties import (
    HUB_INERTIA,
    compute_effective_inertia,
    compute_modal_gains,
)


def build_flexible_plant(axis: int, use_effective_inertia: bool) -> signal.TransferFunction:
    # Defaults mirror spacecraft_model.py
    modal_freqs_hz = [0.4, 1.3]
    modal_damping = [0.02, 0.015]

    inertia = compute_effective_inertia() if use_effective_inertia else HUB_INERTIA
    rotation_axis = np.zeros(3)
    rotation_axis[axis] = 1.0
    modal_gains = compute_modal_gains(inertia, rotation_axis)

    return _build_flexible_plant_tf(
        inertia,
        axis,
        modal_freqs_hz,
        modal_damping,
        modal_gains,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot flexible plant Bode (gain + wrapped phase)")
    parser.add_argument("--axis", type=int, default=2, choices=[0, 1, 2], help="Principal axis (0=x,1=y,2=z)")
    parser.add_argument("--fmin", type=float, default=0.05, help="Min frequency (Hz)")
    parser.add_argument("--fmax", type=float, default=5.0, help="Max frequency (Hz)")
    parser.add_argument("--n", type=int, default=300, help="Number of frequency points")
    parser.add_argument("--use-effective-inertia", action="store_true", help="Use effective inertia (with modal masses)")
    args = parser.parse_args()

    plant = build_flexible_plant(args.axis, args.use_effective_inertia)

    freqs_hz = np.logspace(np.log10(args.fmin), np.log10(args.fmax), args.n)
    w = 2.0 * np.pi * freqs_hz  # rad/s

    _, H = signal.freqresp(plant, w=w)
    mag_db = 20.0 * np.log10(np.maximum(np.abs(H), 1e-20))
    phase_deg = np.rad2deg(np.angle(H))  # wrapped to [-180, 180]

    fig, (ax_mag, ax_phase) = plt.subplots(2, 1, figsize=(8, 6), sharex=True)

    ax_mag.semilogx(freqs_hz, mag_db, "b")
    ax_mag.set_ylabel("Magnitude (dB)")
    ax_mag.set_title("Flexible Plant Bode")
    ax_mag.grid(True, which="both", alpha=0.3)

    ax_phase.semilogx(freqs_hz, phase_deg, "g")
    ax_phase.set_ylabel("Phase (deg)")
    ax_phase.set_xlabel("Frequency (Hz)")
    ax_phase.grid(True, which="both", alpha=0.3)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
