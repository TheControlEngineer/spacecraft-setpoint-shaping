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


def build_flexible_plant(axis: int, use_effective_inertia: bool) -> tuple[signal.TransferFunction, np.ndarray]:
    # Defaults mirror spacecraft_model.py
    modal_freqs_hz = [0.4, 1.3]
    modal_damping = [0.02, 0.015]

    inertia = compute_effective_inertia() if use_effective_inertia else HUB_INERTIA
    rotation_axis = np.zeros(3)
    rotation_axis[axis] = 1.0
    modal_gains = compute_modal_gains(inertia, rotation_axis)

    plant = _build_flexible_plant_tf(
        inertia,
        axis,
        modal_freqs_hz,
        modal_damping,
        modal_gains,
    )
    return plant, inertia


def build_pd_controller(K: float, P: float) -> signal.TransferFunction:
    """PD in sigma-domain: C(s) = K + 4*P*s."""
    num = [4.0 * P, K]
    den = [1.0]
    return signal.TransferFunction(num, den)


def build_filtered_pd_controller(K: float, P: float, cutoff_hz: float) -> signal.TransferFunction:
    """Filtered PD: C(s) = K + 4*P*s/(tau*s + 1)."""
    tau = 1.0 / (2.0 * np.pi * cutoff_hz)
    # K term plus filtered derivative term
    # C(s) = K + (4P s)/(tau s + 1) = (K*(tau s + 1) + 4P s)/(tau s + 1)
    num = [K * tau + 4.0 * P, K]
    den = [tau, 1.0]
    return signal.TransferFunction(num, den)


def build_notch_controller(K: float, P: float, notch_freqs_hz, notch_depth_db: float, notch_width: float) -> signal.TransferFunction:
    """PD with continuous-time notch filters (matching feedback_control.py transfer-function form)."""
    num = np.array([4.0 * P, K], dtype=float)
    den = np.array([1.0], dtype=float)

    depth_factor = 10 ** (-notch_depth_db / 20.0)
    zeta_z = depth_factor * 0.5
    zeta_p = notch_width

    for f_notch in notch_freqs_hz:
        omega_n = 2.0 * np.pi * f_notch
        notch_num = np.array([1.0, 2.0 * zeta_z * omega_n, omega_n**2], dtype=float)
        notch_den = np.array([1.0, 2.0 * zeta_p * omega_n, omega_n**2], dtype=float)
        num = np.convolve(num, notch_num)
        den = np.convolve(den, notch_den)

    return signal.TransferFunction(num, den)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot flexible plant Bode (gain + wrapped phase)")
    parser.add_argument("--axis", type=int, default=2, choices=[0, 1, 2], help="Principal axis (0=x,1=y,2=z)")
    parser.add_argument("--fmin", type=float, default=0.05, help="Min frequency (Hz)")
    parser.add_argument("--fmax", type=float, default=5.0, help="Max frequency (Hz)")
    parser.add_argument("--n", type=int, default=300, help="Number of frequency points")
    parser.add_argument("--bandwidth-hz", type=float, default=0.16,
                        help="Target bandwidth for gain design (Hz)")
    parser.add_argument("--zeta", type=float, default=0.90,
                        help="Desired damping ratio for gain design")
    parser.add_argument("--controller", choices=["standard_pd", "filtered_pd", "notch"], default="standard_pd",
                        help="Controller type for open-loop plot")
    parser.add_argument("--filter-cutoff-hz", type=float, default=None,
                        help="Filtered PD cutoff (Hz). If omitted, uses run_vizard_demo default.")
    parser.add_argument("--notch-freqs", type=str, default="0.4,1.3",
                        help="Comma-separated notch frequencies (Hz) for notch controller")
    parser.add_argument("--notch-depth-db", type=float, default=20.0,
                        help="Notch depth (dB)")
    parser.add_argument("--notch-width", type=float, default=0.3,
                        help="Notch width (relative damping)")
    parser.add_argument("--resonance-prominence-db", type=float, default=3.0,
                        help="Prominence (dB) threshold for auto resonance markers")
    parser.add_argument(
        "--use-hub-inertia",
        action="store_true",
        help="Use hub inertia only (default uses effective inertia with modal masses)",
    )
    args = parser.parse_args()

    use_effective_inertia = not args.use_hub_inertia
    plant, inertia = build_flexible_plant(args.axis, use_effective_inertia)

    # PD gains (same formulas as run_vizard_demo.py)
    I_axis = float(inertia[args.axis, args.axis])
    omega_n = 2.0 * np.pi * args.bandwidth_hz
    K = 4.0 * omega_n**2 * I_axis
    P = 2.0 * args.zeta * I_axis * omega_n
    print(f"PD gains: K={K:.3f} N*m, P={P:.3f} N*m*s")
    if args.controller == "filtered_pd":
        cutoff_hz = args.filter_cutoff_hz
        if cutoff_hz is None:
            # Match run_vizard_demo default for filtered PD
            cutoff_hz = 8.0
        controller = build_filtered_pd_controller(K, P, cutoff_hz)
    elif args.controller == "notch":
        notch_freqs = [float(s) for s in args.notch_freqs.split(",") if s.strip()]
        controller = build_notch_controller(K, P, notch_freqs, args.notch_depth_db, args.notch_width)
    else:
        controller = build_pd_controller(K, P)

    # Open-loop L(s) = G(s)*C(s)
    plant_num = np.atleast_1d(np.squeeze(plant.num))
    plant_den = np.atleast_1d(np.squeeze(plant.den))
    ctrl_num = np.atleast_1d(np.squeeze(controller.num))
    ctrl_den = np.atleast_1d(np.squeeze(controller.den))
    ol_num = np.convolve(plant_num, ctrl_num)
    ol_den = np.convolve(plant_den, ctrl_den)
    open_loop = signal.TransferFunction(ol_num, ol_den)

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

    # Auto-detect resonance peaks (plant) from magnitude (dB)
    peaks, _ = signal.find_peaks(mag_db, prominence=args.resonance_prominence_db)
    peak_freqs = freqs_hz[peaks] if len(peaks) > 0 else np.array([])
    if len(peak_freqs) >= 2:
        # Take the two highest peaks
        peak_mags = mag_db[peaks]
        top_idx = np.argsort(peak_mags)[-2:]
        resonance_freqs = np.sort(peak_freqs[top_idx])
    elif len(peak_freqs) == 1:
        resonance_freqs = np.array([peak_freqs[0]])
    else:
        resonance_freqs = np.array([0.4, 1.3])

    # Auto-detect antiresonance dips (plant) by finding peaks in -mag_db
    dips, _ = signal.find_peaks(-mag_db, prominence=args.resonance_prominence_db)
    dip_freqs = freqs_hz[dips] if len(dips) > 0 else np.array([])
    if len(dip_freqs) >= 2:
        dip_mags = mag_db[dips]
        # Take the two deepest dips
        top_idx = np.argsort(dip_mags)[:2]
        antires_freqs = np.sort(dip_freqs[top_idx])
    elif len(dip_freqs) == 1:
        antires_freqs = np.array([dip_freqs[0]])
    else:
        antires_freqs = np.array([])

    for f_mode in resonance_freqs:
        ax_mag.axvline(f_mode, color="r", linestyle="--", alpha=0.7)
        ax_phase.axvline(f_mode, color="r", linestyle="--", alpha=0.7)

    if len(resonance_freqs) > 0:
        print("Detected resonance frequencies (Hz):", ", ".join(f"{f:.3f}" for f in resonance_freqs))
    if len(antires_freqs) > 0:
        print("Detected antiresonance frequencies (Hz):", ", ".join(f"{f:.3f}" for f in antires_freqs))

    plt.tight_layout()

    # Open-loop plot
    _, L = signal.freqresp(open_loop, w=w)
    L_mag_db = 20.0 * np.log10(np.maximum(np.abs(L), 1e-20))
    L_phase_deg = np.rad2deg(np.angle(L))
    _, C_resp = signal.freqresp(controller, w=w)

    fig2, (ax_mag2, ax_phase2) = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
    ax_mag2.semilogx(freqs_hz, L_mag_db, "b")
    ax_mag2.set_ylabel("Magnitude (dB)")
    ax_mag2.set_title("Open-Loop L(s) Bode")
    ax_mag2.axhline(0.0, color="k", linestyle="--", linewidth=1.0, alpha=0.6)
    for f_mode in resonance_freqs:
        ax_mag2.axvline(f_mode, color="r", linestyle="--", alpha=0.7)
    ax_mag2.grid(True, which="both", alpha=0.3)

    ax_phase2.semilogx(freqs_hz, L_phase_deg, "g")
    ax_phase2.set_ylabel("Phase (deg)")
    ax_phase2.set_xlabel("Frequency (Hz)")
    for f_mode in resonance_freqs:
        ax_phase2.axvline(f_mode, color="r", linestyle="--", alpha=0.7)
    ax_phase2.grid(True, which="both", alpha=0.3)

    # Print open-loop magnitude/phase and controller damping tendency at detected resonances
    if len(resonance_freqs) > 0:
        for f_mode in resonance_freqs:
            idx = int(np.argmin(np.abs(freqs_hz - f_mode)))
            c_val = C_resp[idx]
            c_mag_db = 20.0 * np.log10(np.maximum(np.abs(c_val), 1e-20))
            c_phase_deg = np.rad2deg(np.angle(c_val))
            c_real = np.real(c_val)
            c_imag = np.imag(c_val)
            omega = 2.0 * np.pi * freqs_hz[idx]
            c_eff = c_imag / omega if omega > 0 else 0.0
            damping_flag = "DAMPING" if c_imag > 0 else "EXCITING"
            print(
                f"Open-loop at {freqs_hz[idx]:.3f} Hz: "
                f"|L|={L_mag_db[idx]:.1f} dB, phase={L_phase_deg[idx]:.1f} deg"
            )
            print(
                f"  Controller C(jw): |C|={c_mag_db:.1f} dB, phase={c_phase_deg:.1f} deg, "
                f"Re={c_real:.3e}, Im={c_imag:.3e} => c_eff=Im/ω={c_eff:.3e} ({damping_flag})"
            )

    plt.tight_layout()

    # Nyquist plot of open-loop L(s)
    fig3, ax_nyq = plt.subplots(figsize=(6.5, 6.0))
    ax_nyq.plot(np.real(L), np.imag(L), "b-", linewidth=1.2)
    ax_nyq.plot(np.real(L), -np.imag(L), "b--", linewidth=1.0, alpha=0.5)
    ax_nyq.plot([-1.0], [0.0], "ro", markersize=5)
    ax_nyq.axhline(0.0, color="#999999", linewidth=0.8)
    ax_nyq.axvline(0.0, color="#999999", linewidth=0.8)
    ax_nyq.set_title("Nyquist plot")
    ax_nyq.set_xlabel("Re")
    ax_nyq.set_ylabel("Im")
    ax_nyq.grid(True, alpha=0.3)

    # Stability margins (computed from frequency response)
    mag = np.abs(L)
    phase = np.unwrap(np.angle(L))
    phase_deg = np.rad2deg(phase)

    # Gain crossover (|L| = 1)
    pm = np.inf
    wg = None
    mag_db = 20.0 * np.log10(np.maximum(mag, 1e-20))
    idx_gc = np.where(np.diff(np.sign(mag_db)))[0]
    if len(idx_gc) > 0:
        i = idx_gc[0]
        m1, m2 = mag_db[i], mag_db[i + 1]
        w1, w2 = w[i], w[i + 1]
        if m2 != m1:
            w_gc = w1 + (w2 - w1) * (-m1) / (m2 - m1)
        else:
            w_gc = w1
        ph1, ph2 = phase_deg[i], phase_deg[i + 1]
        if w2 != w1:
            ph_gc = ph1 + (ph2 - ph1) * ((w_gc - w1) / (w2 - w1))
        else:
            ph_gc = ph1
        pm = 180.0 + ph_gc
        wg = w_gc

    # Phase crossover (phase = -180 deg)
    gm = np.inf
    wp = None
    phase_err = phase_deg + 180.0
    idx_pc = np.where(np.diff(np.sign(phase_err)))[0]
    if len(idx_pc) > 0:
        i = idx_pc[0]
        p1, p2 = phase_err[i], phase_err[i + 1]
        w1, w2 = w[i], w[i + 1]
        if p2 != p1:
            w_pc = w1 + (w2 - w1) * (-p1) / (p2 - p1)
        else:
            w_pc = w1
        m1, m2 = mag[i], mag[i + 1]
        if w2 != w1:
            m_pc = m1 + (m2 - m1) * ((w_pc - w1) / (w2 - w1))
        else:
            m_pc = m1
        gm = 1.0 / m_pc if m_pc > 0 else np.inf
        wp = w_pc

    gm_db = 20.0 * np.log10(gm) if np.isfinite(gm) else np.inf
    pm_text = f"{pm:.1f} deg" if np.isfinite(pm) else "inf"
    gm_text = f"{gm_db:.1f} dB" if np.isfinite(gm_db) else "inf"
    wg_hz = (wg / (2.0 * np.pi)) if wg is not None else None
    wp_hz = (wp / (2.0 * np.pi)) if wp is not None else None

    ax_nyq.text(
        0.02, 0.02,
        f"GM: {gm_text}\\nPM: {pm_text}",
        transform=ax_nyq.transAxes,
        fontsize=9,
        bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="#666666", alpha=0.8),
    )
    if wg_hz is not None or wp_hz is not None:
        print("Stability margins:")
        print(f"  GM: {gm_text} at wp={wp_hz:.3f} Hz" if wp_hz is not None else f"  GM: {gm_text}")
        print(f"  PM: {pm_text} at wg={wg_hz:.3f} Hz" if wg_hz is not None else f"  PM: {pm_text}")

    # Margin (Bode-style) plot with crossover markers
    fig4, (axm, axp) = plt.subplots(2, 1, figsize=(8.5, 6.0), sharex=True)
    axm.semilogx(freqs_hz, 20.0 * np.log10(np.maximum(np.abs(L), 1e-20)), "b")
    axm.axhline(0.0, color="#999999", linewidth=0.8)
    if wg is not None:
        axm.axvline(wg_hz, color="r", linestyle="--", alpha=0.7)
    if wp is not None:
        axm.axvline(wp_hz, color="g", linestyle="--", alpha=0.7)
    axm.set_ylabel("Magnitude (dB)")
    axm.set_title("Margin plot")
    axm.grid(True, which="both", alpha=0.3)

    axp.semilogx(freqs_hz, phase_deg, "g")
    axp.axhline(-180.0, color="#999999", linewidth=0.8)
    if wg is not None:
        axp.axvline(wg_hz, color="r", linestyle="--", alpha=0.7, label="Gain crossover")
    if wp is not None:
        axp.axvline(wp_hz, color="g", linestyle="--", alpha=0.7, label="Phase crossover")
    if wp is None:
        axp.text(
            0.02, 0.08,
            "No phase crossover in range\n(phase does not reach -180°)",
            transform=axp.transAxes,
            fontsize=8,
            bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="#666666", alpha=0.8),
        )
    axp.set_ylabel("Phase (deg)")
    axp.set_xlabel("Frequency (Hz)")
    axp.grid(True, which="both", alpha=0.3)
    axp.legend(loc="best", fontsize=8)

    plt.tight_layout()

    plt.show()


if __name__ == "__main__":
    main()
