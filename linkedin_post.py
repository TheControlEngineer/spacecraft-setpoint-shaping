#!/usr/bin/env python3
"""
linkedin_post.py — Educational Input Shaping Animations for LinkedIn

This script creates visually compelling animations that teach:
1. How convolution works (sliding window averaging)
2. How rectangular windows create spectral nulls (sinc function zeros)
3. How choosing window durations T=1/f nulls specific frequencies
4. The dramatic vibration reduction: shaped vs bang-bang commands

The key insight: Convolving with a rectangular window of duration T creates
a sinc function in the frequency domain with zeros at f = 1/T, 2/T, 3/T, ...

By convolving twice:
  - First with Tj = 1/f1 → nulls mode 1 frequency
  - Then with Ts = 1/f2 → nulls mode 2 frequency

Result: A smooth, jerk-and-snap-limited command that doesn't excite flexible modes!

Outputs:
  1) 1_convolution_explained.gif  — Step-by-step convolution visualization
  2) 2_spectral_evolution.gif     — How FFT changes through shaping stages
  3) 3_vibration_comparison.gif   — Dramatic unshaped vs shaped comparison
  4) 4_complete_story.gif         — The complete educational walkthrough
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Tuple, List

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.patches import Rectangle, FancyBboxPatch, Polygon, FancyArrowPatch
from matplotlib.collections import PolyCollection
import matplotlib.patheffects as path_effects


# =============================================================================
# VISUAL THEME
# =============================================================================

def apply_theme() -> None:
    plt.rcParams.update({
        "figure.dpi": 100,
        "savefig.dpi": 100,
        "font.family": "DejaVu Sans",
        "font.size": 11,
        "axes.titlesize": 13,
        "axes.titleweight": "bold",
        "axes.labelsize": 10,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 9,
        "axes.grid": True,
        "grid.alpha": 0.3,
        "grid.linewidth": 0.5,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.linewidth": 1.0,
        "figure.constrained_layout.use": True,
        "figure.facecolor": "white",
        "axes.facecolor": "#FAFAFA",
    })


# Color palette - high contrast for LinkedIn
COL = {
    "unshaped": "#DC3545",     # Bootstrap red
    "shaped": "#198754",       # Bootstrap green
    "base": "#343A40",         # Dark gray
    "window": "#0D6EFD",       # Bootstrap blue
    "stage1": "#FFC107",       # Bootstrap yellow
    "stage2": "#0DCAF0",       # Bootstrap cyan
    "mode1": "#6F42C1",        # Purple
    "mode2": "#FD7E14",        # Orange
    "soft": "#6C757D",         # Gray
    "highlight": "#E91E63",    # Pink
    "bg_light": "#F8F9FA",
    "text": "#212529",
}


# =============================================================================
# CORE SHAPING ALGORITHM
# =============================================================================

def generate_shaped_profile(
    total_time: float = 30.0,
    dt: float = 0.01,
    f1: float = 0.4,
    f2: float = 1.3,
) -> dict:
    """
    Generate bang-bang and shaped acceleration profiles.
    
    The shaped profile is created by convolving the bang-bang with two
    rectangular windows of duration T1=1/f1 and T2=1/f2 to null the
    flexible mode frequencies.
    
    Returns dict with all signals for visualization.
    """
    # Window durations for spectral nulls at f1 and f2
    T1 = 1.0 / f1  # Nulls f1, 2*f1, 3*f1, ...
    T2 = 1.0 / f2  # Nulls f2, 2*f2, 3*f2, ...
    
    # Window sample counts
    w1 = max(3, int(round(T1 / dt)))
    w2 = max(3, int(round(T2 / dt)))
    
    # Actual window durations
    T1_actual = w1 * dt
    T2_actual = w2 * dt
    
    # Base pulse duration accounting for convolution extension
    extension = (w1 - 1 + w2 - 1) * dt
    T_base = total_time - extension
    
    if T_base <= 0:
        raise ValueError("Total time too short for mode frequencies")
    
    # Bang-bang: +1 for first half, -1 for second half
    n_base = int(round(T_base / dt))
    bang_bang = np.zeros(n_base, dtype=float)
    n_half = n_base // 2
    bang_bang[:n_half] = 1.0
    bang_bang[n_half:] = -1.0
    
    # Rectangular windows (normalized)
    window1 = np.ones(w1, dtype=float) / w1
    window2 = np.ones(w2, dtype=float) / w2
    
    # Convolutions with mode='full' (critical for spectral nulling!)
    stage1 = np.convolve(bang_bang, window1, mode='full')
    stage2 = np.convolve(stage1, window2, mode='full')
    
    # Time vectors
    t_base = np.arange(len(bang_bang)) * dt
    t_stage1 = np.arange(len(stage1)) * dt
    t_final = np.arange(len(stage2)) * dt
    
    # Extend bang-bang for overlay plotting
    bang_bang_ext = np.zeros(len(stage2), dtype=float)
    bang_bang_ext[:len(bang_bang)] = bang_bang
    
    stage1_ext = np.zeros(len(stage2), dtype=float)
    stage1_ext[:len(stage1)] = stage1
    
    return {
        "dt": dt,
        "f1": f1,
        "f2": f2,
        "T1": T1_actual,
        "T2": T2_actual,
        "w1": w1,
        "w2": w2,
        "t_base": t_base,
        "t_stage1": t_stage1,
        "t_final": t_final,
        "bang_bang": bang_bang,
        "bang_bang_ext": bang_bang_ext,
        "stage1": stage1,
        "stage1_ext": stage1_ext,
        "shaped": stage2,
        "window1": window1,
        "window2": window2,
    }


# =============================================================================
# FLEXIBLE MODE SIMULATION
# =============================================================================

def simulate_mode(accel: np.ndarray, dt: float, f_mode: float, zeta: float) -> np.ndarray:
    """
    Simulate 2nd-order oscillator: ẍ + 2ζω·ẋ + ω²·x = accel
    Uses RK4 for accuracy.
    """
    w = 2.0 * np.pi * f_mode
    
    def deriv(state, a):
        x, v = state
        return np.array([v, a - 2*zeta*w*v - w*w*x])
    
    state = np.array([0.0, 0.0])
    out = np.zeros(len(accel))
    
    for i, a in enumerate(accel):
        out[i] = state[0]
        k1 = deriv(state, a)
        k2 = deriv(state + 0.5*dt*k1, a)
        k3 = deriv(state + 0.5*dt*k2, a)
        k4 = deriv(state + dt*k3, a)
        state = state + (dt/6)*(k1 + 2*k2 + 2*k3 + k4)
    
    return out


# =============================================================================
# FFT UTILITIES
# =============================================================================

def compute_fft(signal: np.ndarray, dt: float) -> Tuple[np.ndarray, np.ndarray]:
    """Compute one-sided FFT magnitude spectrum."""
    sig = signal - np.mean(signal)
    n = len(sig)
    window = np.hanning(n)
    S = np.fft.rfft(sig * window)
    f = np.fft.rfftfreq(n, dt)
    mag = np.abs(S)
    mag = mag / (np.max(mag) + 1e-12)  # Normalize
    return f, mag


def sinc_envelope(f: np.ndarray, T: float) -> np.ndarray:
    """Theoretical sinc envelope |sinc(f*T)|."""
    x = np.pi * f * T
    with np.errstate(divide='ignore', invalid='ignore'):
        s = np.abs(np.sin(x) / x)
        s[x == 0] = 1.0
    return s


# =============================================================================
# ANIMATION 1: CONVOLUTION EXPLAINED
# =============================================================================

def animate_convolution_explained(prof: dict, out_gif: str = "1_convolution_explained.gif",
                                   fps: int = 20, duration: float = 12.0) -> None:
    """
    Animate how convolution works step-by-step.
    Shows the sliding window, the values being averaged, and the output.
    """
    dt = prof["dt"]
    bang_bang = prof["bang_bang"]
    window1 = prof["window1"]
    w1 = prof["w1"]
    T1 = prof["T1"]
    f1 = prof["f1"]
    
    # For this demo, we'll show the first convolution in detail
    n_in = len(bang_bang)
    n_out = n_in + w1 - 1
    
    # Time for input
    t_in = np.arange(n_in) * dt
    t_out = np.arange(n_out) * dt
    
    # Pre-compute the full convolution
    output = np.convolve(bang_bang, window1, mode='full')
    
    # Animation parameters
    n_frames = int(duration * fps)
    output_indices = np.linspace(0, n_out - 1, n_frames).astype(int)
    
    fig = plt.figure(figsize=(12, 8))
    gs = fig.add_gridspec(3, 2, height_ratios=[1.2, 1.2, 1.0], width_ratios=[2, 1])
    
    ax_in = fig.add_subplot(gs[0, 0])
    ax_window = fig.add_subplot(gs[0, 1])
    ax_detail = fig.add_subplot(gs[1, 0])
    ax_out = fig.add_subplot(gs[2, :])
    
    fig.suptitle("How Convolution Works: Sliding Window Average", fontsize=16, fontweight='bold')
    
    # --- Input signal (static)
    ax_in.set_title("Input: Bang-Bang Acceleration")
    ax_in.set_xlabel("Time (s)")
    ax_in.set_ylabel("Acceleration")
    ax_in.plot(t_in, bang_bang, color=COL["base"], lw=2.5, label="Bang-bang")
    ax_in.axhline(0, color='gray', lw=0.5, alpha=0.5)
    ax_in.set_xlim(-0.5, t_in[-1] + 0.5)
    ax_in.set_ylim(-1.5, 1.5)
    
    # Sliding window highlight (will be updated)
    window_patch = ax_in.axvspan(0, T1, color=COL["window"], alpha=0.3, label=f"Window (T={T1:.2f}s)")
    ax_in.legend(loc="upper right")
    
    # --- Window shape (static)
    ax_window.set_title(f"Rectangular Window\n(Duration T = 1/f₁ = {T1:.2f}s)")
    t_win = np.arange(w1) * dt
    ax_window.fill_between(t_win, 0, window1 * w1, color=COL["window"], alpha=0.7)
    ax_window.plot(t_win, window1 * w1, color=COL["window"], lw=2)
    ax_window.set_xlabel("Time (s)")
    ax_window.set_ylabel("Weight")
    ax_window.set_xlim(-0.1, T1 + 0.1)
    ax_window.axhline(0, color='gray', lw=0.5)
    
    # Add annotation
    ax_window.text(T1/2, 0.5, f"Height = 1/{w1}\nArea = 1", ha='center', va='center',
                   fontsize=10, color='white', fontweight='bold')
    
    # --- Detail view: values being averaged
    ax_detail.set_title("Values Under Window (Being Averaged)")
    ax_detail.set_xlabel("Sample index in window")
    ax_detail.set_ylabel("Value")
    ax_detail.set_ylim(-1.5, 1.5)
    
    bars = ax_detail.bar(range(w1), np.zeros(w1), color=COL["window"], alpha=0.7, edgecolor='white')
    avg_line = ax_detail.axhline(0, color=COL["highlight"], lw=3, label="Average (output)")
    ax_detail.legend(loc="upper right")
    
    # --- Output (grows over time)
    ax_out.set_title("Output: Smoothed Signal (Convolution Result)")
    ax_out.set_xlabel("Time (s)")
    ax_out.set_ylabel("Acceleration")
    ax_out.plot(t_out, output, color=COL["stage1"], lw=1.5, alpha=0.3)  # Ghost of full output
    line_out, = ax_out.plot([], [], color=COL["stage1"], lw=2.5)
    dot_out, = ax_out.plot([], [], 'o', color=COL["highlight"], ms=10, zorder=5)
    ax_out.set_xlim(-0.5, t_out[-1] + 0.5)
    ax_out.set_ylim(-1.3, 1.3)
    ax_out.axhline(0, color='gray', lw=0.5, alpha=0.5)
    
    # Info text
    info_text = ax_out.text(0.02, 0.95, "", transform=ax_out.transAxes, fontsize=11,
                            verticalalignment='top', fontfamily='monospace',
                            bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))
    
    def init():
        line_out.set_data([], [])
        dot_out.set_data([], [])
        return line_out, dot_out
    
    def update(frame):
        k = output_indices[frame]
        
        # Window position in output space
        win_start_idx = k - w1 + 1
        win_end_idx = k
        
        # Get values under window (with padding for edges)
        values_under = []
        for i in range(win_start_idx, win_end_idx + 1):
            if 0 <= i < n_in:
                values_under.append(bang_bang[i])
            else:
                values_under.append(0.0)  # Zero padding
        
        # Update window position on input plot
        t_start = max(0, win_start_idx * dt)
        t_end = min(t_in[-1], (win_end_idx + 1) * dt)
        
        # Remove old patch and create new one
        for p in ax_in.patches:
            p.remove()
        ax_in.axvspan(t_start, t_end, color=COL["window"], alpha=0.3)
        
        # Update bar chart
        for bar, val in zip(bars, values_under):
            bar.set_height(val)
            bar.set_color(COL["window"] if val > 0 else COL["unshaped"])
        
        # Update average line
        avg_val = output[k]
        avg_line.set_ydata([avg_val, avg_val])
        
        # Update output line
        line_out.set_data(t_out[:k+1], output[:k+1])
        dot_out.set_data([t_out[k]], [output[k]])
        
        # Update info
        info_text.set_text(f"Output[{k}] = average of {w1} samples = {avg_val:.3f}")
        
        return line_out, dot_out, avg_line, info_text
    
    ani = FuncAnimation(fig, update, frames=n_frames, init_func=init, blit=False, interval=1000/fps)
    ani.save(out_gif, writer=PillowWriter(fps=fps))
    plt.close(fig)
    print(f"  ✓ Saved {out_gif}")


# =============================================================================
# ANIMATION 2: SPECTRAL EVOLUTION
# =============================================================================

def animate_spectral_evolution(prof: dict, out_gif: str = "2_spectral_evolution.gif",
                                fps: int = 20, duration: float = 15.0) -> None:
    """
    Show how the spectrum changes through each convolution stage.
    Demonstrates the sinc nulls appearing at mode frequencies.
    """
    dt = prof["dt"]
    f1, f2 = prof["f1"], prof["f2"]
    T1, T2 = prof["T1"], prof["T2"]
    
    bang_bang = prof["bang_bang"]
    stage1 = prof["stage1"]
    shaped = prof["shaped"]
    
    # Compute FFTs
    freq_bb, mag_bb = compute_fft(bang_bang, dt)
    freq_s1, mag_s1 = compute_fft(stage1, dt)
    freq_sh, mag_sh = compute_fft(shaped, dt)
    
    # Frequency limit
    fmax = 3.0
    mask_bb = freq_bb <= fmax
    mask_s1 = freq_s1 <= fmax
    mask_sh = freq_sh <= fmax
    
    # Theoretical sinc envelopes
    sinc1 = sinc_envelope(freq_sh[mask_sh], T1)
    sinc2 = sinc_envelope(freq_sh[mask_sh], T2)
    sinc_combined = sinc1 * sinc2
    
    fig = plt.figure(figsize=(14, 8))
    gs = fig.add_gridspec(2, 3, height_ratios=[1, 1.2])
    
    ax_t1 = fig.add_subplot(gs[0, 0])
    ax_t2 = fig.add_subplot(gs[0, 1])
    ax_t3 = fig.add_subplot(gs[0, 2])
    ax_spec = fig.add_subplot(gs[1, :])
    
    fig.suptitle("Spectral Nulling: How Convolution Creates Frequency Notches", 
                 fontsize=16, fontweight='bold')
    
    # Time domain plots (static)
    t_bb = np.arange(len(bang_bang)) * dt
    t_s1 = np.arange(len(stage1)) * dt
    t_sh = np.arange(len(shaped)) * dt
    
    ax_t1.set_title("Stage 0: Bang-Bang", color=COL["base"])
    ax_t1.plot(t_bb, bang_bang, color=COL["base"], lw=2)
    ax_t1.set_xlabel("Time (s)")
    ax_t1.set_ylabel("Accel")
    ax_t1.set_ylim(-1.5, 1.5)
    
    ax_t2.set_title(f"Stage 1: ⊛ rect(T₁={T1:.2f}s)", color=COL["stage1"])
    ax_t2.plot(t_s1, stage1, color=COL["stage1"], lw=2)
    ax_t2.set_xlabel("Time (s)")
    ax_t2.set_ylim(-1.2, 1.2)
    ax_t2.text(0.5, 0.95, f"Nulls f₁={f1} Hz", transform=ax_t2.transAxes, 
               ha='center', va='top', fontsize=10, color=COL["mode1"], fontweight='bold')
    
    ax_t3.set_title(f"Stage 2: ⊛ rect(T₂={T2:.2f}s)", color=COL["stage2"])
    ax_t3.plot(t_sh, shaped, color=COL["stage2"], lw=2)
    ax_t3.set_xlabel("Time (s)")
    ax_t3.set_ylim(-1.0, 1.0)
    ax_t3.text(0.5, 0.95, f"Nulls f₂={f2} Hz", transform=ax_t3.transAxes,
               ha='center', va='top', fontsize=10, color=COL["mode2"], fontweight='bold')
    
    # Spectrum plot
    ax_spec.set_title("Frequency Spectrum Evolution")
    ax_spec.set_xlabel("Frequency (Hz)")
    ax_spec.set_ylabel("Magnitude (normalized)")
    ax_spec.set_xlim(0, fmax)
    ax_spec.set_ylim(1e-4, 2)
    ax_spec.set_yscale('log')
    
    # Mode frequency markers
    for f, name, col in [(f1, f"f₁={f1}Hz", COL["mode1"]), (f2, f"f₂={f2}Hz", COL["mode2"])]:
        ax_spec.axvline(f, color=col, lw=2, ls='--', alpha=0.7)
        ax_spec.text(f, 1.5, name, ha='center', va='bottom', fontsize=11, 
                     color=col, fontweight='bold')
    
    # Lines for animation
    line_bb, = ax_spec.plot([], [], color=COL["base"], lw=2.5, label="Bang-bang", alpha=0.8)
    line_s1, = ax_spec.plot([], [], color=COL["stage1"], lw=2.5, label="After 1st conv", alpha=0.8)
    line_sh, = ax_spec.plot([], [], color=COL["shaped"], lw=2.5, label="After 2nd conv", alpha=0.8)
    
    # Theoretical envelopes (dotted)
    ax_spec.plot(freq_sh[mask_sh], sinc1, ':', color=COL["mode1"], lw=1.5, alpha=0.5)
    ax_spec.plot(freq_sh[mask_sh], sinc2, ':', color=COL["mode2"], lw=1.5, alpha=0.5)
    
    ax_spec.legend(loc='upper right')
    
    # Stage indicator
    stage_text = ax_spec.text(0.02, 0.95, "", transform=ax_spec.transAxes, fontsize=14,
                              fontweight='bold', va='top',
                              bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))
    
    # Animation phases
    n_frames = int(duration * fps)
    phase1_end = n_frames // 3
    phase2_end = 2 * n_frames // 3
    
    def update(frame):
        if frame < phase1_end:
            # Phase 1: Show bang-bang spectrum building up
            progress = frame / phase1_end
            n_show = max(1, int(progress * len(freq_bb[mask_bb])))
            line_bb.set_data(freq_bb[mask_bb][:n_show], mag_bb[mask_bb][:n_show] + 1e-4)
            line_s1.set_data([], [])
            line_sh.set_data([], [])
            stage_text.set_text("Stage 0: Bang-Bang\n(Sharp edges → broad spectrum)")
            stage_text.set_color(COL["base"])
            
        elif frame < phase2_end:
            # Phase 2: Show stage 1 spectrum appearing
            progress = (frame - phase1_end) / (phase2_end - phase1_end)
            n_show = max(1, int(progress * len(freq_s1[mask_s1])))
            line_bb.set_data(freq_bb[mask_bb], mag_bb[mask_bb] + 1e-4)
            line_bb.set_alpha(0.3)
            line_s1.set_data(freq_s1[mask_s1][:n_show], mag_s1[mask_s1][:n_show] + 1e-4)
            line_sh.set_data([], [])
            stage_text.set_text(f"Stage 1: Convolved with rect(T₁={T1:.2f}s)\n→ Creates NULL at f₁={f1} Hz!")
            stage_text.set_color(COL["stage1"])
            
        else:
            # Phase 3: Show final spectrum
            progress = (frame - phase2_end) / (n_frames - phase2_end)
            n_show = max(1, int(progress * len(freq_sh[mask_sh])))
            line_bb.set_data(freq_bb[mask_bb], mag_bb[mask_bb] + 1e-4)
            line_bb.set_alpha(0.2)
            line_s1.set_data(freq_s1[mask_s1], mag_s1[mask_s1] + 1e-4)
            line_s1.set_alpha(0.3)
            line_sh.set_data(freq_sh[mask_sh][:n_show], mag_sh[mask_sh][:n_show] + 1e-4)
            stage_text.set_text(f"Stage 2: Convolved with rect(T₂={T2:.2f}s)\n→ Creates NULL at f₂={f2} Hz!")
            stage_text.set_color(COL["shaped"])
        
        return line_bb, line_s1, line_sh, stage_text
    
    ani = FuncAnimation(fig, update, frames=n_frames, blit=False, interval=1000/fps)
    ani.save(out_gif, writer=PillowWriter(fps=fps))
    plt.close(fig)
    print(f"  ✓ Saved {out_gif}")


# =============================================================================
# ANIMATION 3: VIBRATION COMPARISON
# =============================================================================

def animate_vibration_comparison(prof: dict, out_gif: str = "3_vibration_comparison.gif",
                                  fps: int = 24, duration: float = 12.0) -> None:
    """
    Dramatic side-by-side comparison of unshaped vs shaped vibration response.
    Shows the flexible structure responding to both commands.
    """
    dt = prof["dt"]
    f1, f2 = prof["f1"], prof["f2"]
    
    bang_bang_ext = prof["bang_bang_ext"]
    shaped = prof["shaped"]
    t = prof["t_final"]
    n = len(t)
    
    # Simulate flexible modes
    zeta1, zeta2 = 0.015, 0.010  # Low damping for dramatic effect
    
    mode1_u = simulate_mode(bang_bang_ext, dt, f1, zeta1)
    mode2_u = simulate_mode(bang_bang_ext, dt, f2, zeta2)
    vib_unshaped = mode1_u + mode2_u
    
    mode1_s = simulate_mode(shaped, dt, f1, zeta1)
    mode2_s = simulate_mode(shaped, dt, f2, zeta2)
    vib_shaped = mode1_s + mode2_s
    
    # Normalize for comparison
    vib_max = max(np.max(np.abs(vib_unshaped)), np.max(np.abs(vib_shaped)))
    
    # Compute reduction metrics
    peak_u = np.max(np.abs(vib_unshaped))
    peak_s = np.max(np.abs(vib_shaped))
    rms_u = np.sqrt(np.mean(vib_unshaped**2))
    rms_s = np.sqrt(np.mean(vib_shaped**2))
    reduction_peak = (1 - peak_s/peak_u) * 100
    reduction_rms = (1 - rms_s/rms_u) * 100
    
    fig = plt.figure(figsize=(14, 10))
    gs = fig.add_gridspec(3, 2, height_ratios=[0.7, 1.0, 1.2])
    
    ax_cmd_u = fig.add_subplot(gs[0, 0])
    ax_cmd_s = fig.add_subplot(gs[0, 1])
    ax_struct_u = fig.add_subplot(gs[1, 0])
    ax_struct_s = fig.add_subplot(gs[1, 1])
    ax_vib = fig.add_subplot(gs[2, :])
    
    fig.suptitle("Vibration Suppression: Bang-Bang vs Shaped Command", 
                 fontsize=16, fontweight='bold')
    
    # Command plots
    ax_cmd_u.set_title("UNSHAPED (Bang-Bang)", color=COL["unshaped"], fontsize=12)
    ax_cmd_u.plot(t, bang_bang_ext, color=COL["unshaped"], lw=2)
    ax_cmd_u.set_ylabel("Command")
    ax_cmd_u.set_xlim(0, t[-1])
    
    ax_cmd_s.set_title("SHAPED (Spectral Nulling)", color=COL["shaped"], fontsize=12)
    ax_cmd_s.plot(t, shaped, color=COL["shaped"], lw=2)
    ax_cmd_s.set_xlim(0, t[-1])
    
    # Structure visualization (animated)
    for ax, title, col in [(ax_struct_u, "Flexible Structure", COL["unshaped"]),
                            (ax_struct_s, "Flexible Structure", COL["shaped"])]:
        ax.set_aspect('equal')
        ax.set_xlim(-0.5, 3.5)
        ax.set_ylim(-1.5, 1.5)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_facecolor('#F0F0F0')
    
    # Vibration comparison plot
    ax_vib.set_title("Flexible Mode Vibration Response")
    ax_vib.set_xlabel("Time (s)")
    ax_vib.set_ylabel("Vibration Amplitude")
    ax_vib.set_xlim(0, t[-1])
    ax_vib.set_ylim(-vib_max * 1.2, vib_max * 1.2)
    ax_vib.axhline(0, color='gray', lw=0.5, alpha=0.5)
    
    # Ghost lines
    ax_vib.plot(t, vib_unshaped, color=COL["unshaped"], lw=1, alpha=0.2)
    ax_vib.plot(t, vib_shaped, color=COL["shaped"], lw=1, alpha=0.2)
    
    line_vib_u, = ax_vib.plot([], [], color=COL["unshaped"], lw=2.5, label="Unshaped")
    line_vib_s, = ax_vib.plot([], [], color=COL["shaped"], lw=2.5, label="Shaped")
    ax_vib.legend(loc='upper right')
    
    # Metrics box
    metrics_text = ax_vib.text(0.02, 0.95, "", transform=ax_vib.transAxes, fontsize=11,
                               va='top', fontfamily='monospace',
                               bbox=dict(boxstyle='round', facecolor='white', alpha=0.95))
    
    # Animation
    n_frames = int(duration * fps)
    indices = np.linspace(0, n-1, n_frames).astype(int)
    
    def draw_beam(ax, deflection, color, exaggeration=50):
        """Draw a cantilevered beam with bending."""
        ax.clear()
        ax.set_xlim(-0.5, 3.5)
        ax.set_ylim(-1.5, 1.5)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_facecolor('#F0F0F0')
        
        # Fixed wall
        ax.fill_between([-0.5, 0], [-1.5, -1.5], [1.5, 1.5], color='gray', alpha=0.3)
        ax.axvline(0, color='gray', lw=3)
        
        # Beam with bending
        x_beam = np.linspace(0, 3, 50)
        # Cantilever bending shape: (x/L)^2 
        bend_shape = (x_beam / 3) ** 2
        y_beam = deflection * exaggeration * bend_shape
        
        # Draw beam as thick line
        ax.plot(x_beam, y_beam, color=color, lw=8, solid_capstyle='round')
        
        # Tip mass
        ax.plot(x_beam[-1], y_beam[-1], 'o', color=color, ms=20)
        
        # Deflection annotation
        ax.annotate('', xy=(3.2, y_beam[-1]), xytext=(3.2, 0),
                    arrowprops=dict(arrowstyle='<->', color=color, lw=2))
        ax.text(3.4, y_beam[-1]/2, f'{abs(deflection):.3f}', fontsize=9, va='center', color=color)
    
    def update(frame):
        k = indices[frame]
        
        # Update vibration lines
        line_vib_u.set_data(t[:k+1], vib_unshaped[:k+1])
        line_vib_s.set_data(t[:k+1], vib_shaped[:k+1])
        
        # Update structure visualization
        draw_beam(ax_struct_u, vib_unshaped[k], COL["unshaped"])
        draw_beam(ax_struct_s, vib_shaped[k], COL["shaped"])
        
        # Update metrics
        current_peak_u = np.max(np.abs(vib_unshaped[:k+1])) if k > 0 else 0
        current_peak_s = np.max(np.abs(vib_shaped[:k+1])) if k > 0 else 0
        
        metrics_text.set_text(
            f"Time: {t[k]:.1f}s\n"
            f"Peak (unshaped): {current_peak_u:.4f}\n"
            f"Peak (shaped):   {current_peak_s:.4f}\n"
            f"────────────────────\n"
            f"Final reduction: {reduction_peak:.1f}% (peak)\n"
            f"                 {reduction_rms:.1f}% (RMS)"
        )
        
        return line_vib_u, line_vib_s, metrics_text
    
    ani = FuncAnimation(fig, update, frames=n_frames, blit=False, interval=1000/fps)
    ani.save(out_gif, writer=PillowWriter(fps=fps))
    plt.close(fig)
    print(f"  ✓ Saved {out_gif}")


# =============================================================================
# ANIMATION 4: COMPLETE STORY
# =============================================================================

def animate_complete_story(prof: dict, out_gif: str = "4_complete_story.gif",
                           fps: int = 20, duration: float = 20.0) -> None:
    """
    The complete educational story: convolution → spectral nulls → vibration suppression.
    """
    dt = prof["dt"]
    f1, f2 = prof["f1"], prof["f2"]
    T1, T2 = prof["T1"], prof["T2"]
    
    bang_bang = prof["bang_bang"]
    bang_bang_ext = prof["bang_bang_ext"]
    stage1 = prof["stage1"]
    shaped = prof["shaped"]
    t = prof["t_final"]
    n = len(t)
    
    # Simulate vibrations
    zeta1, zeta2 = 0.015, 0.010
    vib_u = simulate_mode(bang_bang_ext, dt, f1, zeta1) + simulate_mode(bang_bang_ext, dt, f2, zeta2)
    vib_s = simulate_mode(shaped, dt, f1, zeta1) + simulate_mode(shaped, dt, f2, zeta2)
    
    # FFTs
    freq_bb, mag_bb = compute_fft(bang_bang, dt)
    freq_sh, mag_sh = compute_fft(shaped, dt)
    fmax = 3.0
    mask_bb = freq_bb <= fmax
    mask_sh = freq_sh <= fmax
    
    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(3, 3, height_ratios=[1, 1, 1.2])
    
    # Row 1: The shaping process
    ax_base = fig.add_subplot(gs[0, 0])
    ax_s1 = fig.add_subplot(gs[0, 1])
    ax_final = fig.add_subplot(gs[0, 2])
    
    # Row 2: FFT comparison
    ax_fft = fig.add_subplot(gs[1, :])
    
    # Row 3: Vibration comparison
    ax_vib = fig.add_subplot(gs[2, :])
    
    fig.suptitle("Input Shaping: The Complete Picture", fontsize=18, fontweight='bold', y=0.98)
    
    # Time vectors
    t_bb = np.arange(len(bang_bang)) * dt
    t_s1 = np.arange(len(stage1)) * dt
    
    # Row 1: Shaping stages
    ax_base.set_title("① Bang-Bang Command", fontsize=11)
    ax_base.plot(t_bb, bang_bang, color=COL["base"], lw=2)
    ax_base.set_xlabel("Time (s)")
    ax_base.set_ylabel("Accel")
    ax_base.text(0.5, 0.02, "Sharp edges excite\nall frequencies!", 
                 transform=ax_base.transAxes, ha='center', va='bottom',
                 fontsize=9, color=COL["unshaped"], style='italic')
    
    ax_s1.set_title(f"② After 1st Convolution\n(T₁={T1:.2f}s → nulls {f1}Hz)", fontsize=11)
    ax_s1.plot(t_s1, stage1, color=COL["stage1"], lw=2)
    ax_s1.set_xlabel("Time (s)")
    ax_s1.annotate('', xy=(T1/2, 0.9), xytext=(T1/2, 0.5),
                   arrowprops=dict(arrowstyle='->', color=COL["window"], lw=2))
    ax_s1.text(T1/2, 0.95, 'Jerk-limited\nramps', ha='center', va='bottom', fontsize=8)
    
    ax_final.set_title(f"③ After 2nd Convolution\n(T₂={T2:.2f}s → nulls {f2}Hz)", fontsize=11)
    ax_final.plot(t, shaped, color=COL["shaped"], lw=2)
    ax_final.set_xlabel("Time (s)")
    ax_final.text(0.5, 0.02, "Smooth: bounded\njerk AND snap!", 
                  transform=ax_final.transAxes, ha='center', va='bottom',
                  fontsize=9, color=COL["shaped"], style='italic')
    
    # Row 2: FFT
    ax_fft.set_title("Frequency Content: Bang-Bang vs Shaped", fontsize=12)
    ax_fft.set_xlabel("Frequency (Hz)")
    ax_fft.set_ylabel("Magnitude")
    ax_fft.set_xlim(0, fmax)
    ax_fft.set_ylim(1e-4, 2)
    ax_fft.set_yscale('log')
    
    # Mode markers
    for f, name, col in [(f1, f"Mode 1\n{f1} Hz", COL["mode1"]), 
                          (f2, f"Mode 2\n{f2} Hz", COL["mode2"])]:
        ax_fft.axvline(f, color=col, lw=2, ls='--', alpha=0.8)
        ax_fft.axvspan(f-0.05, f+0.05, color=col, alpha=0.15)
        ax_fft.text(f, 1.3, name, ha='center', fontsize=10, color=col, fontweight='bold')
    
    line_fft_bb, = ax_fft.plot([], [], color=COL["unshaped"], lw=2.5, label="Bang-bang")
    line_fft_sh, = ax_fft.plot([], [], color=COL["shaped"], lw=2.5, label="Shaped")
    ax_fft.legend(loc='upper right')
    
    # Row 3: Vibration
    ax_vib.set_title("Resulting Vibration (Sum of Both Modes)", fontsize=12)
    ax_vib.set_xlabel("Time (s)")
    ax_vib.set_ylabel("Vibration")
    ax_vib.set_xlim(0, t[-1])
    vib_max = max(np.max(np.abs(vib_u)), np.max(np.abs(vib_s)))
    ax_vib.set_ylim(-vib_max*1.3, vib_max*1.3)
    ax_vib.axhline(0, color='gray', lw=0.5, alpha=0.5)
    
    # Ghost
    ax_vib.plot(t, vib_u, color=COL["unshaped"], lw=1, alpha=0.15)
    ax_vib.plot(t, vib_s, color=COL["shaped"], lw=1, alpha=0.15)
    
    line_vib_u, = ax_vib.plot([], [], color=COL["unshaped"], lw=2.5, label="Unshaped")
    line_vib_s, = ax_vib.plot([], [], color=COL["shaped"], lw=2.5, label="Shaped")
    ax_vib.legend(loc='upper right')
    
    # Results box
    reduction = (1 - np.max(np.abs(vib_s))/np.max(np.abs(vib_u))) * 100
    result_box = ax_vib.text(0.02, 0.95, "", transform=ax_vib.transAxes,
                             fontsize=12, va='top', fontweight='bold',
                             bbox=dict(boxstyle='round', facecolor='#E8F5E9', 
                                      edgecolor=COL["shaped"], alpha=0.95))
    
    # Animation phases
    n_frames = int(duration * fps)
    phase1 = n_frames // 4      # Show FFTs
    phase2 = n_frames // 2      # Build vibration
    
    def update(frame):
        if frame < phase1:
            # Phase 1: Build FFT display
            progress = frame / phase1
            n_bb = max(1, int(progress * len(freq_bb[mask_bb])))
            n_sh = max(1, int(progress * len(freq_sh[mask_sh])))
            
            line_fft_bb.set_data(freq_bb[mask_bb][:n_bb], mag_bb[mask_bb][:n_bb] + 1e-4)
            line_fft_sh.set_data(freq_sh[mask_sh][:n_sh], mag_sh[mask_sh][:n_sh] + 1e-4)
            line_vib_u.set_data([], [])
            line_vib_s.set_data([], [])
            result_box.set_text("Building spectra...")
            
        else:
            # Phase 2+: Show vibration building
            line_fft_bb.set_data(freq_bb[mask_bb], mag_bb[mask_bb] + 1e-4)
            line_fft_sh.set_data(freq_sh[mask_sh], mag_sh[mask_sh] + 1e-4)
            
            vib_progress = (frame - phase1) / (n_frames - phase1)
            k = min(int(vib_progress * n), n-1)
            
            line_vib_u.set_data(t[:k+1], vib_u[:k+1])
            line_vib_s.set_data(t[:k+1], vib_s[:k+1])
            
            if k > 10:
                cur_pk_u = np.max(np.abs(vib_u[:k+1]))
                cur_pk_s = np.max(np.abs(vib_s[:k+1]))
                cur_red = (1 - cur_pk_s/cur_pk_u) * 100 if cur_pk_u > 0 else 0
                result_box.set_text(f"Vibration Reduction:\n{cur_red:.0f}%")
            else:
                result_box.set_text("Simulating response...")
        
        return line_fft_bb, line_fft_sh, line_vib_u, line_vib_s, result_box
    
    ani = FuncAnimation(fig, update, frames=n_frames, blit=False, interval=1000/fps)
    ani.save(out_gif, writer=PillowWriter(fps=fps))
    plt.close(fig)
    print(f"  ✓ Saved {out_gif}")


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("=" * 60)
    print("Input Shaping Educational Animations")
    print("=" * 60)
    
    apply_theme()
    
    # Parameters
    DT = 0.01        # Fine time step for smooth animation
    TOTAL_TIME = 25.0
    F1 = 0.4         # Mode 1: 0.4 Hz (T1 = 2.5s)
    F2 = 1.3         # Mode 2: 1.3 Hz (T2 ≈ 0.77s)
    
    print(f"\nParameters:")
    print(f"  Mode 1: f₁ = {F1} Hz  →  Window T₁ = {1/F1:.3f} s")
    print(f"  Mode 2: f₂ = {F2} Hz  →  Window T₂ = {1/F2:.3f} s")
    print(f"  Total maneuver time: {TOTAL_TIME} s")
    
    # Generate profiles
    print("\nGenerating shaped profile...")
    prof = generate_shaped_profile(
        total_time=TOTAL_TIME,
        dt=DT,
        f1=F1,
        f2=F2,
    )
    
    print(f"\nCreating animations...")
    
    # Animation 1: Convolution explained
    animate_convolution_explained(
        prof,
        out_gif="1_convolution_explained.gif",
        fps=20,
        duration=12.0
    )
    
    # Animation 2: Spectral evolution
    animate_spectral_evolution(
        prof,
        out_gif="2_spectral_evolution.gif",
        fps=20,
        duration=15.0
    )
    
    # Animation 3: Vibration comparison
    animate_vibration_comparison(
        prof,
        out_gif="3_vibration_comparison.gif",
        fps=24,
        duration=12.0
    )
    
    # Animation 4: Complete story
    animate_complete_story(
        prof,
        out_gif="4_complete_story.gif",
        fps=20,
        duration=20.0
    )
    
    print("\n" + "=" * 60)
    print("Done! Created 4 GIF animations:")
    print("  1_convolution_explained.gif  - How sliding window averaging works")
    print("  2_spectral_evolution.gif     - How nulls appear at mode frequencies")
    print("  3_vibration_comparison.gif   - Dramatic unshaped vs shaped comparison")
    print("  4_complete_story.gif         - The complete educational walkthrough")
    print("=" * 60)


if __name__ == "__main__":
    main()
