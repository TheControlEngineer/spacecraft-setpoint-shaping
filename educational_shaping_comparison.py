"""
Educational Input Shaping Visualization.

Creates visual explanations of ZVD and fourth-order input shaping methods
for a 180 deg pitch (Y-axis) rotation maneuver.

Context:
- Pitch rotation excites solar array flex modes (0.4 Hz, 1.3 Hz).
- Step-and-stare imaging: slew -> wait for settling -> image.
- Goal: minimize post-slew settling time for faster imaging.
"""

from __future__ import annotations

import os
import sys

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import numpy as np
from scipy.fft import fft, fftfreq
repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
src_root = os.path.join(repo_root, "src")
for path in (repo_root, src_root):
    if path not in sys.path:
        sys.path.insert(0, path)

from input_shaping import ZVD, convolve_shapers
from feedforward_control import compute_bang_bang_trajectory

# Professional style
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.size': 10,
    'axes.titlesize': 11,
    'axes.labelsize': 10,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 9,
    'axes.grid': True,
    'grid.alpha': 0.25,
    'grid.linewidth': 0.5,
    'axes.axisbelow': True,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'axes.linewidth': 0.8,
})

# Parameters (same as mission_simulation.py).
# Maneuver: 180 deg pitch rotation to excite flexible modes.
THETA_FINAL = np.radians(180.0)  # Target rotation angle [rad].
DURATION = 30.0  # Slew duration [s].
DT = 0.001  # Sample time [s].
MODE_FREQUENCIES = [0.4, 1.3]  # Flexible mode frequencies [Hz].
DAMPING_RATIOS = [0.02, 0.015]  # Modal damping ratios [-].
I_AXIS = 1125.0  # Effective inertia about the slew axis [kg*m^2].
MAX_TORQUE = 70.0  # Torque limit used in conceptual sizing [Nm].

# Color palette for plot consistency.
C_BLUE = '#2c7bb6'
C_RED = '#d7191c'
C_GREEN = '#1a9641'
C_PURPLE = '#756bb1'
C_ORANGE = '#fd8d3c'
C_GRAY = '#636363'


def create_base_torque_profile():
    """
    Build the base bang-bang torque profile for the reference maneuver.

    Inputs: none (uses module-level constants).
    Outputs: time array, torque array, and acceleration profile.
    Process: uses compute_bang_bang_trajectory and multiplies by I_AXIS.
    """
    t, theta, omega, alpha = compute_bang_bang_trajectory(THETA_FINAL, DURATION, DT)
    return t, I_AXIS * alpha, alpha


def compute_psd(signal_data, dt):
    """
    Compute the single-sided PSD of a time-series signal.

    Inputs:
    - signal_data: array of samples.
    - dt: sample time [s].

    Output:
    - (freq, psd) for positive frequencies only.
    """
    n = len(signal_data)
    fft_data = fft(signal_data)
    freq = fftfreq(n, dt)
    psd = np.abs(fft_data)**2 / n
    pos_idx = freq > 0
    return freq[pos_idx], psd[pos_idx]


def residual_vibration(omega_n, zeta, A, t):
    """
    Evaluate the ZVD residual vibration formula for a shaper.

    Inputs:
    - omega_n: modal natural frequency [rad/s].
    - zeta: modal damping ratio.
    - A: shaper impulse amplitudes.
    - t: shaper impulse times [s].

    Output:
    - Residual vibration magnitude (unitless).
    """
    omega_d = omega_n * np.sqrt(1 - zeta**2)
    V = sum(a * np.exp(-zeta * omega_n * ti) * np.exp(1j * omega_d * ti) 
            for a, ti in zip(A, t))
    return np.abs(V)


def apply_shaper(signal, A, t_shaper, dt):
    """
    Apply an input shaper to a signal using discrete-time convolution.

    Inputs:
    - signal: base signal samples.
    - A: shaper impulse amplitudes.
    - t_shaper: shaper impulse times [s].
    - dt: sample time [s].

    Outputs:
    - (t_shaped, signal_shaped).
    """
    n_shaped = len(signal) + int(t_shaper[-1] / dt)
    signal_shaped = np.zeros(n_shaped)
    for amp, t_imp in zip(A, t_shaper):
        shift = int(round(t_imp / dt))
        for i in range(len(signal)):
            if i + shift < n_shaped:
                signal_shaped[i + shift] += amp * signal[i]
    return np.arange(n_shaped) * dt, signal_shaped


def create_zvd_educational_plot():
    """
    Generate a 6-panel ZVD shaping explanation figure for a pitch rotation.

    Output:
    - Saves 'educational_zvd_shaping.png'.
    """
    print("Generating ZVD plot (PITCH rotation context)...")
    
    t_base, torque_base, _ = create_base_torque_profile()
    
    # ZVD shapers
    omega1, omega2 = 2*np.pi*MODE_FREQUENCIES[0], 2*np.pi*MODE_FREQUENCIES[1]
    zeta1, zeta2 = DAMPING_RATIOS
    A1, t1, _ = ZVD(omega1, zeta1)
    A2, t2, _ = ZVD(omega2, zeta2)
    A_casc, t_casc = convolve_shapers((A1, t1), (A2, t2))
    
    t_shaped, torque_shaped = apply_shaper(torque_base, A_casc, t_casc, DT)
    
    # Frequency analysis
    freq_range = np.linspace(0.1, 10.0, 500)
    rv_casc = np.array([residual_vibration(2*np.pi*f, 0.02, A_casc, t_casc) for f in freq_range])
    
    freq_base, psd_base = compute_psd(torque_base, DT)
    freq_shp, psd_shp = compute_psd(torque_shaped[:len(torque_base)], DT)
    psd_base_n = psd_base / np.max(psd_base)
    psd_shp_n = psd_shp / np.max(psd_base)
    
    # ========================================================================
    # FIGURE
    # ========================================================================
    fig = plt.figure(figsize=(14, 8))
    gs = gridspec.GridSpec(2, 3, figure=fig, 
                           height_ratios=[1, 1],
                           width_ratios=[1, 1, 1],
                           hspace=0.35, wspace=0.35, 
                           left=0.08, right=0.96, top=0.94, bottom=0.08)
    
    # (a) Mode 1 Shaper
    ax1 = fig.add_subplot(gs[0, 0])
    ml, sl, _ = ax1.stem(t1, A1, basefmt=' ', linefmt=C_BLUE, markerfmt='o')
    plt.setp(sl, linewidth=2.5)
    plt.setp(ml, markersize=8, markerfacecolor=C_BLUE, markeredgecolor=C_BLUE)
    ax1.set_ylabel('Amplitude', fontsize=11)
    ax1.set_xlabel('Time (s)', fontsize=11)
    ax1.set_title('(a) Mode 1 ZVD Shaper', fontsize=11, fontweight='bold', pad=8)
    ax1.set_ylim([0, max(A1)*1.25])
    ax1.text(0.02, 0.96, f'f = {MODE_FREQUENCIES[0]} Hz\nn = 3 impulses\n(PITCH excites this mode)', 
             transform=ax1.transAxes, va='top', ha='left', fontsize=9, 
             bbox=dict(boxstyle='round,pad=0.4', facecolor='white', edgecolor='none', alpha=0.8))
    
    # (b) Mode 2 Shaper
    ax2 = fig.add_subplot(gs[0, 1])
    ml, sl, _ = ax2.stem(t2, A2, basefmt=' ', linefmt=C_PURPLE, markerfmt='o')
    plt.setp(sl, linewidth=2.5)
    plt.setp(ml, markersize=8, markerfacecolor=C_PURPLE, markeredgecolor=C_PURPLE)
    ax2.set_ylabel('Amplitude', fontsize=11)
    ax2.set_xlabel('Time (s)', fontsize=11)
    ax2.set_title('(b) Mode 2 ZVD Shaper', fontsize=11, fontweight='bold', pad=8)
    ax2.set_ylim([0, max(A2)*1.25])
    ax2.text(0.02, 0.96, f'f = {MODE_FREQUENCIES[1]} Hz\nn = 3 impulses\n(PITCH excites this mode)', 
             transform=ax2.transAxes, va='top', ha='left', fontsize=9, 
             bbox=dict(boxstyle='round,pad=0.4', facecolor='white', edgecolor='none', alpha=0.8))
    
    # (c) Cascaded Shaper
    ax3 = fig.add_subplot(gs[0, 2])
    ml, sl, _ = ax3.stem(t_casc, A_casc, basefmt=' ', linefmt=C_GREEN, markerfmt='o')
    plt.setp(sl, linewidth=2)
    plt.setp(ml, markersize=6, markerfacecolor=C_GREEN, markeredgecolor=C_GREEN)
    ax3.set_ylabel('Amplitude', fontsize=11)
    ax3.set_xlabel('Time (s)', fontsize=11)
    ax3.set_title('(c) Cascaded Shaper (Multi-Mode)', fontsize=11, fontweight='bold', pad=8)
    ax3.set_ylim([0, max(A_casc)*1.25])
    ax3.text(0.02, 0.96, f'(a) * (b)\nn = 9 impulses', 
             transform=ax3.transAxes, va='top', ha='left', fontsize=9, 
             bbox=dict(boxstyle='round,pad=0.4', facecolor='white', edgecolor='none', alpha=0.8))
    
    # (d) Original Torque
    ax4 = fig.add_subplot(gs[1, 0])
    ax4.plot(t_base, torque_base * 1000, color=C_RED, linewidth=2)
    ax4.axhline(y=0, color='gray', linewidth=0.8, linestyle='--', alpha=0.4)
    ax4.set_ylabel('Torque (mNm)', fontsize=11)
    ax4.set_xlabel('Time (s)', fontsize=11)
    ax4.set_title('(d) Bang-Bang PITCH Command', fontsize=11, fontweight='bold', pad=8)
    ax4.set_xlim([0, DURATION])
    
    # (e) Shaped Torque
    ax5 = fig.add_subplot(gs[1, 1])
    ax5.plot(t_shaped, torque_shaped * 1000, color=C_GREEN, linewidth=2)
    ax5.axhline(y=0, color='gray', linewidth=0.8, linestyle='--', alpha=0.4)
    ax5.set_ylabel('Torque (mNm)', fontsize=11)
    ax5.set_xlabel('Time (s)', fontsize=11)
    ax5.set_title('(e) ZVD-Shaped PITCH Command = (d) * (c)', fontsize=11, fontweight='bold', pad=8)
    ax5.set_xlim([0, t_shaped[-1]])
    
    # (f) Sensitivity
    ax6 = fig.add_subplot(gs[1, 2])
    ax6.plot(freq_range, rv_casc * 100, color=C_GREEN, linewidth=2.5)
    ax6.axvline(x=MODE_FREQUENCIES[0], color=C_BLUE, linestyle='--', lw=2, alpha=0.6)
    ax6.axvline(x=MODE_FREQUENCIES[1], color=C_PURPLE, linestyle='--', lw=2, alpha=0.6)
    ax6.fill_between(freq_range, 0, 5, alpha=0.15, color=C_GREEN)
    ax6.set_xlabel('Frequency (Hz)', fontsize=11)
    ax6.set_ylabel('Residual Vibration (%)', fontsize=11)
    ax6.set_title('(f) Sensitivity Curve', fontsize=11, fontweight='bold', pad=8)
    ax6.set_xlim([0.1, 10.0])
    #ax6.set_ylim([0, 100])
    ax6.text(MODE_FREQUENCIES[0], 5, 'f1', fontsize=10, ha='center', color=C_BLUE, fontweight='bold')
    ax6.text(MODE_FREQUENCIES[1], 5, 'f2', fontsize=10, ha='center', color=C_PURPLE, fontweight='bold')
    
    plt.savefig('educational_zvd_shaping.png', dpi=250, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    print("Saved: educational_zvd_shaping.png")
    plt.close()


def create_fourth_order_educational_plot():
    """
    Generate a 6-panel fourth-order shaping explanation figure for pitch rotation.
    
    Fourth-order shaping uses convolution with rectangular windows to create
    spectral nulls at the modal frequencies. This is a CONCEPTUAL visualization
    showing the frequency-domain approach (vs ZVD's time-domain approach).
    
    Note: The actual implementation in design_shaper.py uses ZVD-convolution
    for better damped-system performance, but this educational plot shows the
    original spectral nulling concept.
    Output:
    - Saves 'educational_fourth_order_shaping.png'.
    """
    print("Generating Fourth-Order plot (PITCH rotation context)...")
    
    t_base, torque_base, _ = create_base_torque_profile()
    
    f1, f2 = sorted(MODE_FREQUENCIES)
    T_jerk, T_snap = 1.0/f1, 1.0/f2  # Window durations for spectral nulls
    
    # Rectangular windows (conceptual frequency-domain approach)
    n_jerk, n_snap = int(T_jerk/DT), int(T_snap/DT)
    win_jerk = np.ones(n_jerk) / n_jerk
    win_snap = np.ones(n_snap) / n_snap
    
    # Base pulse
    T_base_pulse = np.sqrt(THETA_FINAL / (MAX_TORQUE / I_AXIS))
    n_base = int(T_base_pulse / DT)
    pulse = np.ones(n_base) * (MAX_TORQUE / I_AXIS)
    
    # Convolutions
    after_jerk = np.convolve(pulse, win_jerk, mode='full')
    after_snap = np.convolve(after_jerk, win_snap, mode='full')
    
    # Symmetric profile
    alpha = np.concatenate([after_snap, -after_snap[::-1]])
    t_fourth = np.arange(len(alpha)) * DT
    
    # Scale
    omega = np.cumsum(alpha) * DT
    theta = np.cumsum(omega) * DT
    scale = THETA_FINAL / (theta[-1] + 1e-10)
    alpha *= scale
    torque_fourth = I_AXIS * alpha
    
    # Sinc functions (frequency response of rectangular windows)
    freq_sinc = np.linspace(0.01, 2.0, 1000)
    sinc_jerk = np.abs(np.sinc(freq_sinc * T_jerk))**2
    sinc_snap = np.abs(np.sinc(freq_sinc * T_snap))**2
    sinc_comb = sinc_jerk * sinc_snap
    
    # PSDs
    freq_base, psd_base = compute_psd(torque_base, DT)
    freq_4th, psd_4th = compute_psd(torque_fourth, DT)
    psd_base_n = psd_base / np.max(psd_base)
    psd_4th_n = psd_4th / np.max(psd_base)
    
    # ========================================================================
    # FIGURE
    # ========================================================================
    fig = plt.figure(figsize=(14, 8))
    gs = gridspec.GridSpec(2, 3, figure=fig,
                           height_ratios=[1, 1],
                           width_ratios=[1, 1, 1],
                           hspace=0.35, wspace=0.35,
                           left=0.08, right=0.96, top=0.94, bottom=0.08)
    
    # (a) Base Pulse
    t_pulse = np.arange(len(pulse)) * DT
    t_jw = np.arange(n_jerk) * DT
    t_sw = np.arange(n_snap) * DT
    
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.fill_between(t_pulse, 0, pulse*1000, alpha=0.35, color=C_RED, edgecolor=C_RED, linewidth=2)
    ax1.set_ylabel('Acceleration (mrad/s^2)', fontsize=11)
    ax1.set_xlabel('Time (s)', fontsize=11)
    ax1.set_title('(a) Base Pulse', fontsize=11, fontweight='bold', pad=8)
    ax1.text(0.02, 0.96, f'T = {T_base_pulse:.1f} s', transform=ax1.transAxes,
             va='top', ha='left', fontsize=9, bbox=dict(boxstyle='round,pad=0.4', 
             facecolor='white', edgecolor='none', alpha=0.8))
    
    # (b) Jerk-Limiting Window (rectangular)
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.fill_between(t_jw, 0, win_jerk*n_jerk, alpha=0.35, color=C_BLUE, edgecolor=C_BLUE, linewidth=2)
    ax2.set_ylabel('Amplitude', fontsize=11)
    ax2.set_xlabel('Time (s)', fontsize=11)
    ax2.set_title(f'(b) Jerk-Limiting Window', fontsize=11, fontweight='bold', pad=8)
    ax2.text(0.02, 0.96, f'T = 1/f1 = {T_jerk:.2f} s\nnull @ {f1} Hz (Mode 1)', transform=ax2.transAxes,
             va='top', ha='left', fontsize=9, bbox=dict(boxstyle='round,pad=0.4', 
             facecolor='white', edgecolor='none', alpha=0.8))
    
    # (c) Snap-Limiting Window (rectangular)
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.fill_between(t_sw, 0, win_snap*n_snap, alpha=0.35, color=C_PURPLE, edgecolor=C_PURPLE, linewidth=2)
    ax3.set_ylabel('Amplitude', fontsize=11)
    ax3.set_xlabel('Time (s)', fontsize=11)
    ax3.set_title(f'(c) Snap-Limiting Window', fontsize=11, fontweight='bold', pad=8)
    ax3.text(0.02, 0.96, f'T = 1/f2 = {T_snap:.2f} s\nnull @ {f2} Hz (Mode 2)', transform=ax3.transAxes,
             va='top', ha='left', fontsize=9, bbox=dict(boxstyle='round,pad=0.4', 
             facecolor='white', edgecolor='none', alpha=0.8))
    
    # (d) S-Curve Profile
    t_as = np.arange(len(after_snap)) * DT
    ax4 = fig.add_subplot(gs[1, 0])
    ax4.fill_between(t_as, 0, after_snap*1000, alpha=0.3, color=C_GREEN, edgecolor=C_GREEN, linewidth=2)
    ax4.set_ylabel('Acceleration (mrad/s^2)', fontsize=11)
    ax4.set_xlabel('Time (s)', fontsize=11)
    ax4.set_title('(d) S-Curve Profile', fontsize=11, fontweight='bold', pad=8)
    ax4.text(0.02, 0.96, '(a) * (b) * (c)\nsmooth jerk & snap', transform=ax4.transAxes,
             va='top', ha='left', fontsize=9, bbox=dict(boxstyle='round,pad=0.4', 
             facecolor='white', edgecolor='none', alpha=0.8))
    
    # (e) Full Trajectory
    ax5 = fig.add_subplot(gs[1, 1])
    ax5.plot(t_fourth, torque_fourth*1000, color=C_GREEN, linewidth=2.5)
    ax5.axhline(y=0, color='gray', linewidth=0.8, linestyle='--', alpha=0.4)
    ax5.set_ylabel('Torque (mNm)', fontsize=11)
    ax5.set_xlabel('Time (s)', fontsize=11)
    ax5.set_title('(e) Final PITCH Torque Profile (Symmetric)', fontsize=11, fontweight='bold', pad=8)
    ax5.set_xlim([0, t_fourth[-1]])
    
    # (f) Spectral Nulling (Transfer Function)
    ax6 = fig.add_subplot(gs[1, 2])
    
    # Add subtle shaded regions to highlight the null zones
    ax6.axvspan(f1 - 0.05, f1 + 0.05, alpha=0.12, color=C_BLUE, zorder=0)
    ax6.axvspan(f2 - 0.08, f2 + 0.08, alpha=0.12, color=C_PURPLE, zorder=0)
    
    # Plot the transfer functions with clean linear scaling
    ax6.semilogy(freq_sinc, sinc_jerk, color=C_BLUE, lw=2.5, ls='--', alpha=0.8, label='Jerk Window', zorder=2)
    ax6.semilogy(freq_sinc, sinc_snap, color=C_PURPLE, lw=2.5, ls='--', alpha=0.8, label='Snap Window', zorder=2)
    ax6.semilogy(freq_sinc, sinc_comb, color=C_GREEN, lw=3.5, label='Combined', zorder=3)
    
    # Mark the modal frequencies with clean vertical lines
    ax6.axvline(x=f1, color=C_BLUE, ls=':', lw=2, alpha=0.6, zorder=1)
    ax6.axvline(x=f2, color=C_PURPLE, ls=':', lw=2, alpha=0.6, zorder=1)
    
    # Clean axis formatting
    ax6.set_xlabel('Frequency (Hz)', fontsize=11, fontweight='medium')
    ax6.set_ylabel('Transfer Function |H(f)|²', fontsize=11, fontweight='medium')
    ax6.set_title('(f) Spectral Nulling', fontsize=11, fontweight='bold', pad=8)
    ax6.set_xlim([0.1, 2.0])
    ax6.set_ylim([1e-14, 1e0])
    ax6.set_yticks([1e-14,1e-13,1e-12,1e-11,1e-10,1e-9,1e-8,1e-7,1e-6,1e-5,1e-4,1e-3,1e-2,1e-1,1e0])
    
    # Legend
    legend = ax6.legend(loc='upper right', fontsize=9, framealpha=0.95)
    legend.get_frame().set_linewidth(0.8)
    
    # Simple frequency labels at nulls
    ax6.text(f1, 1.02, 'f₁', fontsize=10, ha='center', color=C_BLUE, fontweight='bold')
    ax6.text(f2, 1.02, 'f₂', fontsize=10, ha='center', color=C_PURPLE, fontweight='bold')
    
    plt.savefig('educational_fourth_order_shaping.png', dpi=250, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    print("Saved: educational_fourth_order_shaping.png")
    plt.close()


def main():
    """Generate the educational visualization figures."""
    print(f"\nCreating educational figures for {np.degrees(THETA_FINAL):.0f} deg PITCH slew")
    print(f"Maneuver: 180 deg PITCH rotation to excite solar array flex modes")
    print(f"Modes: {MODE_FREQUENCIES[0]} Hz, {MODE_FREQUENCIES[1]} Hz")
    print(f"Context: Step-and-stare survey (slew -> settle -> image)\n")
    
    create_zvd_educational_plot()
    create_fourth_order_educational_plot()
    
    print("\nDone!")


if __name__ == '__main__':
    main()
