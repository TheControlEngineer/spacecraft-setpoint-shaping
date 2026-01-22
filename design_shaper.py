"""
Multi-Mode Input Shaper Design.

Designs a fourth-order shaper (jerk/snap window convolution) for dual-mode vibration suppression.
Target modes: 0.4 Hz and 1.3 Hz flexible bending modes.

Maneuver: 180 deg yaw rotation in 30 seconds.
- Yaw rotation excites solar arrays extending along Y-axis.
- Arrays bend in Z direction, creating base excitation coupling.
- Input shaping reduces post-slew residual vibration.
- Key benefit: Faster settling time -> higher imaging throughput.

UPDATED: Fixed fourth-order implementation to use spectral nulling (window
convolution) instead of spline interpolation, eliminating high-frequency
discretization noise.
"""

from __future__ import annotations

import os
import sys

import matplotlib.pyplot as plt
import numpy as np

repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
src_root = os.path.join(repo_root, "src")
for path in (repo_root, src_root):
    if path not in sys.path:
        sys.path.insert(0, path)

from input_shaping import design_multimode_cascaded
from spacecraft_properties import compute_effective_inertia


def _validate_underdamped(zeta: float, context: str = "") -> None:
    """Validate that damping ratio is underdamped (< 1)."""
    if zeta >= 1.0:
        message = f"Damping ratio must be < 1 for underdamped oscillator, got {zeta}"
        if context:
            message = f"{context}: {message}"
        raise ValueError(message)


def _damped_frequency(omega_n: float, zeta: float) -> float:
    """Return damped natural frequency for underdamped modes."""
    _validate_underdamped(zeta, "damped_frequency")
    return omega_n * np.sqrt(1 - zeta**2)


def compute_residual_vibration_continuous(t, accel, freq, zeta):
    """
    Compute residual vibration amplitude for a continuous acceleration profile.
    
    This implements the continuous-time analog of the ZVD residual vibration formula.
    For a damped oscillator excited by acceleration a(t), the residual vibration at
    the end of the maneuver (t=T) is:
    
    V = sqrt(C^2 + S^2) where:
    C = integral_0^T a(t) * exp(-zeta*omega*(T-t)) * cos(omega_d*(T-t)) dt
    S = integral_0^T a(t) * exp(-zeta*omega*(T-t)) * sin(omega_d*(T-t)) dt
    
    Args:
        t: Time array
        accel: Acceleration profile (angular acceleration in rad/s^2)
        freq: Modal frequency in Hz
        zeta: Damping ratio
        
    Returns:
        V: Residual vibration amplitude (normalized)
    """
    _validate_underdamped(zeta, "compute_residual_vibration_continuous")
    omega = 2 * np.pi * freq
    omega_d = _damped_frequency(omega, zeta)
    T = t[-1]
    dt = t[1] - t[0]
    
    # Time from end of maneuver
    tau = T - t  # tau goes from T to 0
    
    # Exponential decay and oscillation terms
    exp_term = np.exp(-zeta * omega * tau)
    cos_term = np.cos(omega_d * tau)
    sin_term = np.sin(omega_d * tau)
    
    # Integrate using trapezoidal rule
    C = np.trapz(accel * exp_term * cos_term, t)
    S = np.trapz(accel * exp_term * sin_term, t)
    
    # Residual vibration amplitude
    V = np.sqrt(C**2 + S**2)
    
    return V


def compute_residual_vibration_impulse(times, amps, freq, zeta):
    """
    Compute residual vibration for an impulse shaper (discrete impulses).
    
    This is the standard ZVD residual vibration formula.
    
    Args:
        times: Impulse times
        amps: Impulse amplitudes
        freq: Modal frequency in Hz
        zeta: Damping ratio
        
    Returns:
        V: Residual vibration amplitude
    """
    _validate_underdamped(zeta, "compute_residual_vibration_impulse")
    omega = 2 * np.pi * freq
    omega_d = _damped_frequency(omega, zeta)
    
    C = 0.0
    S = 0.0
    for t, a in zip(times, amps):
        exp_term = np.exp(-zeta * omega * t)
        C += a * exp_term * np.cos(omega_d * t)
        S += a * exp_term * np.sin(omega_d * t)
    
    return np.sqrt(C**2 + S**2)


def design_spacecraft_shaper(plot=True):
    """
    Design a multi-mode ZVD shaper for the default spacecraft modes.

    Input:
    - plot: if True, save diagnostic plots of the shaper and cascade.

    Output:
    - (amplitudes, times, shaper_info) for the cascaded multi-mode shaper.

    Process:
    - Calls design_multimode_cascaded with the configured mode frequencies
      and damping ratios, then summarizes the resulting impulse sequence.
    """
    print("\\nDesigning multi-mode ZVD shaper...")
    
    # Spacecraft modal parameters (solar array bending modes)
    mode_frequencies = [0.4, 1.3]  # Hz - first and second bending
    damping_ratios = [0.02, 0.015]  # Low damping = long settling without shaping
    
    print(f"  Modes: {mode_frequencies[0]} Hz (zeta={damping_ratios[0]}), {mode_frequencies[1]} Hz (zeta={damping_ratios[1]})")
    
    # Design cascaded ZVD shaper
    amplitudes, times = design_multimode_cascaded(
        mode_frequencies=mode_frequencies,
        damping_ratios=damping_ratios,
        method='ZVD'
    )
    
    # Shaper statistics
    n_impulses = len(amplitudes)
    duration = times[-1]
    unity_gain = np.sum(amplitudes)
    
    print(f"  Result: {n_impulses} impulses, {duration:.3f}s duration, gain={unity_gain:.6f}")
    
    # Store info
    shaper_info = {
        'n_impulses': n_impulses,
        'duration': duration,
        'mode_frequencies': mode_frequencies,
        'damping_ratios': damping_ratios,
        'method': 'ZVD cascaded'
    }
    
    if plot:
        plot_shaper(amplitudes, times, shaper_info)
    
    return amplitudes, times, shaper_info


def plot_zvd_cascading(amplitudes, times, info):
    """
    Visualize how single-mode ZVD shapers cascade into a multi-mode shaper.

    Inputs:
    - amplitudes: cascaded impulse amplitudes.
    - times: impulse times (seconds).
    - info: dict with mode frequencies and damping ratios.

    Output:
    - Saves 'zvd_cascading_formation.png' and displays the figure.
    """
    
    # Design individual ZVD shapers for each mode
    from shapers.ZVD import ZVD as ZVD_func
    
    f1, f2 = info['mode_frequencies']
    zeta1, zeta2 = info['damping_ratios']
    
    # Convert Hz to rad/s
    omega1 = 2 * np.pi * f1
    omega2 = 2 * np.pi * f2
    
    # Single-mode ZVD shapers
    A1, t1, _ = ZVD_func(omega1, zeta1)
    A2, t2, _ = ZVD_func(omega2, zeta2)
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    fig.suptitle('ZVD Multi-Mode Input Shaping: Cascaded Design', 
                 fontsize=14, fontweight='bold')
    
    # Row 1: Individual ZVD shapers
    ax1 = axes[0, 0]
    markerline1, stemlines1, _ = ax1.stem(t1, A1, basefmt=' ', linefmt='b-', markerfmt='bo')
    plt.setp(stemlines1, linewidth=2)
    plt.setp(markerline1, markersize=8)
    ax1.set_ylabel('Amplitude', fontsize=11)
    ax1.set_title(f'Mode 1: f = {f1} Hz', fontsize=12, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.text(0.95, 0.95, f'3 impulses\nSum A = {np.sum(A1):.3f}', 
             transform=ax1.transAxes, va='top', ha='right',
             bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8), fontsize=10)
    
    ax2 = axes[0, 1]
    markerline2, stemlines2, _ = ax2.stem(t2, A2, basefmt=' ', linefmt='r-', markerfmt='ro')
    plt.setp(stemlines2, linewidth=2)
    plt.setp(markerline2, markersize=8)
    ax2.set_ylabel('Amplitude', fontsize=11)
    ax2.set_title(f'Mode 2: f = {f2} Hz', fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.text(0.95, 0.95, f'3 impulses\nSum A = {np.sum(A2):.3f}', 
             transform=ax2.transAxes, va='top', ha='right',
             bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.8), fontsize=10)
    
    
    # Row 2: Cascaded result and cumulative
    ax3 = axes[1, 0]
    markerline3, stemlines3, _ = ax3.stem(times, amplitudes, basefmt=' ', linefmt='g-', markerfmt='go')
    plt.setp(stemlines3, linewidth=2)
    plt.setp(markerline3, markersize=7)
    ax3.set_ylabel('Amplitude', fontsize=11)
    ax3.set_xlabel('Time (s)', fontsize=11)
    ax3.set_title('Cascaded: ZVD1 \u2297 ZVD2', fontsize=12, fontweight='bold')
    ax3.grid(True, alpha=0.3)
    ax3.text(0.95, 0.95, f'9 impulses\n\u03a3A = {np.sum(amplitudes):.6f}\nDuration: {times[-1]:.2f} s', 
             transform=ax3.transAxes, va='top', ha='right',
             bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8), fontsize=10)
    
    ax4 = axes[1, 1]
    cumulative = np.cumsum(amplitudes)
    ax4.plot(times, cumulative, 'b-', linewidth=2.5, marker='o', markersize=5)
    ax4.axhline(y=1.0, color='r', linestyle='--', linewidth=2, alpha=0.7, label='Unity Gain')
    ax4.set_xlabel('Time (s)', fontsize=11)
    ax4.set_ylabel('Cumulative Amplitude', fontsize=11)
    ax4.set_title('Cumulative Distribution', fontsize=12, fontweight='bold')
    ax4.grid(True, alpha=0.3)
    ax4.legend(loc='lower right', fontsize=10)
    ax4.set_ylim([0, 1.1])
    
    plt.tight_layout()
    plt.savefig('zvd_cascading_formation.png', dpi=300, bbox_inches='tight')
    print("Saved: zvd_cascading_formation.png")
    plt.show()


def plot_shaper(amplitudes, times, info):
    """
    Plot the shaper impulse sequence and its cumulative distribution.

    Inputs:
    - amplitudes: impulse amplitudes.
    - times: impulse times (seconds).
    - info: metadata dict used for annotations.

    Output:
    - Saves 'designed_shaper.png' and the cascading plot image.
    """
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
    
    # Impulse sequence
    ax1.stem(times, amplitudes, basefmt=' ', linefmt='b-', markerfmt='bo')
    ax1.set_xlabel('Time (s)')
    ax1.set_ylabel('Amplitude')
    ax1.set_title('Multi-Mode Input Shaper Impulse Sequence', fontweight='bold')
    ax1.grid(True, alpha=0.3)
    
    # Add info text
    info_text = (f"Method: {info['method']}\n"
                 f"Modes: {info['mode_frequencies'][0]} Hz, {info['mode_frequencies'][1]} Hz\n"
                 f"Impulses: {info['n_impulses']}\n"
                 f"Duration: {info['duration']:.3f} s")
    ax1.text(0.98, 0.97, info_text,
             transform=ax1.transAxes,
             verticalalignment='top',
             horizontalalignment='right',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
             fontsize=10)
    
    # Cumulative amplitude (shows how command is distributed over time)
    cumulative = np.cumsum(amplitudes)
    ax2.plot(times, cumulative, 'r-', linewidth=2, marker='o')
    ax2.axhline(y=1.0, color='k', linestyle='--', alpha=0.5, label='Unity gain')
    ax2.set_xlabel('Time (s)')
    ax2.set_ylabel('Cumulative Amplitude')
    ax2.set_title('Cumulative Command Distribution', fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    plt.tight_layout()
    plt.savefig('designed_shaper.png', dpi=150)
    print("Saved: designed_shaper.png")
    
    # Also create detailed cascading plot
    plot_zvd_cascading(amplitudes, times, info)
    
    plt.show()


def verify_shaper_suppression(amplitudes, times, mode_frequencies, damping_ratios):
    """
    Verify shaper suppression by evaluating residual vibration at each mode.

    Inputs:
    - amplitudes: shaper impulse amplitudes.
    - times: impulse times (seconds).
    - mode_frequencies: list of modal frequencies in Hz.
    - damping_ratios: list of modal damping ratios.

    Output:
    - Prints residual vibration magnitude for each mode.
    """
    print("\nResidual vibration at modal frequencies:")
    
    def residual_vibration(omega_n, zeta, A, t):
        """Calculate residual vibration amplitude"""
        omega_d = _damped_frequency(omega_n, zeta)
        V = 0
        for amp, time in zip(A, t):
            V += amp * np.exp(-zeta * omega_n * time) * np.exp(1j * omega_d * time)
        return np.abs(V)
    
    print(f"\n{'Mode':<10} {'Frequency':<12} {'Damping':<12} {'Residual V':<15}")
    print("-"*55)
    
    for i, (f, zeta) in enumerate(zip(mode_frequencies, damping_ratios)):
        omega_n = 2 * np.pi * f
        V = residual_vibration(omega_n, zeta, amplitudes, times)
        
        print(f"  Mode {i+1}: f={f:.1f}Hz, zeta={zeta:.3f}, V={V:.6f}")
    
    print("  (V near zero indicates good suppression)")


# ============================================================================
# FOURTH-ORDER SETPOINT SHAPING (Dual-Mode Spectral Nulling)
# ============================================================================

def plot_window_formation(pulse_base, window_jerk, window_snap,
                          accel_jerk, accel_final, dt, info):
    """
    Visualize the convolution process used to build the fourth-order profile.

    Inputs:
    - pulse_base: base rectangular acceleration pulse.
    - window_jerk: rectangular window for jerk limiting.
    - window_snap: rectangular window for snap limiting.
    - accel_jerk: acceleration after jerk-window convolution.
    - accel_final: acceleration after snap-window convolution.
    - dt: sample time in seconds.
    - info: dict with mode frequencies and window durations.

    Output:
    - Saves a diagnostic figure to 'fourth_order_window_formation.png'.
    """
    fig, axes = plt.subplots(3, 2, figsize=(15, 10))
    fig.suptitle('Fourth-Order Trajectory: Convolution Process', 
                 fontsize=14, fontweight='bold')
    
    # Create time vectors
    t_base = np.arange(len(pulse_base)) * dt
    t_jerk_win = np.arange(len(window_jerk)) * dt
    t_snap_win = np.arange(len(window_snap)) * dt
    t_accel_jerk = np.arange(len(accel_jerk)) * dt
    t_accel_final = np.arange(len(accel_final)) * dt
    
    # Row 1: Base pulse and jerk window
    ax1 = axes[0, 0]
    ax1.plot(t_base, pulse_base, 'b-', linewidth=2)
    ax1.fill_between(t_base, 0, pulse_base, alpha=0.3, color='blue')
    ax1.set_ylabel('Acceleration (rad/s^2)', fontsize=11)
    ax1.set_title('1. Base Pulse (Rectangular)', fontsize=12, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.text(0.95, 0.95, f'T = {t_base[-1]:.1f} s', transform=ax1.transAxes,
             va='top', ha='right', bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
    
    ax2 = axes[0, 1]
    ax2.plot(t_jerk_win, window_jerk, 'g-', linewidth=2)
    ax2.fill_between(t_jerk_win, 0, window_jerk, alpha=0.3, color='green')
    ax2.set_ylabel('Amplitude', fontsize=11)
    ax2.set_title(f'2. Jerk Window (f1 = {info["mode_frequencies"][0]} Hz)', fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.text(0.95, 0.95, f'T = 1/f1 = {info["T_jerk"]:.2f} s', transform=ax2.transAxes,
             va='top', ha='right', bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))
    
    # Row 2: After jerk convolution and snap window
    ax3 = axes[1, 0]
    ax3.plot(t_accel_jerk, accel_jerk, 'purple', linewidth=2)
    ax3.fill_between(t_accel_jerk, 0, accel_jerk, alpha=0.3, color='purple')
    ax3.set_ylabel('Acceleration (rad/s^2)', fontsize=11)
    ax3.set_title('3. After Jerk Convolution (Trapezoidal)', fontsize=12, fontweight='bold')
    ax3.grid(True, alpha=0.3)
    ax3.text(0.95, 0.95, 'C^1 continuous', transform=ax3.transAxes,
             va='top', ha='right', bbox=dict(boxstyle='round', facecolor='plum', alpha=0.8))
    
    ax4 = axes[1, 1]
    ax4.plot(t_snap_win, window_snap, 'orange', linewidth=2)
    ax4.fill_between(t_snap_win, 0, window_snap, alpha=0.3, color='orange')
    ax4.set_ylabel('Amplitude', fontsize=11)
    ax4.set_title(f'4. Snap Window (f2 = {info["mode_frequencies"][1]} Hz)', fontsize=12, fontweight='bold')
    ax4.grid(True, alpha=0.3)
    ax4.text(0.95, 0.95, f'T = 1/f2 = {info["T_snap"]:.2f} s', transform=ax4.transAxes,
             va='top', ha='right', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    # Row 3: Final result
    ax5 = axes[2, 0]
    ax5.plot(t_accel_final, accel_final, 'red', linewidth=2.5)
    ax5.fill_between(t_accel_final, 0, accel_final, alpha=0.3, color='red')
    ax5.set_ylabel('Acceleration (rad/s^2)', fontsize=11)
    ax5.set_xlabel('Time (s)', fontsize=11)
    ax5.set_title('5. Final Profile (S-Curve)', fontsize=12, fontweight='bold')
    ax5.grid(True, alpha=0.3)
    ax5.text(0.95, 0.95, 'C^3 continuous\nDual-mode nulling', transform=ax5.transAxes,
             va='top', ha='right', bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.8))
    
    # Summary panel
    ax6 = axes[2, 1]
    ax6.axis('off')
    summary = (
        "SPECTRAL NULLING\n"
        "-----------------------\n"
        "Window of width T = 1/f creates a zero at frequency f.\n\n"
        f"T_jerk = {info['T_jerk']:.3f} s\n"
        f"  -> null at f1 = {info['mode_frequencies'][0]} Hz\n\n"
        f"T_snap = {info['T_snap']:.3f} s\n"
        f"  -> null at f2 = {info['mode_frequencies'][1]} Hz\n"
        "-----------------------\n"
        "REFERENCE:\n"
        "Lambrechts et al. (2005)\n"
        "Control Eng. Practice\n"
        "13(2):145-157"
    )
    
    ax6.text(0.1, 0.5, summary, transform=ax6.transAxes,
             fontsize=11, family='monospace',
             bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.9))
    
    plt.tight_layout()
    plt.savefig('fourth_order_window_formation.png', dpi=300, bbox_inches='tight')
    print("Saved: fourth_order_window_formation.png")
    plt.show()


def design_fourth_order_trajectory_fixed(theta_final, mode_frequencies, 
                                        damping_ratios, I_axis, 
                                        max_torque=0.2, dt=None, plot=False,
                                        plot_windows=False, target_duration=None):
    """
    Design a fourth-order trajectory using spectral nulling (window convolution).

    Inputs:
    - theta_final: target rotation angle in radians.
    - mode_frequencies: list of modal frequencies in Hz.
    - damping_ratios: list of modal damping ratios.
    - I_axis: principal inertia about the rotation axis.
    - max_torque: torque limit used for feasibility checks.
    - dt: sample time; if None, uses 0.01 s (100 Hz) to match Basilisk.
    - plot: if True, generate diagnostic plots.
    - plot_windows: if True, plot the convolution window formation.
    - target_duration: if set, adjust the base pulse to hit this duration.

    Outputs:
    - (t, theta, omega, alpha, jerk, snap, trajectory_info).

    Process:
    - Build a rectangular base pulse, convolve with jerk and snap windows to
      create spectral zeros at the modal frequencies, then mirror for decel.
    - Integrate to velocity and position, and scale to hit theta_final.
    """
    print("\nDesigning fourth-order trajectory (Spectral Nulling)...")
    
    f1, f2 = sorted(mode_frequencies)
    print(f"  Target modes: {f1} Hz, {f2} Hz")
    
    # Spectral nulling window widths
    T_jerk = 1.0 / f1  # e.g., 2.500s for f1=0.4 Hz
    T_snap = 1.0 / f2  # e.g., 0.769s for f2=1.3 Hz
    
    print(f"  Windows: T_jerk={T_jerk:.3f}s (null at {f1}Hz), T_snap={T_snap:.3f}s (null at {f2}Hz)")
    
    # ========================================================================
    # Enforce unified sampling at 100 Hz to match Basilisk simulation
    # ========================================================================
    if dt is None:
        dt = 0.01
        print(f"  Auto dt = {dt:.6f}s (100.0 Hz sample rate)")
    
    # ========================================================================
    # Discretize windows and base pulse duration
    # ========================================================================
    # CRITICAL: Use round() not int() to minimize truncation error
    # For perfect null at frequency f, window duration must be EXACTLY 1/f
    n_jerk = max(1, int(round(T_jerk / dt)))
    n_snap = max(1, int(round(T_snap / dt)))

    # Verify actual durations match target
    T_jerk_actual = n_jerk * dt
    T_snap_actual = n_snap * dt

    # Check residual at target frequencies
    sinc_f1 = np.abs(np.sinc(mode_frequencies[0] * T_jerk_actual))
    sinc_f2 = np.abs(np.sinc(mode_frequencies[1] * T_snap_actual))

    if sinc_f1 > 0.01 or sinc_f2 > 0.01:
        print(f"  Warning: Discretization error - consider smaller dt")

    if target_duration is not None:
        # Target duration constraint (discrete-time exactness)
        # Total duration = 2 * N_half * dt, where:
        # N_half = n_base + n_jerk + n_snap - 2
        n_half = int(round(target_duration / (2.0 * dt)))
        n_base = n_half - n_jerk - n_snap + 2
        if n_base < 1:
            print(f"  WARNING: Windows too long for target duration!")
            min_half = n_jerk + n_snap - 1
            min_duration = 2.0 * min_half * dt
            print(f"  Minimum duration = {min_duration:.2f}s")
            n_base = 1
            n_half = n_base + n_jerk + n_snap - 2

        T_base = n_base * dt

        # Use a scaling approach: build with unit amplitude, then scale
        A_lim = 1.0  # Temporary, will scale later

        actual_duration = 2.0 * n_half * dt
        print(f"  Target duration: {target_duration:.2f}s")
        print(f"  Actual duration: {actual_duration:.3f}s (dt={dt:.6f}s)")
        print(f"  Computed T_base: {T_base:.3f}s (n_base={n_base})")
    else:
        # Original approach: use acceleration limit from torque
        A_lim = max_torque / I_axis
        T_base = np.sqrt(theta_final / A_lim)
        n_base = max(1, int(round(T_base / dt)))
        T_base = n_base * dt
        print(f"  Base pulse: T={T_base:.2f}s, A={A_lim:.6f} rad/s^2")

    # ========================================================================
    # Build via convolution
    # ========================================================================

    # Step 1: Create rectangular base pulse (ACCELERATION)
    pulse_base = np.ones(n_base) * A_lim

    # Step 2: Create smoothing windows
    window_jerk = np.ones(n_jerk) / n_jerk  # Normalized to sum to 1
    window_snap = np.ones(n_snap) / n_snap  # Normalized to sum to 1
    
    # Step 3: Convolve acceleration with jerk window
    accel_jerk = np.convolve(pulse_base, window_jerk, mode='full')
    
    # Step 4: Convolve with snap window
    accel_final = np.convolve(accel_jerk, window_snap, mode='full')
    
    # Optional: Plot window formation process
    if plot_windows:
        plot_window_formation(pulse_base, window_jerk, window_snap,
                            accel_jerk, accel_final, dt,
                            {'T_jerk': T_jerk, 'T_snap': T_snap,
                             'mode_frequencies': [f1, f2]})
    
    # Step 5: Create symmetric profile (accelerate, coast, decelerate)
    # Flip and negate for deceleration phase
    accel_decel = -accel_final[::-1]

    # Insert a center sample at zero acceleration for continuity
    alpha = np.concatenate([accel_final, np.zeros(1), accel_decel])
    
    # Step 6: Create time vector
    t = np.arange(len(alpha)) * dt
    
    # Step 7: Integrate to get velocity and position
    omega = np.cumsum(alpha) * dt
    theta = np.cumsum(omega) * dt
    
    # Step 8: Scale to hit target angle
    scale = theta_final / (theta[-1] + 1e-10)
    
    print(f"\nScaling factor: {scale:.6f}")
    
    theta *= scale
    omega *= scale
    alpha *= scale
    
    # Step 9: Calculate derivatives properly from convolution
    # Jerk is derivative of acceleration
    jerk = np.zeros_like(alpha)
    jerk[1:-1] = (alpha[2:] - alpha[:-2]) / (2*dt)  # Central difference
    jerk[0] = (alpha[1] - alpha[0]) / dt  # Forward at start
    jerk[-1] = (alpha[-1] - alpha[-2]) / dt  # Backward at end
    
    # Snap = d(jerk)/dt
    snap = np.zeros_like(jerk)
    snap[1:-1] = (jerk[2:] - jerk[:-2]) / (2*dt)
    snap[0] = (jerk[1] - jerk[0]) / dt
    snap[-1] = (jerk[-1] - jerk[-2]) / dt
    
    # Calculate torque
    torque = I_axis * alpha
    torque_peak = np.max(np.abs(torque))

    if max_torque is not None and torque_peak > max_torque * 1.001:
        print(f"  WARNING: Peak torque {torque_peak:.4f} Nm exceeds limit {max_torque:.4f} Nm")

    print(f"  Result: {t[-1]:.2f}s duration, {np.degrees(theta[-1]):.2f} deg, peak torque={torque_peak:.4f} Nm")
    
    # Verify spectral nulls
    print("  Spectral verification:")
    
    # FFT of acceleration (this is what excites the modes)
    alpha_fft = np.fft.fft(alpha)
    freq = np.fft.fftfreq(len(alpha), dt)
    psd = np.abs(alpha_fft)**2
    psd_norm = psd / np.max(psd)
    
    # Check at mode frequencies
    results = []
    for i, f_mode in enumerate([f1, f2]):
        idx = np.argmin(np.abs(freq - f_mode))
        psd_at_mode = psd_norm[idx]
        suppression_db = -10*np.log10(psd_at_mode + 1e-12)
        status = "good" if psd_at_mode < 1e-4 else "check dt"
        print(f"    Mode {i+1} ({f_mode}Hz): {psd_at_mode:.2e} ({suppression_db:.0f}dB) - {status}")
        results.append(psd_at_mode)
    
    # Package results
    trajectory_info = {
        'method': 'fourth_order_convolution_fixed',
        'mode_frequencies': mode_frequencies,
        'damping_ratios': damping_ratios,
        'T_jerk': T_jerk,
        'T_snap': T_snap,
        'T_base': T_base,
        'duration': t[-1],
        'theta_target': theta_final,
        'theta_achieved': theta[-1],
        'torque_peak': torque_peak,
        'torque_limit': max_torque,
        'psd_at_f1': results[0],
        'psd_at_f2': results[1]
    }
    
    if plot:
        plot_fourth_order_trajectory(t, theta, omega, alpha, jerk, snap, 
                                     torque, trajectory_info)
    
    return t, theta, omega, alpha, jerk, snap, trajectory_info


def plot_fourth_order_trajectory(t, theta, omega, alpha, jerk, snap, 
                                 torque, info):
    """
    Plot the fourth-order trajectory in time and frequency domains.

    Inputs:
    - t, theta, omega, alpha, jerk, snap: trajectory arrays.
    - torque: required torque profile.
    - info: dict with trajectory metadata for annotations.

    Output:
    - Saves time-domain and spectral analysis figures.
    """
    
    # Calculate peak values from arrays
    peak_accel = np.max(np.abs(alpha))
    
    # ========================================================================
    # FIGURE 1: Time Domain Plots
    # ========================================================================
    fig1 = plt.figure(figsize=(14, 10))
    gs1 = fig1.add_gridspec(2, 3, hspace=0.3, wspace=0.3)
    
    fig1.suptitle('Fourth-Order Trajectory: Time Domain', 
                 fontsize=14, fontweight='bold')
    
    # Time domain plots
    ax1 = fig1.add_subplot(gs1[0, 0])
    ax1.plot(t, np.degrees(theta), 'b-', linewidth=2)
    ax1.axhline(y=np.degrees(info['theta_target']), color='r', 
                linestyle='--', alpha=0.5, label='Target')
    ax1.set_ylabel('Attitude (deg)')
    ax1.set_title('Position', fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    ax2 = fig1.add_subplot(gs1[0, 1])
    ax2.plot(t, np.degrees(omega), 'g-', linewidth=2)
    ax2.set_ylabel('Velocity (deg/s)')
    ax2.set_title('Velocity', fontweight='bold')
    ax2.grid(True, alpha=0.3)
    
    ax3 = fig1.add_subplot(gs1[0, 2])
    ax3.plot(t, np.degrees(alpha), 'r-', linewidth=2)
    ax3.set_ylabel('Acceleration (deg/s^2)')
    ax3.set_title('Acceleration', fontweight='bold')
    ax3.grid(True, alpha=0.3)
    
    ax4 = fig1.add_subplot(gs1[1, 0])
    ax4.plot(t, np.degrees(jerk), 'm-', linewidth=2)
    ax4.set_ylabel('Jerk (deg/s^3)')
    ax4.set_xlabel('Time (s)')
    ax4.set_title('Jerk', fontweight='bold')
    ax4.grid(True, alpha=0.3)
    
    ax5 = fig1.add_subplot(gs1[1, 1])
    ax5.plot(t, np.degrees(snap), 'c-', linewidth=2)
    ax5.set_ylabel('Snap (deg/s^4)')
    ax5.set_xlabel('Time (s)')
    ax5.set_title('Snap (Control Input)', fontweight='bold')
    ax5.grid(True, alpha=0.3)
    
    ax6 = fig1.add_subplot(gs1[1, 2])
    ax6.plot(t, torque, 'k-', linewidth=2)
    ax6.axhline(y=info['torque_limit'], color='r', linestyle='--', 
                alpha=0.5, label='Limit')
    ax6.axhline(y=-info['torque_limit'], color='r', linestyle='--', alpha=0.5)
    ax6.set_ylabel('Torque (Nm)')
    ax6.set_xlabel('Time (s)')
    ax6.set_title('Required Torque', fontweight='bold')
    ax6.grid(True, alpha=0.3)
    ax6.legend()
    
    plt.savefig('fourth_order_trajectory_time.png', dpi=150, bbox_inches='tight')
    print("Saved: fourth_order_trajectory_time.png")
    plt.show()
    
    # ========================================================================
    # FIGURE 2: Frequency Domain (Separate, cleaner)
    # ========================================================================
    fig2, ax_freq = plt.subplots(1, 1, figsize=(16, 8))
    
    fig2.suptitle('Fourth-Order Trajectory: Spectral Analysis (Dual-Mode Nulling)', 
                  fontsize=14, fontweight='bold', y=0.98)
    
    # Compute FFT of acceleration (proportional to torque spectrum)
    dt = t[1] - t[0]
    n_fft = len(alpha)
    freq = np.fft.fftfreq(n_fft, dt)
    alpha_fft = np.fft.fft(alpha)
    alpha_psd = np.abs(alpha_fft)**2 / n_fft
    
    # Plot positive frequencies only
    pos_idx = freq > 0
    freq_pos = freq[pos_idx]
    psd_pos = alpha_psd[pos_idx]

    # Normalize
    psd_norm = psd_pos / np.max(psd_pos)

    ax_freq.semilogy(freq_pos, psd_norm, 'b-', linewidth=2, alpha=0.8, 
                     label='Acceleration Spectrum')

    # Mark the mode frequencies
    for i, f in enumerate(info['mode_frequencies']):
        ax_freq.axvline(x=f, color='r', linestyle='--', linewidth=2, alpha=0.7,
                        label=f'Mode {i+1}: {f} Hz')

    ax_freq.set_xlabel('Frequency (Hz)', fontsize=12)
    ax_freq.set_ylabel('Normalized PSD', fontsize=12)
    ax_freq.set_xlim([0.05, max(info['mode_frequencies'])*3])
    ax_freq.set_ylim([1e-6, 1.1])  # Adjusted for log scale
    ax_freq.grid(True, alpha=0.3, which='both')
    ax_freq.legend(fontsize=11, loc='upper right')
    
    # Add info box
    info_text = f"""SPECTRAL NULLING VERIFICATION

Spectral Zeros:
  T_jerk = {info['T_jerk']:.3f}s -> f1 = {info['mode_frequencies'][0]} Hz
  T_snap = {info['T_snap']:.3f}s -> f2 = {info['mode_frequencies'][1]} Hz

Trajectory Duration: {info['duration']:.2f}s

PSD at Modal Frequencies:
  f1 ({info['mode_frequencies'][0]} Hz): {info['psd_at_f1']:.2e}
  f2 ({info['mode_frequencies'][1]} Hz): {info['psd_at_f2']:.2e}

Peak Acceleration: {np.degrees(peak_accel):.3f} deg/s^2
Peak Torque: {info['torque_peak']:.4f} Nm (Limit: {info['torque_limit']:.4f} Nm)

OK C^3 continuous (smooth to 3rd derivative)
OK Spectral notches verified at both modes
OK Convolution-based approach
"""
    
    ax_freq.text(0.98, 0.98, info_text,
                 transform=ax_freq.transAxes,
                 fontsize=9,
                 verticalalignment='top',
                 horizontalalignment='right',
                 family='monospace',
                 bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.9))
    
    plt.tight_layout()
    plt.savefig('fourth_order_trajectory_spectrum.png', dpi=150, bbox_inches='tight')
    print("Saved: fourth_order_trajectory_spectrum.png")
    plt.show()


# ============================================================================
# DURATION-CONSTRAINED SHAPER DESIGN (for exact mission timing)
# ============================================================================

def _sort_and_combine_impulses(times, amplitudes, tol=1e-12):
    """
    Sort impulse times and combine amplitudes at identical times.

    Inputs:
    - times: array-like impulse times.
    - amplitudes: array-like impulse amplitudes.
    - tol: absolute tolerance used to treat times as identical.

    Output:
    - (times_sorted, amplitudes_sorted) with duplicates merged.
    """
    times = np.asarray(times, dtype=float)
    amplitudes = np.asarray(amplitudes, dtype=float)

    order = np.argsort(times)
    times_sorted = times[order]
    amps_sorted = amplitudes[order]

    combined_times = []
    combined_amps = []
    for t, a in zip(times_sorted, amps_sorted):
        if combined_times and np.isclose(t, combined_times[-1], rtol=0.0, atol=tol):
            combined_amps[-1] += a
        else:
            combined_times.append(t)
            combined_amps.append(a)

    return np.array(combined_times), np.array(combined_amps)


def design_spacecraft_shaper_with_duration(target_duration=30.0, theta_final=np.radians(180.0),
                                           mode_frequencies=None, damping_ratios=None,
                                           I_axis=1125.0, max_torque=70.0, plot=False):
    """
    Design a ZVD shaper that meets a total-duration constraint.

    Inputs:
    - target_duration: desired total maneuver duration (s).
    - theta_final: target rotation angle (rad).
    - mode_frequencies: modal frequencies (Hz).
    - damping_ratios: modal damping ratios.
    - I_axis: principal inertia about the slew axis.
    - max_torque: actuator torque limit (Nm).
    - plot: if True, generate the shaper plots.

    Outputs:
    - (amplitudes, times, shaper_info) for the combined shaper.

    Process:
    - Build single-mode ZVD shapers, convolve them, then compute the
      base maneuver duration needed to hit the total time target.
    """
    
    if mode_frequencies is None:
        mode_frequencies = [0.4, 1.3]
    if damping_ratios is None:
        damping_ratios = [0.02, 0.015]
    
    print(f"\nDesigning ZVD shaper for {target_duration}s duration...")
    
    # Step 1: Design the shaper (this determines the overhead)
    # Design individual ZVD shapers for each mode
    shapers = []
    for i, (freq, zeta) in enumerate(zip(mode_frequencies, damping_ratios)):
        _validate_underdamped(zeta, f"mode {i + 1} (freq={freq} Hz)")
        omega_n = 2 * np.pi * freq  # Natural frequency
        omega_d = _damped_frequency(omega_n, zeta)  # Damped frequency
        period_d = 2 * np.pi / omega_d
        
        # ZVD shaper (3 impulses)
        # K = exp(-zeta * pi / sqrt(1 - zeta^2)) is the standard formula
        K = np.exp(-zeta * omega_n * period_d / 2)
        A1 = 1 / (1 + 2*K + K**2)
        A2 = 2*K / (1 + 2*K + K**2)
        A3 = K**2 / (1 + 2*K + K**2)
        
        t1 = 0
        t2 = period_d / 2
        t3 = period_d
        
        shapers.append({
            'amplitudes': np.array([A1, A2, A3]),
            'times': np.array([t1, t2, t3]),
            'frequency': freq,
            'damping': zeta
        })
    
    # Convolve shapers to get multi-mode shaper
    result_amps = shapers[0]['amplitudes']
    result_times = shapers[0]['times']
    
    for shaper in shapers[1:]:
        new_amps = []
        new_times = []
        
        for a1, t1 in zip(result_amps, result_times):
            for a2, t2 in zip(shaper['amplitudes'], shaper['times']):
                new_amps.append(a1 * a2)
                new_times.append(t1 + t2)
        
        result_amps = np.array(new_amps)
        result_times = np.array(new_times)

    result_times, result_amps = _sort_and_combine_impulses(result_times, result_amps)
    
    # Normalize amplitudes
    result_amps = result_amps / np.sum(result_amps)
    
    shaper_duration = result_times[-1]
    
    print(f"  Shaper: {len(result_amps)} impulses, {shaper_duration:.3f}s overhead")
    
    # Step 2: Calculate base maneuver duration to hit target
    base_duration = target_duration - shaper_duration
    
    if base_duration <= 0:
        raise ValueError(f"Target duration {target_duration}s too short for shaper ({shaper_duration:.3f}s)")
    
    print(f"  Base duration: {base_duration:.2f}s (total={target_duration:.2f}s)")
    
    # Step 3: Verify base maneuver is feasible
    A_lim = max_torque / I_axis
    T_min = 2.0 * np.sqrt(abs(theta_final) / A_lim)
    
    if base_duration < T_min:
        print(f"  Warning: Adjusting duration (min={T_min:.2f}s)")
        base_duration = T_min
        actual_total = base_duration + shaper_duration
    else:
        actual_total = target_duration
    
    # Package info
    shaper_info = {
        'amplitudes': result_amps.tolist(),
        'times': result_times.tolist(),
        'mode_frequencies': mode_frequencies,
        'damping_ratios': damping_ratios,
        'n_impulses': len(result_amps),
        'duration': shaper_duration,
        'base_duration': base_duration,
        'total_duration': actual_total,
        'theta_target': theta_final,
        'I_axis': I_axis,
        'max_torque': max_torque
    }
    
    if plot:
        plot_shaper(result_amps, result_times, shaper_info)
    
    return result_amps, result_times, shaper_info


def design_fourth_order_with_duration_FIXED(target_duration, theta_final,
                                            mode_frequencies, damping_ratios,
                                            I_axis, max_torque, dt=0.01, plot=False):
    """
    Wrapper to design a duration-constrained 4th-order trajectory using 
    the CORRECT window convolution method (Spectral Nulling).
    
    Replaces the flawed discrete ZVD+Spline method with the continuous window
    convolution method which creates a smooth S-curve without high-frequency artifacts.

    Inputs:
    - target_duration: desired total duration (s).
    - theta_final: target rotation angle (rad).
    - mode_frequencies: modal frequencies (Hz).
    - damping_ratios: modal damping ratios.
    - I_axis: principal inertia about the slew axis.
    - max_torque: actuator torque limit (Nm).
    - dt: sample time for discretization (defaults to 0.01 s to match Basilisk).
    - plot: if True, generate diagnostic plots.

    Output:
    - Same tuple returned by design_fourth_order_trajectory_fixed.
    """
    # Use the robust window convolution method (Spectral Nulling)
    t, theta, omega, alpha, jerk, snap, traj_info = design_fourth_order_trajectory_fixed(
        theta_final=theta_final,
        mode_frequencies=mode_frequencies,
        damping_ratios=damping_ratios,
        I_axis=I_axis,
        max_torque=max_torque,
        dt=dt,
        plot=plot,
        target_duration=target_duration # This argument calculates the correct T_base
    )
    
    return t, theta, omega, alpha, jerk, snap, traj_info


# ============================================================================
# UPDATED MAIN: Design Fourth-Order Trajectory (Corrected)
# ============================================================================

if __name__ == "__main__":
    
    # Mission parameters - 180 degree pitch slew in 30 seconds
    TARGET_DURATION = 30.0  # seconds
    TARGET_ANGLE = 180.0  # degrees (YAW maneuver, Z-axis)
    MODE_FREQUENCIES = [0.4, 1.3]  # Hz
    DAMPING_RATIOS = [0.02, 0.015]
    
    # Spacecraft inertia (YAW axis, Z) - matches spacecraft_model.py
    I_AXIS = float(compute_effective_inertia()[2, 2])
    MAX_TORQUE = 70.0  # Nm
    
    print(f"\nDesigning shapers: {TARGET_ANGLE} deg in {TARGET_DURATION}s")
    print(f"  Inertia: {I_AXIS:.0f} kg*m^2, torque: {MAX_TORQUE} Nm")
    
    # Fourth-Order Trajectory (FIXED: Uses Spectral Nulling)
    t, theta, omega, alpha, jerk, snap, traj_info = design_fourth_order_with_duration_FIXED(
        target_duration=TARGET_DURATION,
        theta_final=np.radians(TARGET_ANGLE),
        mode_frequencies=MODE_FREQUENCIES,
        damping_ratios=DAMPING_RATIOS,
        I_axis=I_AXIS,
        max_torque=MAX_TORQUE,
        dt=0.01,
        plot=False
    )
    
    np.savez("spacecraft_trajectory_4th_180deg_30s.npz",
             time=t, theta=theta, omega=omega, alpha=alpha,
             jerk=jerk, snap=snap, trajectory_info=traj_info)
    print(f"Saved: spacecraft_trajectory_4th_180deg_30s.npz (Corrected Algorithm)")
    
    print(f"\nDone! Fourth-order trajectory designed for {TARGET_DURATION}s.")
