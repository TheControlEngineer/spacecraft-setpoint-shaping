"""
Comprehensive Diagnostic Script for Control System Analysis

This script investigates:
1. Feedforward-only slew capability (can it achieve 180 degrees?)
2. Feedback controller vibration amplification
3. Phase margin reduction with filtered PD and AVC
4. Sensitivity function analysis at modal frequencies
5. Closed-loop stability and performance

Author: Diagnostic Analysis
"""

from __future__ import annotations

import numpy as np
from scipy import signal
from scipy.integrate import odeint
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass

# Import from the mission simulation
import sys
sys.path.insert(0, '.')
from spacecraft_properties import HUB_INERTIA


# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass
class DiagnosticConfig:
    """Configuration for diagnostic analysis."""
    # Spacecraft
    inertia: np.ndarray = None
    rotation_axis: np.ndarray = None

    # Modal parameters
    modal_freqs_hz: List[float] = None
    modal_damping: List[float] = None
    modal_gains: List[float] = None

    # Maneuver
    slew_angle_deg: float = 180.0
    slew_duration_s: float = 30.0

    # Control
    bandwidth_hz: float = 0.1
    damping_ratio: float = 0.7

    def __post_init__(self):
        if self.inertia is None:
            self.inertia = HUB_INERTIA.copy()
        if self.rotation_axis is None:
            self.rotation_axis = np.array([0.0, 0.0, 1.0])
        if self.modal_freqs_hz is None:
            self.modal_freqs_hz = [0.4, 1.3]
        if self.modal_damping is None:
            self.modal_damping = [0.02, 0.015]
        if self.modal_gains is None:
            self.modal_gains = [0.0015, 0.0008]


def get_config() -> DiagnosticConfig:
    return DiagnosticConfig()


# =============================================================================
# PART 1: FEEDFORWARD-ONLY SLEW VERIFICATION
# =============================================================================

def verify_feedforward_slew(config: DiagnosticConfig) -> Dict:
    """Verify that feedforward alone can achieve 180-degree slew."""
    print("\n" + "="*70)
    print("PART 1: FEEDFORWARD-ONLY SLEW VERIFICATION")
    print("="*70)

    theta_final = np.radians(config.slew_angle_deg)
    duration = config.slew_duration_s
    I_z = config.inertia[2, 2]  # Z-axis inertia

    # Bang-bang trajectory
    dt = 0.001
    t = np.arange(0, duration + dt, dt)
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
        else:
            t_dec = ti - t_half
            alpha[i] = -alpha_max
            omega[i] = alpha_max * t_half - alpha_max * t_dec
            theta[i] = (0.5 * alpha_max * t_half**2 +
                       alpha_max * t_half * t_dec -
                       0.5 * alpha_max * t_dec**2)

    torque = I_z * alpha

    # Results
    final_angle_deg = np.degrees(theta[-1])
    peak_rate_deg_s = np.degrees(np.max(np.abs(omega)))
    peak_torque = np.max(np.abs(torque))

    print(f"\nBang-Bang Trajectory Analysis:")
    print(f"  Target angle: {config.slew_angle_deg:.1f} deg")
    print(f"  Achieved angle: {final_angle_deg:.4f} deg")
    print(f"  Angle error: {abs(config.slew_angle_deg - final_angle_deg):.6f} deg")
    print(f"  Peak angular rate: {peak_rate_deg_s:.2f} deg/s")
    print(f"  Peak torque: {peak_torque:.2f} N.m")
    print(f"  Maneuver duration: {duration:.1f} s")

    # Verify with numerical integration of rigid body dynamics
    print(f"\n  Verification via numerical integration:")

    def rigid_body_dynamics(y, t_val, torque_func, I):
        theta, omega = y
        tau = torque_func(t_val)
        alpha = tau / I
        return [omega, alpha]

    def torque_interp(t_val):
        if t_val < 0:
            return 0.0
        if t_val > t[-1]:
            return 0.0
        idx = int(t_val / dt)
        if idx >= len(torque):
            idx = len(torque) - 1
        return torque[idx]

    y0 = [0.0, 0.0]
    y_integrated = odeint(rigid_body_dynamics, y0, t, args=(torque_interp, I_z))
    theta_integrated = y_integrated[:, 0]

    final_integrated_deg = np.degrees(theta_integrated[-1])
    print(f"  Integrated final angle: {final_integrated_deg:.4f} deg")
    print(f"  Integration vs analytical error: {abs(final_angle_deg - final_integrated_deg):.6f} deg")

    success = abs(config.slew_angle_deg - final_angle_deg) < 0.1
    print(f"\n  FEEDFORWARD SLEW CAPABILITY: {'PASS' if success else 'FAIL'}")

    return {
        'success': success,
        'final_angle_deg': final_angle_deg,
        'peak_rate_deg_s': peak_rate_deg_s,
        'peak_torque': peak_torque,
        'time': t,
        'theta': theta,
        'omega': omega,
        'torque': torque,
    }


# =============================================================================
# PART 2: FEEDBACK CONTROLLER ANALYSIS
# =============================================================================

def build_plant_transfer_function(config: DiagnosticConfig) -> Tuple[signal.TransferFunction, Dict]:
    """Build the flexible spacecraft plant transfer function."""
    I_z = config.inertia[2, 2]

    # Rigid body: G_rigid(s) = 1/(I*s^2)
    rigid_num = np.array([1.0])
    rigid_den = np.array([I_z, 0.0, 0.0])

    # Add flexible modes in parallel
    current_num = rigid_num
    current_den = rigid_den

    for i, (f_hz, zeta, gain) in enumerate(zip(
        config.modal_freqs_hz, config.modal_damping, config.modal_gains
    )):
        omega_n = 2 * np.pi * f_hz

        # Mode transfer function: G_mode(s) = gain/I / (s^2 + 2*zeta*omega_n*s + omega_n^2)
        mode_num = np.array([gain / I_z])
        mode_den = np.array([1.0, 2.0 * zeta * omega_n, omega_n**2])

        # Parallel addition: G_total = G_current + G_mode
        term1 = np.convolve(current_num, mode_den)
        term2 = np.convolve(mode_num, current_den)

        max_len = max(len(term1), len(term2))
        term1 = np.pad(term1, (max_len - len(term1), 0), mode='constant')
        term2 = np.pad(term2, (max_len - len(term2), 0), mode='constant')

        current_num = term1 + term2
        current_den = np.convolve(current_den, mode_den)

    plant = signal.TransferFunction(current_num, current_den)

    # Analyze plant
    zeros = np.roots(current_num)
    poles = np.roots(current_den)

    info = {
        'zeros': zeros,
        'poles': poles,
        'num': current_num,
        'den': current_den,
    }

    return plant, info


def build_controllers(config: DiagnosticConfig) -> Dict[str, signal.TransferFunction]:
    """Build all controller transfer functions.

    FIXED controller design:
    - Standard PD: Pure PD at 0.1 Hz bandwidth
    - Filtered PD: PD with filter at 10x bandwidth (well above crossover)
    - AVC: Reduced bandwidth (first_mode/6) with filter at 5x bandwidth

    The key fix is placing the derivative filter WELL ABOVE the crossover
    frequency to avoid phase margin loss.
    """
    I_z = config.inertia[2, 2]
    first_mode_hz = config.modal_freqs_hz[0]

    controllers = {}
    info = {}

    # =========================================================================
    # STANDARD PD: Bandwidth = 0.1 Hz, no filter
    # =========================================================================
    bw_std = 0.1
    omega_bw_std = 2 * np.pi * bw_std
    K_std = I_z * omega_bw_std**2
    P_std = 2 * config.damping_ratio * I_z * omega_bw_std

    controllers['standard_pd'] = signal.TransferFunction([P_std, K_std], [1.0])
    info['standard_pd'] = {'K': K_std, 'P': P_std, 'bandwidth_hz': bw_std, 'type': 'PD'}

    # =========================================================================
    # FILTERED PD: Bandwidth = 0.1 Hz, filter at 10x bandwidth (1 Hz)
    # This preserves phase margin by placing filter well above crossover
    # =========================================================================
    bw_filt = 0.1
    omega_bw_filt = 2 * np.pi * bw_filt
    K_filt = I_z * omega_bw_filt**2
    P_filt = 2 * config.damping_ratio * I_z * omega_bw_filt
    filter_cutoff_filt = 10.0 * bw_filt  # 1 Hz
    tau_filt = 1.0 / (2 * np.pi * filter_cutoff_filt)

    controllers['filtered_pd'] = signal.TransferFunction(
        [K_filt * tau_filt + P_filt, K_filt], [tau_filt, 1.0]
    )
    info['filtered_pd'] = {
        'K': K_filt, 'P': P_filt, 'bandwidth_hz': bw_filt,
        'filter_cutoff_hz': filter_cutoff_filt, 'type': 'Filtered PD'
    }

    # Also create comparison controllers with different filter cutoffs
    for filter_mult, name in [(1.2, 'filtered_pd_low'), (5.0, 'filtered_pd_mid'), (10.0, 'filtered_pd_high')]:
        fc = filter_mult * bw_filt
        tau = 1.0 / (2 * np.pi * fc)
        controllers[name] = signal.TransferFunction([K_filt * tau + P_filt, K_filt], [tau, 1.0])
        info[name] = {
            'K': K_filt, 'P': P_filt, 'bandwidth_hz': bw_filt,
            'filter_cutoff_hz': fc, 'type': 'Filtered PD'
        }

    # =========================================================================
    # AVC: Reduced bandwidth = first_mode/6, filter at 5x bandwidth
    # Low bandwidth stays well below modal frequencies
    # =========================================================================
    bw_avc = first_mode_hz / 6.0  # 0.067 Hz for 0.4 Hz mode
    omega_bw_avc = 2 * np.pi * bw_avc
    K_avc = I_z * omega_bw_avc**2
    P_avc = 2 * config.damping_ratio * I_z * omega_bw_avc
    filter_cutoff_avc = 5.0 * bw_avc
    tau_avc = 1.0 / (2 * np.pi * filter_cutoff_avc)

    controllers['avc'] = signal.TransferFunction(
        [K_avc * tau_avc + P_avc, K_avc], [tau_avc, 1.0]
    )
    info['avc'] = {
        'K': K_avc, 'P': P_avc, 'bandwidth_hz': bw_avc,
        'filter_cutoff_hz': filter_cutoff_avc, 'type': 'AVC (Low-BW Filtered PD)'
    }

    return controllers, info


def analyze_feedback_stability(config: DiagnosticConfig) -> Dict:
    """Comprehensive feedback stability and performance analysis."""
    print("\n" + "="*70)
    print("PART 2: FEEDBACK CONTROLLER STABILITY ANALYSIS")
    print("="*70)

    plant, plant_info = build_plant_transfer_function(config)
    controllers, ctrl_info = build_controllers(config)

    print(f"\nPlant Analysis:")
    print(f"  Poles: {plant_info['poles']}")
    print(f"  Zeros: {plant_info['zeros']}")
    print(f"  Double integrator (poles at s=0): {'Yes' if np.sum(np.abs(plant_info['poles']) < 1e-10) >= 2 else 'No'}")

    # Frequency analysis
    freqs = np.logspace(-3, 2, 2000)
    omega = 2 * np.pi * freqs

    results = {}

    print(f"\nController Analysis:")
    print(f"  Bandwidth target: {config.bandwidth_hz} Hz")
    print(f"  First modal frequency: {config.modal_freqs_hz[0]} Hz")
    print(f"  Ratio (mode/bandwidth): {config.modal_freqs_hz[0]/config.bandwidth_hz:.1f}x")

    _, plant_resp = signal.freqresp(plant, omega)

    for ctrl_name, ctrl_tf in controllers.items():
        print(f"\n  --- {ctrl_name.upper()} ---")
        ci = ctrl_info[ctrl_name]
        print(f"  Type: {ci['type']}")
        print(f"  K={ci['K']:.2f}, P={ci['P']:.2f}")
        if 'filter_cutoff_hz' in ci:
            print(f"  Filter cutoff: {ci['filter_cutoff_hz']:.3f} Hz")

        # Open-loop transfer function L = P * C
        _, ctrl_resp = signal.freqresp(ctrl_tf, omega)
        L = plant_resp * ctrl_resp

        # Sensitivity and complementary sensitivity
        S = 1.0 / (1.0 + L)
        T = L / (1.0 + L)

        # Stability margins
        gm_db, pm_deg, wgc, wpc = compute_margins(L, freqs)

        print(f"  Gain margin: {gm_db:.1f} dB" if np.isfinite(gm_db) else "  Gain margin: inf")
        print(f"  Phase margin: {pm_deg:.1f} deg")
        if wgc is not None:
            print(f"  Gain crossover freq: {wgc:.3f} Hz")
        if wpc is not None:
            print(f"  Phase crossover freq: {wpc:.3f} Hz")

        # Sensitivity at modal frequencies
        print(f"  Sensitivity at modal frequencies:")
        for f_mode in config.modal_freqs_hz:
            idx = np.argmin(np.abs(freqs - f_mode))
            s_db = 20 * np.log10(np.abs(S[idx]) + 1e-12)
            t_db = 20 * np.log10(np.abs(T[idx]) + 1e-12)
            print(f"    f={f_mode} Hz: |S|={s_db:.1f} dB, |T|={t_db:.1f} dB")
            if s_db > 0:
                print(f"      WARNING: |S| > 0 dB means disturbance AMPLIFICATION!")

        # Peak sensitivity (indicates robustness)
        s_peak_db = 20 * np.log10(np.max(np.abs(S)))
        s_peak_freq = freqs[np.argmax(np.abs(S))]
        print(f"  Peak sensitivity: {s_peak_db:.1f} dB at {s_peak_freq:.3f} Hz")
        if s_peak_db > 6:
            print(f"      WARNING: Peak > 6 dB indicates poor robustness!")

        # Check closed-loop stability
        ol_num = np.convolve(plant_info['num'], np.atleast_1d(np.squeeze(ctrl_tf.num)))
        ol_den = np.convolve(plant_info['den'], np.atleast_1d(np.squeeze(ctrl_tf.den)))
        cl_den = np.polyadd(ol_den, ol_num)
        cl_poles = np.roots(cl_den)

        unstable_poles = [p for p in cl_poles if p.real > 1e-10]
        if unstable_poles:
            print(f"  CLOSED-LOOP UNSTABLE! RHP poles: {unstable_poles}")
        else:
            print(f"  Closed-loop: STABLE")

        results[ctrl_name] = {
            'L': L,
            'S': S,
            'T': T,
            'gm_db': gm_db,
            'pm_deg': pm_deg,
            'wgc': wgc,
            'wpc': wpc,
            's_peak_db': s_peak_db,
            'cl_poles': cl_poles,
            'info': ci,
        }

    results['freqs'] = freqs
    results['plant_resp'] = plant_resp
    return results


def compute_margins(L: np.ndarray, freqs: np.ndarray) -> Tuple[float, float, Optional[float], Optional[float]]:
    """Compute gain and phase margins from loop transfer function."""
    mag = np.abs(L)
    phase_deg = np.degrees(np.unwrap(np.angle(L)))

    gm_db = np.inf
    pm_deg = np.inf
    wgc = None  # Gain crossover frequency
    wpc = None  # Phase crossover frequency

    # Phase crossover (phase = -180) for gain margin
    for i in range(len(phase_deg) - 1):
        if phase_deg[i] > -180.0 and phase_deg[i+1] <= -180.0:
            # Interpolate
            f1, f2 = freqs[i], freqs[i+1]
            p1, p2 = phase_deg[i], phase_deg[i+1]
            f_pc = f1 + (f2 - f1) * (-180.0 - p1) / (p2 - p1) if p2 != p1 else f1

            # Interpolate magnitude
            m1, m2 = mag[i], mag[i+1]
            if m1 > 0 and m2 > 0:
                log_m = np.log10(m1) + (np.log10(m2) - np.log10(m1)) * (f_pc - f1) / (f2 - f1) if f2 != f1 else np.log10(m1)
                mag_pc = 10 ** log_m
            else:
                mag_pc = m1

            gm_db = -20 * np.log10(mag_pc + 1e-12)
            wpc = f_pc
            break

    # Gain crossover (|L| = 1) for phase margin
    for i in range(len(mag) - 1):
        if mag[i] > 1.0 and mag[i+1] <= 1.0:
            # Interpolate
            f1, f2 = freqs[i], freqs[i+1]
            m1, m2 = mag[i], mag[i+1]

            if m1 > 0 and m2 > 0:
                log_f = np.log10(f1) + (0.0 - np.log10(m1)) * (np.log10(f2) - np.log10(f1)) / (np.log10(m2) - np.log10(m1)) if np.log10(m2) != np.log10(m1) else np.log10(f1)
                f_gc = 10 ** log_f
            else:
                f_gc = f1

            # Interpolate phase
            p1, p2 = phase_deg[i], phase_deg[i+1]
            phase_gc = p1 + (p2 - p1) * (f_gc - f1) / (f2 - f1) if f2 != f1 else p1

            pm_deg = 180.0 + phase_gc
            wgc = f_gc
            break

    return gm_db, pm_deg, wgc, wpc


# =============================================================================
# PART 3: VIBRATION AMPLIFICATION ANALYSIS
# =============================================================================

def analyze_vibration_amplification(config: DiagnosticConfig, ctrl_results: Dict) -> Dict:
    """Analyze whether feedback amplifies or suppresses vibration."""
    print("\n" + "="*70)
    print("PART 3: VIBRATION AMPLIFICATION ANALYSIS")
    print("="*70)

    freqs = ctrl_results['freqs']
    results = {}

    print("\nDisturbance Rejection Analysis:")
    print("(Negative dB = attenuation, Positive dB = amplification)")
    print()

    # For each controller, check sensitivity at modal frequencies
    for ctrl_name in ['standard_pd', 'filtered_pd', 'avc']:
        if ctrl_name not in ctrl_results:
            continue

        S = ctrl_results[ctrl_name]['S']
        T = ctrl_results[ctrl_name]['T']

        print(f"{ctrl_name.upper()}:")

        # Check broadband behavior
        low_freq_idx = np.argmin(np.abs(freqs - 0.01))
        high_freq_idx = np.argmin(np.abs(freqs - 10.0))

        s_low = 20 * np.log10(np.abs(S[low_freq_idx]) + 1e-12)
        s_high = 20 * np.log10(np.abs(S[high_freq_idx]) + 1e-12)

        print(f"  Low freq (0.01 Hz): |S| = {s_low:.1f} dB")
        print(f"  High freq (10 Hz): |S| = {s_high:.1f} dB")

        # Check at modal frequencies
        amplification_freqs = []
        for f_mode in config.modal_freqs_hz:
            idx = np.argmin(np.abs(freqs - f_mode))
            s_db = 20 * np.log10(np.abs(S[idx]) + 1e-12)
            if s_db > 0:
                amplification_freqs.append((f_mode, s_db))
                print(f"  Mode at {f_mode} Hz: |S| = {s_db:.1f} dB  <-- AMPLIFICATION!")
            else:
                print(f"  Mode at {f_mode} Hz: |S| = {s_db:.1f} dB  (attenuation)")

        # Find frequency range with amplification
        s_mag = np.abs(S)
        amp_mask = s_mag > 1.0
        if np.any(amp_mask):
            amp_freqs = freqs[amp_mask]
            print(f"  Amplification frequency range: {amp_freqs[0]:.3f} - {amp_freqs[-1]:.3f} Hz")
            print(f"  Peak amplification: {20*np.log10(np.max(s_mag)):.1f} dB at {freqs[np.argmax(s_mag)]:.3f} Hz")

        results[ctrl_name] = {
            'amplification_freqs': amplification_freqs,
            'has_amplification': len(amplification_freqs) > 0,
        }
        print()

    return results


# =============================================================================
# PART 4: CONTROLLER REDESIGN RECOMMENDATIONS
# =============================================================================

def recommend_controller_fixes(config: DiagnosticConfig, ctrl_results: Dict) -> Dict:
    """Analyze issues and recommend fixes."""
    print("\n" + "="*70)
    print("PART 4: DIAGNOSIS AND RECOMMENDATIONS")
    print("="*70)

    recommendations = []

    # Issue 1: Bandwidth vs modal frequency ratio
    ratio = config.modal_freqs_hz[0] / config.bandwidth_hz
    print(f"\n1. BANDWIDTH ANALYSIS:")
    print(f"   Control bandwidth: {config.bandwidth_hz} Hz")
    print(f"   First modal frequency: {config.modal_freqs_hz[0]} Hz")
    print(f"   Ratio: {ratio:.1f}x")

    if ratio < 4:
        print(f"   ISSUE: Ratio < 4 means controller has significant gain at modal frequencies")
        print(f"   This can cause vibration amplification through the sensitivity function")
        recommendations.append({
            'issue': 'Bandwidth too close to modal frequencies',
            'fix': f'Reduce bandwidth to < {config.modal_freqs_hz[0]/4:.3f} Hz',
        })
    else:
        print(f"   OK: Sufficient separation between bandwidth and modes")

    # Issue 2: Filtered PD filter cutoff
    print(f"\n2. FILTER CUTOFF ANALYSIS:")
    for name in ['filtered_pd', 'filtered_pd_low', 'filtered_pd_mid', 'filtered_pd_high']:
        if name in ctrl_results:
            info = ctrl_results[name]['info']
            if 'filter_cutoff_hz' in info:
                fc = info['filter_cutoff_hz']
                pm = ctrl_results[name]['pm_deg']
                print(f"   {name}: cutoff={fc:.3f} Hz, PM={pm:.1f} deg")

    # Issue 3: Phase margin comparison
    print(f"\n3. PHASE MARGIN COMPARISON:")
    pm_std = ctrl_results.get('standard_pd', {}).get('pm_deg', 0)
    pm_filt = ctrl_results.get('filtered_pd', {}).get('pm_deg', 0)
    pm_avc = ctrl_results.get('avc', {}).get('pm_deg', 0)

    print(f"   Standard PD: {pm_std:.1f} deg")
    print(f"   Filtered PD: {pm_filt:.1f} deg")
    print(f"   AVC: {pm_avc:.1f} deg")

    if pm_filt < pm_std - 20:
        print(f"   ISSUE: Filtered PD loses {pm_std - pm_filt:.0f} deg phase margin!")
        print(f"   The derivative filter adds phase lag near crossover")
        recommendations.append({
            'issue': 'Filter causes excessive phase lag',
            'fix': 'Increase filter cutoff frequency or reduce derivative gain',
        })

    # Issue 4: PPF in AVC
    if pm_avc < pm_filt:
        print(f"\n   ISSUE: AVC has less phase margin than Filtered PD!")
        print(f"   PPF compensators may be adding phase lag instead of lead")
        recommendations.append({
            'issue': 'PPF compensators reduce phase margin',
            'fix': 'Redesign PPF with proper phase lead at modal frequencies',
        })

    # Recommended new design
    print(f"\n4. RECOMMENDED CONTROLLER PARAMETERS:")

    # Conservative bandwidth
    new_bw = config.modal_freqs_hz[0] / 6  # 6x separation
    new_K = config.inertia[2, 2] * (2 * np.pi * new_bw)**2
    new_P = 2 * config.damping_ratio * config.inertia[2, 2] * (2 * np.pi * new_bw)
    new_fc = 1.5 * new_bw  # Filter just above bandwidth

    print(f"   New bandwidth: {new_bw:.4f} Hz (was {config.bandwidth_hz} Hz)")
    print(f"   New K: {new_K:.2f} (was {config.inertia[2, 2] * (2*np.pi*config.bandwidth_hz)**2:.2f})")
    print(f"   New P: {new_P:.2f} (was {2*config.damping_ratio*config.inertia[2, 2]*(2*np.pi*config.bandwidth_hz):.2f})")
    print(f"   New filter cutoff: {new_fc:.4f} Hz")

    return {
        'recommendations': recommendations,
        'new_params': {
            'bandwidth_hz': new_bw,
            'K': new_K,
            'P': new_P,
            'filter_cutoff_hz': new_fc,
        }
    }


# =============================================================================
# PART 5: TIME-DOMAIN SIMULATION
# =============================================================================

def simulate_closed_loop_response(config: DiagnosticConfig) -> Dict:
    """Simulate closed-loop response to verify behavior."""
    print("\n" + "="*70)
    print("PART 5: TIME-DOMAIN CLOSED-LOOP SIMULATION")
    print("="*70)

    I_z = config.inertia[2, 2]
    theta_target = np.radians(config.slew_angle_deg)

    # Controller gains
    omega_bw = 2 * np.pi * config.bandwidth_hz
    K = I_z * omega_bw**2
    P = 2 * config.damping_ratio * I_z * omega_bw

    # Modal parameters
    omega_n1 = 2 * np.pi * config.modal_freqs_hz[0]
    zeta1 = config.modal_damping[0]
    gain1 = config.modal_gains[0]

    omega_n2 = 2 * np.pi * config.modal_freqs_hz[1]
    zeta2 = config.modal_damping[1]
    gain2 = config.modal_gains[1]

    # State vector: [theta, omega, q1, q1_dot, q2, q2_dot]
    def dynamics(y, t, torque_func, ctrl_type):
        theta, omega, q1, q1_dot, q2, q2_dot = y

        # Error
        theta_err = theta_target - theta
        omega_err = 0 - omega  # Target rate is zero at final position

        # Controller torque
        if ctrl_type == 'feedforward_only':
            tau_ctrl = torque_func(t)
        elif ctrl_type == 'feedback_only':
            tau_ctrl = K * theta_err + P * omega_err
        elif ctrl_type == 'combined':
            tau_ff = torque_func(t)
            tau_fb = K * theta_err + P * omega_err
            tau_ctrl = tau_ff + tau_fb
        else:
            tau_ctrl = 0.0

        # Rigid body dynamics (simplified - ignoring modal coupling to attitude)
        alpha = tau_ctrl / I_z

        # Modal dynamics excited by torque
        q1_ddot = gain1 * tau_ctrl - 2*zeta1*omega_n1*q1_dot - omega_n1**2*q1
        q2_ddot = gain2 * tau_ctrl - 2*zeta2*omega_n2*q2_dot - omega_n2**2*q2

        return [omega, alpha, q1_dot, q1_ddot, q2_dot, q2_ddot]

    # Feedforward torque function
    duration = config.slew_duration_s
    t_half = duration / 2
    alpha_max = 4 * theta_target / duration**2

    def ff_torque(t):
        if t < 0 or t > duration:
            return 0.0
        if t <= t_half:
            return I_z * alpha_max
        else:
            return -I_z * alpha_max

    # Simulate
    dt = 0.01
    t_sim = np.arange(0, 90, dt)  # 90 seconds total
    y0 = [0, 0, 0, 0, 0, 0]

    results = {}

    for ctrl_type in ['feedforward_only', 'feedback_only', 'combined']:
        print(f"\nSimulating {ctrl_type}...")
        y = odeint(dynamics, y0, t_sim, args=(ff_torque, ctrl_type))

        theta = y[:, 0]
        omega = y[:, 1]
        q1 = y[:, 2]
        q2 = y[:, 4]

        final_angle_deg = np.degrees(theta[-1])
        final_error_deg = config.slew_angle_deg - final_angle_deg

        # Modal vibration
        total_vib = np.sqrt(q1**2 + q2**2)
        post_maneuver_mask = t_sim > duration
        if np.any(post_maneuver_mask):
            residual_vib_rms = np.sqrt(np.mean(total_vib[post_maneuver_mask]**2)) * 1000  # mm
            residual_vib_peak = np.max(np.abs(total_vib[post_maneuver_mask])) * 1000
        else:
            residual_vib_rms = 0
            residual_vib_peak = 0

        print(f"  Final angle: {final_angle_deg:.2f} deg (error: {final_error_deg:.2f} deg)")
        print(f"  Residual vibration: RMS={residual_vib_rms:.4f} mm, Peak={residual_vib_peak:.4f} mm")

        results[ctrl_type] = {
            'time': t_sim,
            'theta': theta,
            'omega': omega,
            'q1': q1,
            'q2': q2,
            'total_vib': total_vib,
            'final_angle_deg': final_angle_deg,
            'final_error_deg': final_error_deg,
            'residual_vib_rms_mm': residual_vib_rms,
            'residual_vib_peak_mm': residual_vib_peak,
        }

    return results


# =============================================================================
# PART 6: GENERATE PLOTS
# =============================================================================

def generate_diagnostic_plots(config: DiagnosticConfig, ctrl_results: Dict, sim_results: Dict):
    """Generate diagnostic plots."""
    print("\n" + "="*70)
    print("PART 6: GENERATING DIAGNOSTIC PLOTS")
    print("="*70)

    fig, axes = plt.subplots(3, 2, figsize=(14, 12))

    freqs = ctrl_results['freqs']

    # Plot 1: Open-loop Bode magnitude
    ax = axes[0, 0]
    for ctrl_name in ['standard_pd', 'filtered_pd', 'avc']:
        if ctrl_name in ctrl_results:
            L = ctrl_results[ctrl_name]['L']
            ax.semilogx(freqs, 20*np.log10(np.abs(L) + 1e-12), label=ctrl_name, linewidth=1.5)
    ax.axhline(0, color='k', linestyle='--', alpha=0.5)
    for f_mode in config.modal_freqs_hz:
        ax.axvline(f_mode, color='r', linestyle=':', alpha=0.7)
    ax.set_xlabel('Frequency (Hz)')
    ax.set_ylabel('|L| (dB)')
    ax.set_title('Open-Loop Magnitude')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xlim([0.001, 100])

    # Plot 2: Sensitivity function
    ax = axes[0, 1]
    for ctrl_name in ['standard_pd', 'filtered_pd', 'avc']:
        if ctrl_name in ctrl_results:
            S = ctrl_results[ctrl_name]['S']
            ax.semilogx(freqs, 20*np.log10(np.abs(S) + 1e-12), label=ctrl_name, linewidth=1.5)
    ax.axhline(0, color='k', linestyle='--', alpha=0.5, label='Amplification threshold')
    ax.axhline(6, color='r', linestyle='--', alpha=0.5, label='6 dB limit')
    for f_mode in config.modal_freqs_hz:
        ax.axvline(f_mode, color='r', linestyle=':', alpha=0.7)
    ax.set_xlabel('Frequency (Hz)')
    ax.set_ylabel('|S| (dB)')
    ax.set_title('Sensitivity Function (|S| > 0 dB = amplification)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xlim([0.001, 100])
    ax.set_ylim([-60, 20])

    # Plot 3: Complementary sensitivity
    ax = axes[1, 0]
    for ctrl_name in ['standard_pd', 'filtered_pd', 'avc']:
        if ctrl_name in ctrl_results:
            T = ctrl_results[ctrl_name]['T']
            ax.semilogx(freqs, 20*np.log10(np.abs(T) + 1e-12), label=ctrl_name, linewidth=1.5)
    ax.axhline(0, color='k', linestyle='--', alpha=0.5)
    for f_mode in config.modal_freqs_hz:
        ax.axvline(f_mode, color='r', linestyle=':', alpha=0.7)
    ax.set_xlabel('Frequency (Hz)')
    ax.set_ylabel('|T| (dB)')
    ax.set_title('Complementary Sensitivity')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xlim([0.001, 100])

    # Plot 4: Nyquist
    ax = axes[1, 1]
    for ctrl_name in ['standard_pd', 'filtered_pd', 'avc']:
        if ctrl_name in ctrl_results:
            L = ctrl_results[ctrl_name]['L']
            ax.plot(L.real, L.imag, label=ctrl_name, linewidth=1.5)
    ax.plot(-1, 0, 'rx', markersize=10, markeredgewidth=2, label='Critical point')
    ax.set_xlabel('Real')
    ax.set_ylabel('Imaginary')
    ax.set_title('Nyquist Diagram')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')
    ax.set_xlim([-3, 1])
    ax.set_ylim([-2, 2])

    # Plot 5: Time-domain angle
    ax = axes[2, 0]
    for ctrl_type, data in sim_results.items():
        ax.plot(data['time'], np.degrees(data['theta']), label=ctrl_type, linewidth=1.5)
    ax.axhline(config.slew_angle_deg, color='k', linestyle='--', alpha=0.5, label='Target')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Angle (deg)')
    ax.set_title('Slew Angle Response')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 6: Time-domain vibration
    ax = axes[2, 1]
    for ctrl_type, data in sim_results.items():
        ax.plot(data['time'], data['total_vib'] * 1000, label=ctrl_type, linewidth=1.5)
    ax.axvline(config.slew_duration_s, color='k', linestyle='--', alpha=0.5, label='Maneuver end')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Modal Vibration (mm)')
    ax.set_title('Modal Vibration Response')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('diagnostic_control_analysis.png', dpi=150, bbox_inches='tight')
    print("\nSaved: diagnostic_control_analysis.png")
    plt.close()


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("="*70)
    print("COMPREHENSIVE CONTROL SYSTEM DIAGNOSTIC ANALYSIS")
    print("="*70)

    config = get_config()

    print(f"\nConfiguration:")
    print(f"  Inertia (Z-axis): {config.inertia[2,2]} kg.m^2")
    print(f"  Modal frequencies: {config.modal_freqs_hz} Hz")
    print(f"  Modal damping: {config.modal_damping}")
    print(f"  Modal gains: {config.modal_gains}")
    print(f"  Slew angle: {config.slew_angle_deg} deg")
    print(f"  Slew duration: {config.slew_duration_s} s")
    print(f"  Control bandwidth: {config.bandwidth_hz} Hz")

    # Run diagnostics
    ff_results = verify_feedforward_slew(config)
    ctrl_results = analyze_feedback_stability(config)
    vib_results = analyze_vibration_amplification(config, ctrl_results)
    recommendations = recommend_controller_fixes(config, ctrl_results)
    sim_results = simulate_closed_loop_response(config)

    # Generate plots
    generate_diagnostic_plots(config, ctrl_results, sim_results)

    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)

    print(f"\n1. Feedforward slew: {'PASS' if ff_results['success'] else 'FAIL'}")
    print(f"   Achieved {ff_results['final_angle_deg']:.2f} deg (target: {config.slew_angle_deg} deg)")

    print(f"\n2. Feedback vibration amplification:")
    for ctrl_name, vr in vib_results.items():
        status = "YES - AMPLIFIES" if vr['has_amplification'] else "NO - Attenuates"
        print(f"   {ctrl_name}: {status}")

    print(f"\n3. Phase margins:")
    for ctrl_name in ['standard_pd', 'filtered_pd', 'avc']:
        if ctrl_name in ctrl_results:
            pm = ctrl_results[ctrl_name]['pm_deg']
            print(f"   {ctrl_name}: {pm:.1f} deg")

    print(f"\n4. Recommendations:")
    for i, rec in enumerate(recommendations['recommendations'], 1):
        print(f"   {i}. Issue: {rec['issue']}")
        print(f"      Fix: {rec['fix']}")

    print(f"\n5. Suggested new parameters:")
    new = recommendations['new_params']
    print(f"   Bandwidth: {new['bandwidth_hz']:.4f} Hz")
    print(f"   K: {new['K']:.2f}")
    print(f"   P: {new['P']:.2f}")
    print(f"   Filter cutoff: {new['filter_cutoff_hz']:.4f} Hz")

    return {
        'config': config,
        'ff_results': ff_results,
        'ctrl_results': ctrl_results,
        'vib_results': vib_results,
        'recommendations': recommendations,
        'sim_results': sim_results,
    }


if __name__ == "__main__":
    results = main()
