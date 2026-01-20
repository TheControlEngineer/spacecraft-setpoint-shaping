"""
COMPREHENSIVE SPACECRAFT CONTROL SYSTEM INVESTIGATION

Requirements:
1. Check if feedforward part can slew spacecraft to target
2. Check if feedback or feedforward is exciting structural resonance
3. Investigate modal gain and see if that is the problem
4. Target: ≤0.1 degrees pointing error, >62 degrees phase margin, no mode excitation

Author: GNC Analysis
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from dataclasses import dataclass
from typing import Dict, List, Tuple

# Import from project files
from spacecraft_properties import HUB_INERTIA, compute_effective_inertia, FLEX_MODE_MASS, FLEX_MODE_LOCATIONS
from mission_simulation import default_config, _compute_bang_bang_trajectory, _zvd_shaper_params, _apply_input_shaper


@dataclass
class InvestigationResults:
    """Store all investigation results."""
    feedforward_ok: bool
    feedback_excites_modes: Dict[str, bool]
    feedforward_excites_modes: Dict[str, bool]
    modal_gain_problem: bool
    phase_margins: Dict[str, float]
    pointing_errors: Dict[str, float]
    recommendations: List[str]


def compute_actual_modal_gains():
    """
    INVESTIGATE MODAL GAINS
    
    Modal gain = coupling factor between torque and modal displacement
    For a spring-mass at distance r from rotation axis:
        gain = r / (J * omega_n^2)
    
    This determines how much the mode is excited by control torque.
    """
    print("=" * 80)
    print("MODAL GAIN INVESTIGATION")
    print("=" * 80)
    
    # Spacecraft parameters
    J_z = HUB_INERTIA[2, 2]  # Inertia about Z-axis
    modal_freqs_hz = [0.4, 1.3]
    modal_mass = FLEX_MODE_MASS  # 5 kg per mode point
    
    print(f"\nSpacecraft Inertia (Z-axis): J_z = {J_z} kg·m²")
    print(f"Modal mass per location: m = {modal_mass} kg")
    print(f"Modal frequencies: {modal_freqs_hz} Hz")
    
    # Modal locations (from spacecraft_properties.py)
    locations = list(FLEX_MODE_LOCATIONS.values())
    print(f"\nModal locations:")
    for name, loc in FLEX_MODE_LOCATIONS.items():
        print(f"  {name}: {loc} m")
    
    # Compute theoretical modal gains
    print("\n" + "-" * 60)
    print("THEORETICAL MODAL GAIN CALCULATION")
    print("-" * 60)
    
    # For a rotating body, a point mass at distance r from rotation axis
    # experiences base excitation: acceleration = r * alpha (angular accel)
    # The modal displacement response is: x = (r * alpha) / omega_n^2 = (r * tau) / (J * omega_n^2)
    # So modal gain = r / (J * omega_n^2)
    
    # But there's ALSO the reciprocal effect: modal displacement causes torque
    # For collocated control: delta_tau = k * x where k ~ m * omega_n^2
    # The overall coupling gain (torque to modal displacement and back) is:
    #   modal_gain = (m * r) / J
    
    for i, (f_mode, r_mode) in enumerate(zip(modal_freqs_hz, [3.5, 4.5])):  # Y-positions
        omega_n = 2 * np.pi * f_mode
        
        # Theoretical gain (displacement per unit torque)
        gain_disp = r_mode / (J_z * omega_n**2)
        
        # Coupling factor (for transfer function)
        coupling = modal_mass * r_mode / J_z
        
        print(f"\nMode {i+1} ({f_mode} Hz):")
        print(f"  Distance from Z-axis: r = {r_mode} m")
        print(f"  Angular frequency: ω_n = {omega_n:.2f} rad/s")
        print(f"  Displacement gain: r/(J·ω²) = {gain_disp:.6f} m/(N·m)")
        print(f"  Coupling factor: m·r/J = {coupling:.6f}")
    
    # Compare with values in default_config
    config = default_config()
    print("\n" + "-" * 60)
    print("COMPARISON WITH CONFIGURATION VALUES")
    print("-" * 60)
    print(f"\nCurrent modal_gains in config: {config.modal_gains}")
    print(f"Current control_modal_gains in config: {config.control_modal_gains}")
    
    # Check if modal gains are realistic
    theoretical_gains = []
    for i, f_mode in enumerate(modal_freqs_hz):
        omega_n = 2 * np.pi * f_mode
        r_mode = [3.5, 4.5][i]  # Approximate distance
        gain = r_mode / (J_z * omega_n**2)
        theoretical_gains.append(gain)
        
    print(f"\nTheoretical modal gains: {theoretical_gains}")
    
    # The issue: modal_gains=[0.0015, 0.0008] are for small vibration
    # control_modal_gains=[0.15, 0.08] are 100x larger for analysis
    # This inconsistency may be causing problems!
    
    if config.control_modal_gains and config.modal_gains:
        ratio = [c/m if m > 0 else 0 for c, m in zip(config.control_modal_gains, config.modal_gains)]
        print(f"\nRatio of control_modal_gains to modal_gains: {ratio}")
        if any(r > 10 for r in ratio):
            print("⚠ WARNING: control_modal_gains are much larger than modal_gains!")
            print("   This means control analysis sees different modes than simulation!")
    
    return theoretical_gains


def analyze_feedforward_slew():
    """
    REQUIREMENT 1: Check if feedforward can slew to target
    """
    print("\n" + "=" * 80)
    print("FEEDFORWARD SLEW CAPABILITY CHECK")
    print("=" * 80)
    
    config = default_config()
    theta_final = np.radians(config.slew_angle_deg)  # 180 degrees in radians
    duration = config.slew_duration_s  # 30 seconds
    
    # Compute bang-bang trajectory
    t, theta, omega, alpha = _compute_bang_bang_trajectory(theta_final, duration, dt=0.001, settling_time=30.0)
    
    # Get inertia
    J_z = config.inertia[2, 2]
    J_eff = compute_effective_inertia()[2, 2]  # With flex masses
    
    # Compute required torque
    torque_max = J_eff * np.max(np.abs(alpha))
    
    print(f"\nSlew Parameters:")
    print(f"  Target angle: {config.slew_angle_deg} degrees ({theta_final:.4f} rad)")
    print(f"  Duration: {duration} s")
    print(f"  Hub inertia (Z): {J_z:.1f} kg·m²")
    print(f"  Effective inertia (Z): {J_eff:.1f} kg·m²")
    
    print(f"\nTrajectory Results:")
    print(f"  Final angle achieved: {np.degrees(theta[-1]):.2f} degrees")
    print(f"  Peak angular rate: {np.degrees(np.max(np.abs(omega))):.2f} deg/s")
    print(f"  Peak angular acceleration: {np.max(np.abs(alpha)):.4f} rad/s²")
    print(f"  Peak torque required: {torque_max:.2f} N·m")
    
    # Check with RW limits (from spacecraft_model.py)
    rw_max_torque = 70.0  # Nm per wheel
    print(f"\n  RW max torque: {rw_max_torque} N·m/wheel")
    
    angle_error = abs(np.degrees(theta[-1]) - config.slew_angle_deg)
    slew_ok = angle_error < 1.0 and torque_max < rw_max_torque
    
    print(f"\n✓ Feedforward can reach target: {slew_ok}")
    if not slew_ok:
        print(f"  ✗ Angle error: {angle_error:.3f} degrees")
        print(f"  ✗ Torque exceeds limit: {torque_max > rw_max_torque}")
    
    return slew_ok, theta, omega, alpha, t


def analyze_feedforward_mode_excitation():
    """
    REQUIREMENT 2a: Check if feedforward excites structural modes
    """
    print("\n" + "=" * 80)
    print("FEEDFORWARD MODE EXCITATION ANALYSIS")
    print("=" * 80)
    
    config = default_config()
    theta_final = np.radians(config.slew_angle_deg)
    duration = config.slew_duration_s
    
    # Compute trajectories for different methods
    methods = {}
    
    # 1. Unshaped bang-bang
    t_bb, theta_bb, omega_bb, alpha_bb = _compute_bang_bang_trajectory(theta_final, duration, dt=0.001, settling_time=30.0)
    torque_bb = config.inertia[2, 2] * alpha_bb
    methods['unshaped'] = {'time': t_bb, 'torque': torque_bb, 'alpha': alpha_bb}
    
    # 2. ZVD shaped
    amps_zvd, delays_zvd = _zvd_shaper_params(config.modal_freqs_hz[0], config.modal_damping[0])
    t_zvd, alpha_zvd = _apply_input_shaper(t_bb[:len(alpha_bb)], alpha_bb, amps_zvd, delays_zvd)
    torque_zvd = config.inertia[2, 2] * alpha_zvd
    methods['zvd'] = {'time': t_zvd, 'torque': torque_zvd, 'alpha': alpha_zvd}
    
    print(f"\nFeedforward Methods:")
    print(f"  Unshaped duration: {t_bb[-1]:.1f} s")
    print(f"  ZVD shaper duration: {t_zvd[-1]:.1f} s (extended by {t_zvd[-1] - t_bb[-1]:.2f} s)")
    
    # Compute FFT of torque to find energy at modal frequencies
    excitation = {}
    modal_freqs = config.modal_freqs_hz
    
    for name, data in methods.items():
        t = data['time']
        torque = data['torque']
        
        # Zero-pad for better frequency resolution
        n = len(torque)
        dt = t[1] - t[0] if len(t) > 1 else 0.001
        fs = 1.0 / dt
        
        # FFT
        fft_vals = np.fft.fft(torque, n=4*n)  # Zero-pad for resolution
        freqs_fft = np.fft.fftfreq(4*n, dt)
        
        # Only positive frequencies
        pos_mask = freqs_fft > 0
        freqs_pos = freqs_fft[pos_mask]
        mag = np.abs(fft_vals[pos_mask])
        
        # Find energy at modal frequencies
        mode_energy = []
        for f_mode in modal_freqs:
            idx = np.argmin(np.abs(freqs_pos - f_mode))
            # Sum energy in small band around mode
            band_mask = np.abs(freqs_pos - f_mode) < 0.05
            energy = np.sum(mag[band_mask]**2)
            mode_energy.append(energy)
        
        excitation[name] = {
            'mode_energy': mode_energy,
            'total_energy': np.sum(mag**2)
        }
        
        print(f"\n{name.upper()}:")
        for i, f_mode in enumerate(modal_freqs):
            rel_energy = mode_energy[i] / excitation[name]['total_energy'] * 100
            print(f"  Mode {i+1} ({f_mode} Hz): Energy = {mode_energy[i]:.2e} ({rel_energy:.2f}% of total)")
    
    # Compare ZVD vs unshaped
    print("\n" + "-" * 60)
    print("ZVD SHAPING EFFECTIVENESS:")
    print("-" * 60)
    
    ff_excites = {}
    for i, f_mode in enumerate(modal_freqs):
        reduction = (1 - excitation['zvd']['mode_energy'][i] / excitation['unshaped']['mode_energy'][i]) * 100
        print(f"  Mode {i+1} ({f_mode} Hz): {reduction:.1f}% reduction with ZVD")
        ff_excites[f'unshaped_mode{i+1}'] = excitation['unshaped']['mode_energy'][i] > 1e-6
        ff_excites[f'zvd_mode{i+1}'] = excitation['zvd']['mode_energy'][i] > 1e-6
    
    return excitation, ff_excites


def analyze_feedback_mode_excitation():
    """
    REQUIREMENT 2b: Check if feedback excites structural modes
    
    Key analysis: Look at complementary sensitivity T(jω) at modal frequencies.
    High |T(jω)| means reference commands at that frequency are NOT attenuated,
    which means the feedback controller CAN excite modes if there's any
    disturbance or noise at those frequencies.
    
    Also look at controller gain |C(jω)| - high gain at modal frequencies
    means the controller amplifies measurement noise at those frequencies,
    which then excites the modes.
    """
    print("\n" + "=" * 80)
    print("FEEDBACK MODE EXCITATION ANALYSIS")
    print("=" * 80)
    
    config = default_config()
    
    # Build flexible plant
    axis = 2
    I = config.inertia[axis, axis]
    modal_freqs = config.modal_freqs_hz
    modal_damping = config.modal_damping
    modal_gains = config.control_modal_gains or config.modal_gains  # Use control gains for analysis
    
    print(f"\nUsing modal gains for analysis: {modal_gains}")
    print(f"(Note: config.modal_gains = {config.modal_gains})")
    
    # Frequency range
    freqs = np.logspace(-2, 1, 1000)  # 0.01 to 10 Hz
    omega = 2 * np.pi * freqs
    
    # Build flexible plant TF
    # G(s) = 1/(I*s^2) + Sum k_i/(s^2 + 2*zeta*omega*s + omega^2)
    def build_plant():
        rigid_num = np.array([1.0])
        rigid_den = np.array([I, 0.0, 0.0])
        
        current_num = rigid_num
        current_den = rigid_den
        
        for f_mode, zeta, gain in zip(modal_freqs, modal_damping, modal_gains):
            omega_n = 2 * np.pi * f_mode
            mode_num = np.array([gain / I])
            mode_den = np.array([1.0, 2*zeta*omega_n, omega_n**2])
            
            term1 = np.convolve(current_num, mode_den)
            term2 = np.convolve(mode_num, current_den)
            
            if len(term1) > len(term2):
                term2 = np.pad(term2, (len(term1) - len(term2), 0))
            elif len(term2) > len(term1):
                term1 = np.pad(term1, (len(term2) - len(term1), 0))
            
            current_num = term1 + term2
            current_den = np.convolve(current_den, mode_den)
        
        return signal.TransferFunction(current_num, current_den)
    
    plant = build_plant()
    _, G = signal.freqresp(plant, omega)
    
    # Test different controller configurations
    controllers = {}
    
    # Design parameters
    first_mode = min(modal_freqs)
    
    # 1. High bandwidth PD (BAD - will excite modes)
    bw_high = 0.1  # 0.1 Hz - too close to first mode!
    omega_n_high = 2 * np.pi * bw_high
    K_high = I * omega_n_high**2
    P_high = 2 * 0.7 * I * omega_n_high
    C_high = K_high + 1j * omega * P_high
    controllers['PD (bw=0.1Hz)'] = C_high
    
    # 2. Low bandwidth PD (BETTER)
    bw_low = first_mode / 6.0  # ~0.067 Hz
    omega_n_low = 2 * np.pi * bw_low
    K_low = I * omega_n_low**2
    P_low = 2 * 0.7 * I * omega_n_low
    C_low = K_low + 1j * omega * P_low
    controllers['PD (bw=first_mode/6)'] = C_low
    
    # 3. Filtered PD (derivative filtered below first mode)
    bw_filt = first_mode / 6.0
    omega_n_filt = 2 * np.pi * bw_filt
    K_filt = I * omega_n_filt**2
    P_filt = 2 * 0.7 * I * omega_n_filt
    filter_cutoff = first_mode / 2.0  # 0.2 Hz
    tau = 1 / (2 * np.pi * filter_cutoff)
    # C(s) = K + P*s/(tau*s + 1) = K + P*jw/(tau*jw + 1)
    C_filt = K_filt + P_filt * 1j * omega / (tau * 1j * omega + 1)
    controllers['Filtered PD (fc=0.2Hz)'] = C_filt
    
    print("\n" + "-" * 60)
    print("CONTROLLER CONFIGURATIONS")
    print("-" * 60)
    print(f"\nFirst modal frequency: {first_mode} Hz")
    print(f"\nPD (high bw): K={K_high:.1f}, P={P_high:.1f}, bw=0.1 Hz")
    print(f"PD (low bw):  K={K_low:.1f}, P={P_low:.1f}, bw={bw_low:.4f} Hz")
    print(f"Filtered PD:  K={K_filt:.1f}, P={P_filt:.1f}, bw={bw_filt:.4f} Hz, filter={filter_cutoff:.3f} Hz")
    
    # Analyze each controller
    fb_excites = {}
    results = {}
    
    print("\n" + "-" * 60)
    print("OPEN-LOOP AND SENSITIVITY ANALYSIS")
    print("-" * 60)
    
    for name, C in controllers.items():
        # Open-loop: L = G * C
        L = G * C
        
        # Sensitivity: S = 1/(1+L)
        S = 1 / (1 + L)
        
        # Complementary sensitivity: T = L/(1+L)
        T = L / (1 + L)
        
        # Controller gain at modal frequencies
        S_db = 20 * np.log10(np.abs(S))
        T_db = 20 * np.log10(np.abs(T))
        C_db = 20 * np.log10(np.abs(C))
        
        # Find values at modal frequencies
        print(f"\n{name}:")
        
        excites = False
        for i, f_mode in enumerate(modal_freqs):
            idx = np.argmin(np.abs(freqs - f_mode))
            
            s_val = S_db[idx]
            t_val = T_db[idx]
            c_val = C_db[idx]
            
            print(f"  Mode {i+1} ({f_mode} Hz):")
            print(f"    |S(jω)| = {s_val:+.2f} dB  (disturbance rejection)")
            print(f"    |T(jω)| = {t_val:+.2f} dB  (command following)")
            print(f"    |C(jω)| = {c_val:+.2f} dB  (controller gain)")
            
            # Check for problems
            if s_val > 0:
                print(f"    ⚠ S > 0 dB: AMPLIFIES disturbances at this mode!")
                excites = True
            if t_val > -3:
                print(f"    ⚠ T > -3 dB: Commands can EXCITE this mode!")
                excites = True
            if c_val > 60:
                print(f"    ⚠ |C| > 60 dB: High controller gain may excite mode via noise!")
                excites = True
        
        # Stability margins
        phase = np.angle(L, deg=True)
        mag = np.abs(L)
        
        # Find gain crossover (|L| = 1)
        gc_idx = np.argmin(np.abs(mag - 1))
        pm = 180 + phase[gc_idx]
        
        # Find phase crossover (phase = -180)
        pc_idx = np.argmin(np.abs(phase + 180))
        gm = -20 * np.log10(mag[pc_idx]) if mag[pc_idx] > 0 else np.inf
        
        print(f"  Stability margins:")
        print(f"    Phase margin: {pm:.1f}°")
        print(f"    Gain margin: {gm:.1f} dB")
        
        if pm < 62:
            print(f"    ✗ PM < 62° requirement!")
        else:
            print(f"    ✓ PM > 62° OK")
        
        fb_excites[name] = excites
        results[name] = {
            'S': S, 'T': T, 'C': C, 'L': L,
            'pm': pm, 'gm': gm
        }
    
    return results, fb_excites, freqs


def compute_pointing_error():
    """
    REQUIREMENT 4: Compute pointing error
    
    Pointing error comes from:
    1. Feedforward tracking error during slew
    2. Residual vibration after slew (depends on modal excitation)
    3. Feedback settling error
    """
    print("\n" + "=" * 80)
    print("POINTING ERROR ANALYSIS")
    print("=" * 80)
    
    config = default_config()
    
    # For a bang-bang trajectory with feedback, pointing error is primarily
    # from the transient response during settling.
    
    # Theoretical settling with PD control:
    # For critically damped (zeta=0.7), 2% settling time = 4/(zeta*wn)
    
    first_mode = min(config.modal_freqs_hz)
    bw = first_mode / 6.0
    omega_n = 2 * np.pi * bw
    zeta = 0.7
    
    t_settle_2pct = 4 / (zeta * omega_n)
    t_settle_01deg = t_settle_2pct * np.log(0.02 / (0.1/180)) / np.log(0.02)  # Scale for 0.1 deg
    
    print(f"\nClosed-loop natural frequency: {omega_n/(2*np.pi):.4f} Hz")
    print(f"Damping ratio: {zeta}")
    print(f"2% settling time: {t_settle_2pct:.1f} s")
    
    # The actual pointing error depends on:
    # 1. How well feedforward cancels the reference
    # 2. How well feedback rejects disturbances (S(jω))
    # 3. Residual modal vibration coupling to attitude
    
    # For a well-designed system:
    # - Feedforward tracks accurately during slew
    # - Feedback settles to < 0.1 deg error after settling time
    # - Modal vibration contributes additional pointing jitter
    
    # Modal vibration contribution to pointing:
    # theta_vibration ~ modal_gain * modal_displacement
    # For our gains: [0.0015, 0.0008] m/(N·m)
    # With ~10 mm peak vibration → ~0.015 mrad → ~0.001 deg
    
    modal_gains = config.modal_gains
    typical_vibration_mm = 10  # mm
    pointing_from_vibration = []
    
    print(f"\nModal vibration contribution to pointing error:")
    for i, gain in enumerate(modal_gains):
        # Vibration in m
        vib_m = typical_vibration_mm / 1000
        # This is coupling to pointing: gain is in units that relate torque to displacement
        # The effect on pointing is approximately: delta_theta ~ vib_m / arm_length
        arm_length = [3.5, 4.5][i]  # Distance from rotation axis
        pointing_rad = vib_m / arm_length
        pointing_deg = np.degrees(pointing_rad)
        pointing_from_vibration.append(pointing_deg)
        print(f"  Mode {i+1}: {typical_vibration_mm} mm vibration → {pointing_deg*1000:.3f} mdeg pointing")
    
    total_vibration_pointing = np.sqrt(sum(p**2 for p in pointing_from_vibration))
    
    print(f"\nTotal vibration contribution: {total_vibration_pointing*1000:.3f} mdeg ({total_vibration_pointing*60:.2f} arcsec)")
    
    # Check if we can meet 0.1 deg requirement
    print("\n" + "-" * 60)
    print("REQUIREMENT CHECK: Pointing error ≤ 0.1 degrees")
    print("-" * 60)
    
    # Assuming feedback settles well, vibration is the main contributor
    if total_vibration_pointing < 0.1:
        print(f"✓ Vibration contribution ({total_vibration_pointing*1000:.1f} mdeg) << 0.1 deg limit")
        print("  With proper feedback, 0.1 deg pointing is ACHIEVABLE")
    else:
        print(f"✗ Vibration contribution ({total_vibration_pointing*1000:.1f} mdeg) may be too high")
    
    return {
        'vibration_contribution_deg': total_vibration_pointing,
        'settling_time_s': t_settle_2pct
    }


def generate_recommendations(fb_results, fb_excites, ff_excites, pointing):
    """Generate recommendations based on analysis."""
    print("\n" + "=" * 80)
    print("RECOMMENDATIONS")
    print("=" * 80)
    
    recommendations = []
    
    # 1. Check phase margins
    print("\n1. PHASE MARGIN REQUIREMENT (> 62°):")
    pm_ok = False
    best_controller = None
    best_pm = 0
    for name, data in fb_results.items():
        pm = data['pm']
        status = "✓" if pm > 62 else "✗"
        print(f"   {status} {name}: PM = {pm:.1f}°")
        if pm > 62:
            pm_ok = True
            if pm > best_pm:
                best_pm = pm
                best_controller = name
    
    if not pm_ok:
        recommendations.append("Need to reduce bandwidth or add phase lead to achieve PM > 62°")
    
    # 2. Check mode excitation
    print("\n2. MODE EXCITATION:")
    
    # Find best controller (one that doesn't excite modes AND has PM > 62)
    for name, excites in fb_excites.items():
        print(f"   {name}: {'Excites modes' if excites else 'Does not excite modes'}")
    
    if best_controller:
        recommendations.append(f"Use {best_controller} controller configuration (PM={best_pm:.1f}°)")
    
    # 3. Pointing error
    print("\n3. POINTING ERROR REQUIREMENT (≤ 0.1°):")
    vib_deg = pointing['vibration_contribution_deg']
    if vib_deg < 0.01:  # Very small
        print(f"   ✓ Vibration contribution is negligible ({vib_deg*1000:.2f} mdeg)")
    elif vib_deg < 0.1:
        print(f"   ✓ Vibration contribution is acceptable ({vib_deg*1000:.2f} mdeg)")
    else:
        print(f"   ⚠ Vibration contribution may be too high ({vib_deg*1000:.2f} mdeg)")
        recommendations.append("Consider input shaping to reduce residual vibration")
    
    # 4. Specific recommendations
    print("\n" + "-" * 60)
    print("SPECIFIC RECOMMENDATIONS:")
    print("-" * 60)
    
    recs = [
        "1. Use LOW BANDWIDTH PD: Set bandwidth = first_mode/6 ≈ 0.067 Hz",
        "   This keeps gain crossover well below first resonance.",
        "",
        "2. ADD DERIVATIVE FILTER: Set filter cutoff = first_mode/2 ≈ 0.2 Hz",
        "   This reduces high-frequency controller gain without affecting PM much.",
        "",
        "3. USE INPUT SHAPING: Apply ZVD or 4th-order shaping to feedforward",
        "   This eliminates most mode excitation during slew.",
        "",
        "4. IF PM is still < 62°: Consider PPF with proper tuning:",
        "   - PPF filter frequency = 0.9 × modal frequency",
        "   - PPF damping = 0.5",
        "   - PPF gain: Start low (1-5) and increase until PM drops",
        "",
        "5. VERIFY modal gains match between simulation and analysis",
        "   Current discrepancy: modal_gains vs control_modal_gains"
    ]
    
    for rec in recs:
        print(f"   {rec}")
    
    return recommendations


def main():
    """Run comprehensive investigation."""
    print("=" * 80)
    print("COMPREHENSIVE SPACECRAFT CONTROL INVESTIGATION")
    print("=" * 80)
    print("\nTarget Requirements:")
    print("  - Pointing error <= 0.1 degrees")
    print("  - Phase margin > 62 degrees") 
    print("  - No excitation of structural modes (0.4 Hz, 1.3 Hz)")
    
    # 1. Investigate modal gains
    modal_gains = compute_actual_modal_gains()
    
    # 2. Check feedforward capability
    ff_ok, theta, omega, alpha, t = analyze_feedforward_slew()
    
    # 3. Check feedforward mode excitation
    ff_excitation, ff_excites = analyze_feedforward_mode_excitation()
    
    # 4. Check feedback mode excitation
    fb_results, fb_excites, freqs = analyze_feedback_mode_excitation()
    
    # 5. Compute pointing error
    pointing = compute_pointing_error()
    
    # 6. Generate recommendations
    recommendations = generate_recommendations(fb_results, fb_excites, ff_excites, pointing)
    
    print("\n" + "=" * 80)
    print("INVESTIGATION COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()
