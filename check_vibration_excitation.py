"""
Proper analysis of vibration excitation by feedback controller.

Checks:
1. Actual time-domain vibration data from simulations
2. Complementary sensitivity T(jω) = L/(1+L) at modal frequencies
   (this shows if controller COMMANDS excite modes)
3. Closed-loop gain at modal frequencies
4. Whether PD derivative term is exciting high-frequency modes
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import signal
from pathlib import Path

from spacecraft_properties import HUB_INERTIA
from mission_simulation import default_config, _compute_control_analysis


def check_actual_vibration_data():
    """Load and check actual vibration from NPZ files."""
    
    print('='*70)
    print('1. ACTUAL VIBRATION DATA FROM SIMULATIONS')
    print('='*70)
    
    for method in ['unshaped', 'zvd', 'fourth']:
        for controller in ['standard_pd', 'filtered_pd', 'avc']:
            try:
                npz_file = f'vizard_demo_{method}_{controller}.npz'
                data = np.load(npz_file, allow_pickle=True)
                
                time = data['time']
                mode1 = data['mode1']
                mode2 = data['mode2']
                
                # Feedback phase: after 35s
                fb_idx = time >= 35.0
                
                if np.any(fb_idx):
                    mode1_fb = mode1[fb_idx]
                    mode2_fb = mode2[fb_idx]
                    
                    # RMS during feedback
                    rms1 = np.sqrt(np.mean(mode1_fb**2)) * 1000  # mm
                    rms2 = np.sqrt(np.mean(mode2_fb**2)) * 1000  # mm
                    max1 = np.max(np.abs(mode1_fb)) * 1000
                    max2 = np.max(np.abs(mode2_fb)) * 1000
                    
                    print(f'\n{method}_{controller}:')
                    print(f'  Mode 1 (0.4 Hz): RMS={rms1:.3f}mm, Max={max1:.3f}mm')
                    print(f'  Mode 2 (1.3 Hz): RMS={rms2:.3f}mm, Max={max2:.3f}mm')
                    
            except FileNotFoundError:
                print(f'\n{method}_{controller}: File not found - need to generate')
            except Exception as e:
                print(f'\n{method}_{controller}: Error - {e}')


def check_complementary_sensitivity():
    """Check complementary sensitivity T(jω) at modal frequencies.
    
    T(jω) = L/(1+L) shows how controller responds to reference commands.
    High |T(jω)| at modal frequencies means controller AMPLIFIES those frequencies.
    """
    
    config = default_config()
    ctrl_analysis = _compute_control_analysis(config)
    
    freqs = ctrl_analysis['freqs']
    
    print('\n' + '='*70)
    print('2. COMPLEMENTARY SENSITIVITY T(jω) AT MODAL FREQUENCIES')
    print('   (High gain here means controller EXCITES modes!)')
    print('='*70)
    
    for controller_name in ['standard_pd', 'filtered_pd', 'avc']:
        print(f'\n{controller_name.upper()}:')
        
        # Complementary sensitivity T(s) = L/(1+L)
        T = ctrl_analysis['T'][controller_name]
        T_mag_db = 20 * np.log10(np.abs(T) + 1e-12)
        
        # Check at modal frequencies
        idx_04 = np.argmin(np.abs(freqs - 0.4))
        idx_13 = np.argmin(np.abs(freqs - 1.3))
        
        t_04 = T_mag_db[idx_04]
        t_13 = T_mag_db[idx_13]
        
        print(f'  T @ 0.4 Hz: {t_04:.2f} dB ({10**(t_04/20):.3f}x)')
        print(f'  T @ 1.3 Hz: {t_13:.2f} dB ({10**(t_13/20):.3f}x)')
        
        # Find peak in T around modal frequencies
        idx_mode1_range = (freqs >= 0.35) & (freqs <= 0.45)
        idx_mode2_range = (freqs >= 1.2) & (freqs <= 1.4)
        
        peak1_idx = np.argmax(T_mag_db[idx_mode1_range])
        peak2_idx = np.argmax(T_mag_db[idx_mode2_range])
        
        peak1_freq = freqs[idx_mode1_range][peak1_idx]
        peak1_mag = T_mag_db[idx_mode1_range][peak1_idx]
        
        peak2_freq = freqs[idx_mode2_range][peak2_idx]
        peak2_mag = T_mag_db[idx_mode2_range][peak2_idx]
        
        print(f'  Peak near Mode 1: {peak1_mag:.2f} dB at {peak1_freq:.3f} Hz')
        print(f'  Peak near Mode 2: {peak2_mag:.2f} dB at {peak2_freq:.3f} Hz')
        
        # Assessment
        if t_04 > 3.0:  # >3dB means >1.4x amplification
            print(f'  ✗ HIGH GAIN at 0.4 Hz - controller will EXCITE mode 1!')
        elif t_04 > 0:
            print(f'  ⚠ Positive gain at 0.4 Hz - mild excitation risk')
        else:
            print(f'  ✓ Low gain at 0.4 Hz - won\'t excite mode 1')


def check_pd_high_frequency_gain():
    """Check if PD controller has high gain at modal frequencies.
    
    Standard PD: C(s) = K + P*s has INCREASING gain with frequency!
    This is the problem - it amplifies high-frequency disturbances.
    """
    
    print('\n' + '='*70)
    print('3. PD CONTROLLER GAIN AT MODAL FREQUENCIES')
    print('   (PD has increasing gain → excites high frequencies)')
    print('='*70)
    
    config = default_config()
    I = float(config.inertia[2, 2])
    
    # Standard PD gains
    bandwidth = 0.1
    omega_bw = 2 * np.pi * bandwidth
    K = I * omega_bw**2
    P = 2 * 0.7 * I * omega_bw
    
    print(f'\nStandard PD: C(s) = {P:.1f}*s + {K:.1f}')
    
    # Evaluate at modal frequencies
    freqs_test = [0.1, 0.4, 1.3, 5.0]  # Bandwidth, Mode 1, Mode 2, High freq
    
    print(f'\nController gain |C(jω)|:')
    for f in freqs_test:
        omega = 2 * np.pi * f
        C = K + 1j * P * omega
        C_mag = np.abs(C)
        C_db = 20 * np.log10(C_mag)
        print(f'  At {f:5.1f} Hz: {C_db:6.1f} dB ({C_mag:8.1f} N·m/rad)')
    
    print(f'\nFiltered PD: C(s) = K + P*s/(τ*s + 1), τ = 1/(2π*1.0Hz) = 0.159s')
    tau = 1.0 / (2 * np.pi * 1.0)
    
    print(f'\nController gain |C(jω)|:')
    for f in freqs_test:
        omega = 2 * np.pi * f
        C = K + P * (1j * omega) / (tau * 1j * omega + 1)
        C_mag = np.abs(C)
        C_db = 20 * np.log10(C_mag)
        print(f'  At {f:5.1f} Hz: {C_db:6.1f} dB ({C_mag:8.1f} N·m/rad)')
    
    print('\nConclusion:')
    print('  Standard PD: Gain increases with frequency → excites modes')
    print('  Filtered PD: Gain rolls off above 1 Hz → reduces excitation')


def plot_sensitivity_functions():
    """Plot S and T to visualize the problem."""
    
    config = default_config()
    ctrl_analysis = _compute_control_analysis(config)
    
    freqs = ctrl_analysis['freqs']
    
    fig, axes = plt.subplots(2, 1, figsize=(10, 8))
    
    # Plot S (disturbance attenuation)
    ax = axes[0]
    for controller_name in ['standard_pd', 'filtered_pd', 'avc']:
        S = ctrl_analysis['S'][controller_name]
        S_mag_db = 20 * np.log10(np.abs(S) + 1e-12)
        ax.semilogx(freqs, S_mag_db, label=controller_name)
    
    ax.axhline(0, color='k', linestyle='--', linewidth=0.5, alpha=0.5)
    ax.axvline(0.4, color='r', linestyle=':', alpha=0.5, label='Mode 1')
    ax.axvline(1.3, color='b', linestyle=':', alpha=0.5, label='Mode 2')
    ax.set_ylabel('Sensitivity S [dB]')
    ax.set_title('Sensitivity S(jω) = 1/(1+L) - Disturbance Rejection')
    ax.grid(True, alpha=0.3)
    ax.legend()
    ax.set_ylim([-20, 10])
    
    # Plot T (reference tracking / controller output)
    ax = axes[1]
    for controller_name in ['standard_pd', 'filtered_pd', 'avc']:
        T = ctrl_analysis['T'][controller_name]
        T_mag_db = 20 * np.log10(np.abs(T) + 1e-12)
        ax.semilogx(freqs, T_mag_db, label=controller_name)
    
    ax.axhline(0, color='k', linestyle='--', linewidth=0.5, alpha=0.5)
    ax.axvline(0.4, color='r', linestyle=':', alpha=0.5, label='Mode 1')
    ax.axvline(1.3, color='b', linestyle=':', alpha=0.5, label='Mode 2')
    ax.set_xlabel('Frequency [Hz]')
    ax.set_ylabel('Complementary Sensitivity T [dB]')
    ax.set_title('Complementary Sensitivity T(jω) = L/(1+L) - Controller Excitation')
    ax.grid(True, alpha=0.3)
    ax.legend()
    ax.set_ylim([-60, 20])
    
    plt.tight_layout()
    plt.savefig('analysis/complementary_sensitivity_check.png', dpi=150)
    print(f'\nSaved plot: analysis/complementary_sensitivity_check.png')


if __name__ == '__main__':
    check_actual_vibration_data()
    check_complementary_sensitivity()
    check_pd_high_frequency_gain()
    plot_sensitivity_functions()
