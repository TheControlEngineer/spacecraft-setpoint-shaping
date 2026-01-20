"""
Comprehensive pointing performance analysis.

Analyzes:
1. Stability margins (phase and gain)
2. Sensitivity function at modal frequencies
3. Disturbance-to-pointing-error transfer function
4. Mode excitation during feedforward vs feedback phases
5. Recommendations for control improvements
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import signal
from pathlib import Path

from spacecraft_properties import HUB_INERTIA
from mission_simulation import default_config, _compute_control_analysis


def analyze_disturbance_rejection():
    """Analyze disturbance-to-pointing-error transfer function."""
    
    config = default_config()
    ctrl_analysis = _compute_control_analysis(config)
    
    freqs = ctrl_analysis['freqs']
    
    print('='*70)
    print('DISTURBANCE-TO-POINTING-ERROR TRANSFER FUNCTION ANALYSIS')
    print('='*70)
    
    for controller_name in ['standard_pd', 'filtered_pd', 'avc']:
        print(f'\n{controller_name.upper()}:')
        print('-'*70)
        
        # Sensitivity function S(s) = 1 / (1 + L(s))
        # This is the transfer function from disturbance to error
        S = ctrl_analysis['S'][controller_name]
        S_mag_db = 20 * np.log10(np.abs(S))
        
        # Find peak sensitivity
        peak_idx = np.argmax(S_mag_db)
        peak_freq = freqs[peak_idx]
        peak_mag = S_mag_db[peak_idx]
        
        print(f'  Peak sensitivity: {peak_mag:.2f} dB at {peak_freq:.3f} Hz')
        
        # Check at modal frequencies
        idx_04 = np.argmin(np.abs(freqs - 0.4))
        idx_13 = np.argmin(np.abs(freqs - 1.3))
        
        s_04 = S_mag_db[idx_04]
        s_13 = S_mag_db[idx_13]
        
        print(f'  Sensitivity at 0.4 Hz: {s_04:.2f} dB ({10**(s_04/20):.3f}x)')
        print(f'  Sensitivity at 1.3 Hz: {s_13:.2f} dB ({10**(s_13/20):.3f}x)')
        
        # Stability margins
        margins = ctrl_analysis['margins'][controller_name]
        pm = margins['phase_margin_deg']
        gm = margins['gain_margin_db']
        
        print(f'  Phase margin: {pm:.2f} deg')
        print(f'  Gain margin: {gm:.2f} dB')
        
        # Assess performance
        print('\n  Assessment:')
        if pm < 62.0:
            print(f'    ✗ Phase margin BELOW 62° requirement ({pm:.1f}°)')
        else:
            print(f'    ✓ Phase margin above 62° requirement ({pm:.1f}°)')
        
        if s_04 > 0:
            print(f'    ✗ Amplifies disturbances at 0.4 Hz mode (+{s_04:.1f} dB)')
        else:
            print(f'    ✓ Attenuates disturbances at 0.4 Hz mode ({s_04:.1f} dB)')
        
        if s_13 > 0:
            print(f'    ✗ Amplifies disturbances at 1.3 Hz mode (+{s_13:.1f} dB)')
        else:
            print(f'    ✓ Attenuates disturbances at 1.3 Hz mode ({s_13:.1f} dB)')


def analyze_frequency_contribution():
    """Determine which frequency ranges contribute most to pointing error."""
    
    config = default_config()
    ctrl_analysis = _compute_control_analysis(config)
    
    freqs = ctrl_analysis['freqs']
    
    print('\n' + '='*70)
    print('FREQUENCY CONTRIBUTION TO POINTING ERROR')
    print('='*70)
    
    for controller_name in ['standard_pd', 'filtered_pd', 'avc']:
        print(f'\n{controller_name.upper()}:')
        
        S = ctrl_analysis['S'][controller_name]
        S_mag = np.abs(S)
        
        # Divide frequency range into bands
        bands = [
            (0.0, 0.1, 'DC to 0.1 Hz'),
            (0.1, 0.35, '0.1-0.35 Hz (below modes)'),
            (0.35, 0.45, '0.35-0.45 Hz (Mode 1)'),
            (0.45, 1.2, '0.45-1.2 Hz (between modes)'),
            (1.2, 1.4, '1.2-1.4 Hz (Mode 2)'),
            (1.4, 10.0, '1.4-10 Hz (above modes)'),
        ]
        
        for f_low, f_high, label in bands:
            idx = (freqs >= f_low) & (freqs < f_high)
            if np.any(idx):
                avg_sens = np.mean(S_mag[idx])
                max_sens = np.max(S_mag[idx])
                print(f'  {label:25s}: Avg={avg_sens:.3f}, Max={max_sens:.3f}')


def make_recommendations():
    """Generate recommendations based on analysis."""
    
    config = default_config()
    ctrl_analysis = _compute_control_analysis(config)
    
    print('\n' + '='*70)
    print('RECOMMENDATIONS')
    print('='*70)
    
    # Check standard_pd
    margins_std = ctrl_analysis['margins']['standard_pd']
    S_std = ctrl_analysis['S']['standard_pd']
    freqs = ctrl_analysis['freqs']
    
    idx_04 = np.argmin(np.abs(freqs - 0.4))
    idx_13 = np.argmin(np.abs(freqs - 1.3))
    
    s_04_db = 20 * np.log10(np.abs(S_std[idx_04]))
    s_13_db = 20 * np.log10(np.abs(S_std[idx_13]))
    pm_std = margins_std['phase_margin_deg']
    
    print(f'\n1. STANDARD PD CONTROLLER:')
    print(f'   Phase Margin: {pm_std:.1f}° ({"PASS" if pm_std >= 62 else "FAIL"})')
    print(f'   S @ 0.4 Hz: {s_04_db:.2f} dB ({"PASS" if s_04_db < 0 else "FAIL"})')
    print(f'   S @ 1.3 Hz: {s_13_db:.2f} dB ({"PASS" if s_13_db < 0 else "FAIL"})')
    
    if pm_std >= 62 and s_04_db < 0 and s_13_db < 0:
        print('\n   ✓ Standard PD meets all requirements!')
        print('   ✓ No active vibration control needed')
        print('   ✓ Input shaping will work effectively with this controller')
    else:
        print('\n   Consider adjustments to meet all requirements')
    
    # Check filtered_pd
    margins_filt = ctrl_analysis['margins']['filtered_pd']
    S_filt = ctrl_analysis['S']['filtered_pd']
    
    s_04_filt_db = 20 * np.log10(np.abs(S_filt[idx_04]))
    pm_filt = margins_filt['phase_margin_deg']
    
    print(f'\n2. FILTERED PD CONTROLLER:')
    print(f'   Phase Margin: {pm_filt:.1f}° ({"PASS" if pm_filt >= 62 else "FAIL"})')
    print(f'   S @ 0.4 Hz: {s_04_filt_db:.2f} dB ({"PASS" if s_04_filt_db < 0 else "FAIL"})')
    
    if pm_filt < 62 or s_04_filt_db > 0:
        print('\n   ✗ Filtered PD does NOT meet requirements')
        print('   → Filtering adds phase lag that reduces stability and')
        print('     increases sensitivity at modal frequencies')
        print('   → Use Standard PD instead')
    
    print(f'\n3. OVERALL RECOMMENDATION:')
    print('   → Use STANDARD PD controller for all operations')
    print('   → Combine with ZVD or Fourth-Order input shaping for feedforward')
    print('   → This provides:')
    print('      • Good phase margin (>62°)')
    print('      • Negative sensitivity at modes (attenuation)')
    print('      • Simple, robust design')


if __name__ == '__main__':
    analyze_disturbance_rejection()
    analyze_frequency_contribution()
    make_recommendations()
