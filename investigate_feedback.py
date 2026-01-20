"""Investigate feedback vibration issue and filter settings."""
import numpy as np

print("=" * 60)
print("INVESTIGATING FEEDBACK VIBRATION")
print("=" * 60)

# Check the POINTING ERROR (final)
print("\n0. POINTING ERROR - DETAILED CHECK:")
files = [
    ('standard_pd', 'vizard_demo_unshaped_standard_pd.npz'),
    ('filtered_pd', 'vizard_demo_unshaped_filtered_pd.npz'),
    ('avc', 'vizard_demo_unshaped_avc.npz'),
]

for name, f in files:
    d = np.load(f)
    t = d['time']
    sigma = d['sigma']
    
    # Check actual values
    print(f"\n  {name}:")
    print(f"    Initial sigma: {sigma[0]}")
    print(f"    Final sigma:   {sigma[-1]}")
    
    # What angle does final sigma represent?
    s = sigma[-1]
    norm_s = np.linalg.norm(s)
    angle_deg = 4 * np.arctan(norm_s) * 180 / np.pi
    print(f"    Final MRP norm: {norm_s:.6f} -> angle: {angle_deg:.2f}°")
    
    # What is the target?
    # For 90° rotation about Z, MRP = tan(45/2) * [0,0,1] = 0.41421 * [0,0,1]
    target_90deg = np.array([0.0, 0.0, 0.41421356])
    err_to_90 = np.linalg.norm(sigma[-1] - target_90deg)
    err_to_90_deg = 4 * np.arctan(err_to_90) * 180 / np.pi
    print(f"    Error to 90° target: {err_to_90_deg:.4f}°")
    
    # For 180° rotation about Z, MRP = tan(90/2) * [0,0,1] = 1.0 * [0,0,1]
    target_180deg = np.array([0.0, 0.0, 1.0])
    err_to_180 = np.linalg.norm(sigma[-1] - target_180deg)
    err_to_180_deg = 4 * np.arctan(err_to_180) * 180 / np.pi
    print(f"    Error to 180° target: {err_to_180_deg:.4f}°")

# Compare controllers on unshaped
print("\n1. UNSHAPED - Comparing controllers:")
for name, f in files:
    d = np.load(f)
    t = d['time']
    m1 = d['mode1']
    m2 = d['mode2']
    settle_idx = np.searchsorted(t, 30)
    
    # RMS after maneuver
    rms1 = np.sqrt(np.mean(m1[settle_idx:]**2)) * 1000
    rms2 = np.sqrt(np.mean(m2[settle_idx:]**2)) * 1000
    
    # Check if growing or decaying
    window1 = np.sqrt(np.mean(m1[settle_idx:settle_idx+1000]**2)) * 1000
    window2 = np.sqrt(np.mean(m1[-1000:]**2)) * 1000
    trend = "DECAYING" if window2 < window1 else "GROWING!"
    
    print(f"  {name:12s}: mode1 RMS={rms1:.4f}mm, mode2 RMS={rms2:.4f}mm | {trend}")
    print(f"               30-40s: {window1:.4f}mm -> 80-90s: {window2:.4f}mm")

# ROOT CAUSE: filter cutoff analysis
print("\n2. FILTER PHASE LAG ANALYSIS (new 0.5 Hz cutoff):")
print("   Control bandwidth: 0.10 Hz")
print("   Filter cutoff: 0.50 Hz (increased from 0.20 Hz)")
print("   First mode: 0.40 Hz, Second mode: 1.30 Hz")
print("\n   Phase lag at each frequency (1st order LP filter):")
for freq in [0.10, 0.40, 0.50, 1.30]:
    fc = 0.50  # new filter cutoff
    phase_lag = np.arctan(freq / fc) * 180 / np.pi
    mag_atten = 1 / np.sqrt(1 + (freq/fc)**2)
    print(f"     f={freq:.2f} Hz: phase lag={phase_lag:.1f}°, gain={mag_atten:.3f}")

print("\n   IMPROVEMENT: Only 38.7° phase lag at mode 1 (was 63.4° with 0.2 Hz cutoff)")

# Check FFT of feedback torque to see modal content
print("\n3. FEEDBACK TORQUE FREQUENCY CONTENT (post-maneuver):")
from scipy import signal

for name, f in files:
    d = np.load(f)
    t = d['time']
    fb_torque = d.get('fb_torque', None)
    if fb_torque is not None and fb_torque.ndim == 2:
        fb_z = fb_torque[:, 2]
        settle_idx = np.searchsorted(t, 35)
        dt = t[1] - t[0]
        
        # Get spectrum
        freqs, psd = signal.welch(fb_z[settle_idx:], fs=1/dt, nperseg=2048)
        
        # Find peaks near modes
        idx_04 = np.argmin(np.abs(freqs - 0.4))
        idx_13 = np.argmin(np.abs(freqs - 1.3))
        
        print(f"  {name:12s}: PSD at 0.4Hz={psd[idx_04]:.2e}, at 1.3Hz={psd[idx_13]:.2e}")
