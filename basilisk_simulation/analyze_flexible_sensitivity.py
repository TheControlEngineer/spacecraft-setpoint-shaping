"""
Analyze sensitivity and complementary sensitivity with FLEXIBLE plant included.
This shows the actual resonant peaks and how each controller affects them.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
import control
from feedback_control import MRPFeedbackController, FilteredDerivativeController, ActiveVibrationController
from mission_simulation import _build_flexible_plant_tf

# Spacecraft parameters
inertia = np.diag([900.0, 800.0, 600.0])
modal_freqs_hz = [0.4, 1.3]
modal_damping = [0.02, 0.015]
modal_gains = [0.0015, 0.0008]
axis = 2

# Control parameters
first_mode = modal_freqs_hz[0]
bandwidth = first_mode / 6.0  # Current: 0.067 Hz
wn = 2 * np.pi * bandwidth
K = wn**2 * inertia[axis, axis]
P = 2 * 0.7 * wn * inertia[axis, axis]

filter_cutoff = first_mode / 2.0  # Current: 0.2 Hz
ppf_gains = [5.0, 10.0]

print("=" * 80)
print("FLEXIBLE PLANT SENSITIVITY ANALYSIS")
print("=" * 80)
print(f"\nControl parameters:")
print(f"  Bandwidth: {bandwidth:.4f} Hz ({bandwidth/first_mode:.3f}×first_mode)")
print(f"  K = {K:.2f} N·m/rad, P = {P:.2f} N·m·s/rad")
print(f"  Filter cutoff: {filter_cutoff:.3f} Hz ({filter_cutoff/first_mode:.2f}×first_mode)")
print(f"  PPF gains: {ppf_gains}")
print(f"\nFlexible modes:")
print(f"  Mode 1: {modal_freqs_hz[0]} Hz, damping = {modal_damping[0]}")
print(f"  Mode 2: {modal_freqs_hz[1]} Hz, damping = {modal_damping[1]}")

# Build controllers
std_pd = MRPFeedbackController(inertia, K=K, P=P)

filt_pd = FilteredDerivativeController(
    inertia, K=K, P=P,
    filter_freq_hz=filter_cutoff
)

avc = ActiveVibrationController(
    inertia, K=K, P=P,
    modal_freqs_hz=modal_freqs_hz,
    modal_damping=modal_damping,
    modal_gains=modal_gains,
    ppf_gains=ppf_gains,
    filter_freq_hz=filter_cutoff
)

# Get open-loop transfer functions WITH FLEXIBILITY
L_std = std_pd.get_open_loop_tf(
    axis=axis, 
    modal_freqs_hz=modal_freqs_hz,
    modal_damping=modal_damping,
    modal_gains=modal_gains,
    include_flexibility=True
)
L_filt = filt_pd.get_open_loop_tf(axis=axis, include_flexibility=True)
L_avc = avc.get_open_loop_tf(axis=axis, include_flexibility=True)

# Compute margins
gm_std, pm_std, wgc_std, wpc_std = control.margin(L_std)
gm_filt, pm_filt, wgc_filt, wpc_filt = control.margin(L_filt)
gm_avc, pm_avc, wgc_avc, wpc_avc = control.margin(L_avc)

print("\n" + "=" * 80)
print("STABILITY MARGINS")
print("=" * 80)
print(f"Standard PD:   PM = {pm_std:.1f}°")
print(f"Filtered PD:   PM = {pm_filt:.1f}°")
print(f"AVC:           PM = {pm_avc:.1f}°")

# Frequency sweep
freqs = np.logspace(-2, 1, 1000)  # 0.01 Hz to 10 Hz
w = 2 * np.pi * freqs

# Compute frequency responses
_, L_std_mag = signal.freqresp(L_std, w)
_, L_filt_mag = signal.freqresp(L_filt, w)
_, L_avc_mag = signal.freqresp(L_avc, w)

# Compute sensitivity S = 1/(1+L) and complementary sensitivity T = L/(1+L)
S_std = 1 / (1 + L_std_mag)
S_filt = 1 / (1 + L_filt_mag)
S_avc = 1 / (1 + L_avc_mag)

T_std = L_std_mag / (1 + L_std_mag)
T_filt = L_filt_mag / (1 + L_filt_mag)
T_avc = L_avc_mag / (1 + L_avc_mag)

# Convert to dB
S_std_db = 20 * np.log10(np.abs(S_std))
S_filt_db = 20 * np.log10(np.abs(S_filt))
S_avc_db = 20 * np.log10(np.abs(S_avc))

T_std_db = 20 * np.log10(np.abs(T_std))
T_filt_db = 20 * np.log10(np.abs(T_filt))
T_avc_db = 20 * np.log10(np.abs(T_avc))

# Find values at modal frequencies
def find_nearest_idx(array, value):
    return np.argmin(np.abs(array - value))

idx_04 = find_nearest_idx(freqs, 0.4)
idx_13 = find_nearest_idx(freqs, 1.3)

print("\n" + "=" * 80)
print("SENSITIVITY S(jω) AT MODAL FREQUENCIES (disturbance rejection)")
print("Negative dB = attenuation (good), Positive dB = amplification (bad)")
print("=" * 80)

print(f"\nAt 0.4 Hz (Mode 1):")
print(f"  Standard PD: {S_std_db[idx_04]:+.2f} dB  (|S| = {np.abs(S_std[idx_04]):.3f})")
print(f"  Filtered PD: {S_filt_db[idx_04]:+.2f} dB  (|S| = {np.abs(S_filt[idx_04]):.3f})")
print(f"  AVC:         {S_avc_db[idx_04]:+.2f} dB  (|S| = {np.abs(S_avc[idx_04]):.3f})")

print(f"\nAt 1.3 Hz (Mode 2):")
print(f"  Standard PD: {S_std_db[idx_13]:+.2f} dB  (|S| = {np.abs(S_std[idx_13]):.3f})")
print(f"  Filtered PD: {S_filt_db[idx_13]:+.2f} dB  (|S| = {np.abs(S_filt[idx_13]):.3f})")
print(f"  AVC:         {S_avc_db[idx_13]:+.2f} dB  (|S| = {np.abs(S_avc[idx_13]):.3f})")

# Find peaks in S near modal frequencies
def find_peak_in_range(freqs, data_db, f_min, f_max):
    mask = (freqs >= f_min) & (freqs <= f_max)
    if np.any(mask):
        peak_idx = np.argmax(data_db[mask])
        peak_freq = freqs[mask][peak_idx]
        peak_val = data_db[mask][peak_idx]
        return peak_freq, peak_val
    return None, None

print("\n" + "=" * 80)
print("PEAK SENSITIVITY IN RESONANCE REGIONS")
print("=" * 80)

for mode_hz, mode_name in [(0.4, "Mode 1"), (1.3, "Mode 2")]:
    f_min, f_max = mode_hz * 0.8, mode_hz * 1.2
    print(f"\n{mode_name} ({mode_hz} Hz) region [{f_min:.2f}-{f_max:.2f} Hz]:")
    
    f_std, s_std = find_peak_in_range(freqs, S_std_db, f_min, f_max)
    f_filt, s_filt = find_peak_in_range(freqs, S_filt_db, f_min, f_max)
    f_avc, s_avc = find_peak_in_range(freqs, S_avc_db, f_min, f_max)
    
    print(f"  Standard PD: Peak {s_std:+.2f} dB at {f_std:.3f} Hz")
    print(f"  Filtered PD: Peak {s_filt:+.2f} dB at {f_filt:.3f} Hz")
    print(f"  AVC:         Peak {s_avc:+.2f} dB at {f_avc:.3f} Hz")

print("\n" + "=" * 80)
print("COMPLEMENTARY SENSITIVITY T(jω) AT MODAL FREQUENCIES")
print("High T means controller output EXCITES modes (bad)")
print("=" * 80)

print(f"\nAt 0.4 Hz (Mode 1):")
print(f"  Standard PD: {T_std_db[idx_04]:+.2f} dB  (|T| = {np.abs(T_std[idx_04]):.3f})")
print(f"  Filtered PD: {T_filt_db[idx_04]:+.2f} dB  (|T| = {np.abs(T_filt[idx_04]):.3f})")
print(f"  AVC:         {T_avc_db[idx_04]:+.2f} dB  (|T| = {np.abs(T_avc[idx_04]):.3f})")

print(f"\nAt 1.3 Hz (Mode 2):")
print(f"  Standard PD: {T_std_db[idx_13]:+.2f} dB  (|T| = {np.abs(T_std[idx_13]):.3f})")
print(f"  Filtered PD: {T_filt_db[idx_13]:+.2f} dB  (|T| = {np.abs(T_filt[idx_13]):.3f})")
print(f"  AVC:         {T_avc_db[idx_13]:+.2f} dB  (|T| = {np.abs(T_avc[idx_13]):.3f})")

# Find peaks in T near modal frequencies
print("\n" + "=" * 80)
print("PEAK COMPLEMENTARY SENSITIVITY IN RESONANCE REGIONS")
print("=" * 80)

for mode_hz, mode_name in [(0.4, "Mode 1"), (1.3, "Mode 2")]:
    f_min, f_max = mode_hz * 0.8, mode_hz * 1.2
    print(f"\n{mode_name} ({mode_hz} Hz) region [{f_min:.2f}-{f_max:.2f} Hz]:")
    
    f_std, t_std = find_peak_in_range(freqs, T_std_db, f_min, f_max)
    f_filt, t_filt = find_peak_in_range(freqs, T_filt_db, f_min, f_max)
    f_avc, t_avc = find_peak_in_range(freqs, T_avc_db, f_min, f_max)
    
    print(f"  Standard PD: Peak {t_std:+.2f} dB at {f_std:.3f} Hz")
    print(f"  Filtered PD: Peak {t_filt:+.2f} dB at {f_filt:.3f} Hz")
    print(f"  AVC:         Peak {t_avc:+.2f} dB at {f_avc:.3f} Hz")

# Create plots
fig, axes = plt.subplots(2, 1, figsize=(12, 10))

# Plot 1: Sensitivity S(jω) with flexible modes
ax1 = axes[0]
ax1.semilogx(freqs, S_std_db, 'b-', linewidth=2, label='Standard PD')
ax1.semilogx(freqs, S_filt_db, 'g-', linewidth=2, label='Filtered PD')
ax1.semilogx(freqs, S_avc_db, 'r-', linewidth=2, label='AVC')

ax1.axhline(0, color='k', linestyle='--', linewidth=1, alpha=0.5, label='0 dB (unity)')
ax1.axvline(modal_freqs_hz[0], color='orange', linestyle=':', alpha=0.7, label=f'Mode 1: {modal_freqs_hz[0]} Hz')
ax1.axvline(modal_freqs_hz[1], color='purple', linestyle=':', alpha=0.7, label=f'Mode 2: {modal_freqs_hz[1]} Hz')

ax1.set_xlabel('Frequency (Hz)', fontsize=12)
ax1.set_ylabel('Sensitivity |S(jω)| (dB)', fontsize=12)
ax1.set_title('Sensitivity Function S(jω) = 1/(1+L) WITH FLEXIBLE MODES\n(Negative = disturbance attenuation, Positive = amplification)', fontsize=13, fontweight='bold')
ax1.grid(True, which='both', alpha=0.3)
ax1.legend(loc='upper right', fontsize=10)
ax1.set_xlim([0.01, 10])
ax1.set_ylim([-40, 30])

# Plot 2: Complementary sensitivity T(jω) with flexible modes
ax2 = axes[1]
ax2.semilogx(freqs, T_std_db, 'b-', linewidth=2, label='Standard PD')
ax2.semilogx(freqs, T_filt_db, 'g-', linewidth=2, label='Filtered PD')
ax2.semilogx(freqs, T_avc_db, 'r-', linewidth=2, label='AVC')

ax2.axhline(0, color='k', linestyle='--', linewidth=1, alpha=0.5, label='0 dB (unity)')
ax2.axvline(modal_freqs_hz[0], color='orange', linestyle=':', alpha=0.7, label=f'Mode 1: {modal_freqs_hz[0]} Hz')
ax2.axvline(modal_freqs_hz[1], color='purple', linestyle=':', alpha=0.7, label=f'Mode 2: {modal_freqs_hz[1]} Hz')

ax2.set_xlabel('Frequency (Hz)', fontsize=12)
ax2.set_ylabel('Complementary Sensitivity |T(jω)| (dB)', fontsize=12)
ax2.set_title('Complementary Sensitivity T(jω) = L/(1+L) WITH FLEXIBLE MODES\n(High values = controller excites modes)', fontsize=13, fontweight='bold')
ax2.grid(True, which='both', alpha=0.3)
ax2.legend(loc='lower left', fontsize=10)
ax2.set_xlim([0.01, 10])
ax2.set_ylim([-50, 30])

plt.tight_layout()
plt.savefig('analysis/flexible_sensitivity_analysis.png', dpi=150)
print("\n" + "=" * 80)
print(f"Saved plot: analysis/flexible_sensitivity_analysis.png")
print("=" * 80)

print("\n" + "=" * 80)
print("SUMMARY & RECOMMENDATIONS")
print("=" * 80)

print("\nCurrent design issues:")
if pm_filt < 62 or pm_avc < 62:
    print(f"  ✗ Phase margin too low (requirement: > 62°)")
    print(f"    Filtered PD: {pm_filt:.1f}°, AVC: {pm_avc:.1f}°")

if S_std_db[idx_04] < S_filt_db[idx_04]:
    print(f"  ✗ Filtered PD/AVC have WORSE sensitivity than Standard PD at Mode 1")
    print(f"    Standard PD: {S_std_db[idx_04]:.2f} dB vs Filtered PD: {S_filt_db[idx_04]:.2f} dB")

print("\nPossible solutions:")
print("  1. Increase bandwidth (currently {:.4f} Hz = first_mode/6)".format(bandwidth))
print("  2. Increase filter cutoff (currently {:.3f} Hz = first_mode/2)".format(filter_cutoff))
print("  3. Adjust PPF gains to add damping without destabilizing")
print("  4. Accept Standard PD with high gain if T(jω) shows it doesn't excite modes")
