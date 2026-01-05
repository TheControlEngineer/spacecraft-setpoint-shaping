import sys
import os
import numpy as np

import matplotlib.pyplot as plt

# 1. Get the absolute path of the current script (shaper_simulation.py)
# Result: .../spacecraft_input_shaping/examples/shaper_simulation.py
script_path = os.path.abspath(__file__)

# 2. Get the directory containing the script
# Result: .../spacecraft_input_shaping/examples
script_dir = os.path.dirname(script_path)

# 3. Get the parent directory (Project Root)
# Result: .../spacecraft_input_shaping
project_root = os.path.dirname(script_dir)

# 4. Add project root to system path so Python sees the 'shapers' folder

sys.path.append(project_root)
from shapers.ZV import ZV
from shapers.ZVD import ZVD
from shapers.ZVDD import ZVDD
from shapers.EI import EI

def residual_vibration(omega, amplitudes, times, zeta):
    """Calculate residual vibration amplitude at frequency omega"""
    V = 0
    for A, t in zip(amplitudes, times):
        V += A * np.exp(-zeta * omega * t) * np.exp(1j * omega * t)
    return np.abs(V)

# System parameters
omega_n = np.pi  # 0.5 Hz nominal
zeta = 0.02

# Get all shapers
A_zv, t_zv, _ = ZV(omega_n, zeta)
A_zvd, t_zvd, _ = ZVD(omega_n, zeta)
A_zvdd, t_zvdd = ZVDD(omega_n, zeta)
A_ei, t_ei = EI(omega_n, zeta, Vtol=0.10, tol_band=0.20)

# Frequency sweep: ±30% to see beyond the ±20% design range
freq_errors = np.linspace(-0.30, 0.30, 500)
omega_sweep = omega_n * (1 + freq_errors)

# Calculate residual vibration for each shaper across frequency range
V_zv = [residual_vibration(w, A_zv, t_zv, zeta) for w in omega_sweep]
V_zvd = [residual_vibration(w, A_zvd, t_zvd, zeta) for w in omega_sweep]
V_zvdd = [residual_vibration(w, A_zvdd, t_zvdd, zeta) for w in omega_sweep]
V_ei = [residual_vibration(w, A_ei, t_ei, zeta) for w in omega_sweep]

# Plot
plt.figure(figsize=(12, 8))

plt.subplot(2, 1, 1)
plt.plot(freq_errors * 100, V_zv, 'r-', linewidth=2, label='ZV')
plt.plot(freq_errors * 100, V_zvd, 'b-', linewidth=2, label='ZVD')
plt.plot(freq_errors * 100, V_zvdd, 'g-', linewidth=2, label='ZVDD')
plt.plot(freq_errors * 100, V_ei, 'm-', linewidth=2, label='EI (Vtol=10%)')
plt.axhline(y=0.10, color='k', linestyle='--', linewidth=1, label='10% tolerance')
plt.axvline(x=-20, color='gray', linestyle=':', alpha=0.5)
plt.axvline(x=20, color='gray', linestyle=':', alpha=0.5)
plt.xlabel('Frequency Error (%)')
plt.ylabel('Residual Vibration Amplitude')
plt.title(f'Robustness Comparison (fn = {omega_n/(2*np.pi):.2f} Hz, ζ = {zeta})')
plt.legend()
plt.grid(True, alpha=0.3)
plt.xlim([-30, 30])
plt.ylim([0, 0.4])

# Zoomed view on ±20% range
plt.subplot(2, 1, 2)
mask = np.abs(freq_errors) <= 0.20
plt.plot(freq_errors[mask] * 100, np.array(V_zv)[mask], 'r-', linewidth=2, label='ZV')
plt.plot(freq_errors[mask] * 100, np.array(V_zvd)[mask], 'b-', linewidth=2, label='ZVD')
plt.plot(freq_errors[mask] * 100, np.array(V_zvdd)[mask], 'g-', linewidth=2, label='ZVDD')
plt.plot(freq_errors[mask] * 100, np.array(V_ei)[mask], 'm-', linewidth=2, label='EI (Vtol=10%)')
plt.axhline(y=0.10, color='k', linestyle='--', linewidth=1, label='10% tolerance')
plt.xlabel('Frequency Error (%)')
plt.ylabel('Residual Vibration Amplitude')
plt.title('Zoomed View: ±20% Uncertainty Range')
plt.legend()
plt.grid(True, alpha=0.3)
plt.xlim([-20, 20])

plt.tight_layout()
plt.show()

# Print statistics
print("\nRobustness Statistics (within ±20% frequency uncertainty):")
print("="*70)
mask_20 = np.abs(freq_errors) <= 0.20
for name, V in [('ZV', V_zv), ('ZVD', V_zvd), ('ZVDD', V_zvdd), ('EI', V_ei)]:
    V_array = np.array(V)[mask_20]
    print(f"{name:<6} Max: {np.max(V_array):.4f}  |  Mean: {np.mean(V_array):.4f}  |  @ ±20%: {V_array[[0,-1]]}")