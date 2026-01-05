import sys
import os
import numpy as np
import matplotlib.pyplot as plt  # Fixed typo: matpoltlib -> matplotlib

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

# Now these imports will work because Python can find 'shapers' in the path
from shapers.ZV import ZV
from shapers.ZVD import ZVD
from shapers.ZVDD import ZVDD



def simulate_oscillator(amplitudes, times, omega_n, zeta, t_eval):
    """
    Simulate 1-DOF oscillator response to impulse sequence
    
    Parameters:
    -----------
    amplitudes : array
        Impulse amplitudes
    times : array  
        Impulse times
    omega_n : float
        Natural frequency (rad/s)
    zeta : float
        Damping ratio
    t_eval : array
        Time points to evaluate response
        
    Returns:
    --------
    x : array
        Displacement response at t_eval
    """
    omega_d = omega_n * np.sqrt(1 - zeta**2)
    x = np.zeros_like(t_eval)
    
    # Loop over each impulse and add its contribution
    for A_i, t_i in zip(amplitudes, times):
        # Only compute response after impulse occurs
        mask = t_eval >= t_i
        dt = t_eval[mask] - t_i
        x[mask] += (A_i / omega_d) * np.exp(-zeta * omega_n * dt) * np.sin(omega_d * dt)
    
    return x

# System parameters
omega_n = np.pi  # 0.5 Hz natural frequency
zeta = 0.02      # Light damping
f_n = omega_n / (2 * np.pi)

# Time vector for simulation
t_sim = np.linspace(0, 10, 2000)  # 10 seconds

# Get shapers
A_zv, t_zv, _ = ZV(omega_n, zeta)
A_zvd, t_zvd, _ = ZVD(omega_n, zeta)
A_zvdd, t_zvdd = ZVDD(omega_n, zeta)

# Unshaped (single impulse)
A_unshaped = np.array([1.0])
t_unshaped = np.array([0.0])

# Simulate all cases
x_unshaped = simulate_oscillator(A_unshaped, t_unshaped, omega_n, zeta, t_sim)
x_zv = simulate_oscillator(A_zv, t_zv, omega_n, zeta, t_sim)
x_zvd = simulate_oscillator(A_zvd, t_zvd, omega_n, zeta, t_sim)
x_zvdd = simulate_oscillator(A_zvdd, t_zvdd, omega_n, zeta, t_sim)

# Plot results
plt.figure(figsize=(12, 8))

plt.subplot(2, 1, 1)
plt.plot(t_sim, x_unshaped, 'k-', linewidth=2, label='Unshaped')
plt.plot(t_sim, x_zv, 'r-', linewidth=1.5, label='ZV')
plt.plot(t_sim, x_zvd, 'b-', linewidth=1.5, label='ZVD')
plt.plot(t_sim, x_zvdd, 'g-', linewidth=1.5, label='ZVDD')
plt.xlabel('Time (s)')
plt.ylabel('Displacement')
plt.title(f'Vibration Suppression Comparison (f_n = {f_n:.2f} Hz, ζ = {zeta})')
plt.legend()
plt.grid(True, alpha=0.3)
plt.xlim([0, 10])

# Zoom in on settling behavior
plt.subplot(2, 1, 2)
plt.plot(t_unshaped, A_unshaped, 'k-', linewidth=2, label='Unshaped')
plt.plot(t_zv, A_zv, 'r-', linewidth=1.5, label='ZV')
plt.plot(t_zvd, A_zvd, 'b-', linewidth=1.5, label='ZVD')
plt.plot(t_zvdd, A_zvdd, 'g-', linewidth=1.5, label='ZVDD')
plt.xlabel('Time (s)')
plt.ylabel('Force Amplitude')
plt.title('Impulse force sequences for each shaper')
plt.legend()
plt.grid(True, alpha=0.3)


plt.tight_layout()
plt.show()

# Print settling times and peak residuals
print(f"\n{'Shaper':<10} {'Duration (s)':<15} {'Residual Amplitude':<20}")
print("="*50)
print(f"{'Unshaped':<10} {0.0:<15.2f} {np.max(np.abs(x_unshaped[t_sim > 5])):<20.6f}")
print(f"{'ZV':<10} {t_zv[-1]:<15.2f} {np.max(np.abs(x_zv[t_sim > 5])):<20.6f}")
print(f"{'ZVD':<10} {t_zvd[-1]:<15.2f} {np.max(np.abs(x_zvd[t_sim > 5])):<20.6f}")
print(f"{'ZVDD':<10} {t_zvdd[-1]:<15.2f} {np.max(np.abs(x_zvdd[t_sim > 5])):<20.6f}")


