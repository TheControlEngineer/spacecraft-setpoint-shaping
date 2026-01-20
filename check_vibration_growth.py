"""Check if vibration is growing or decaying during settling."""
import numpy as np

d = np.load('vizard_demo_unshaped_standard_pd.npz')
t = d['time']
m1 = d['mode1']
m2 = d['mode2']

print("Checking vibration decay for unshaped + standard_pd:")
print("\nMode 1 RMS in 10-second windows after maneuver (30s):")
for start in [30, 40, 50, 60, 70, 80]:
    start_idx = np.searchsorted(t, start)
    end_idx = np.searchsorted(t, start + 10)
    if end_idx <= len(m1):
        rms1 = np.sqrt(np.mean(m1[start_idx:end_idx]**2)) * 1000
        rms2 = np.sqrt(np.mean(m2[start_idx:end_idx]**2)) * 1000
        print(f"  {start}-{start+10}s: mode1={rms1:.4f}mm, mode2={rms2:.4f}mm")

print("\nMode 1 peak values:")
print(f"  Max at t={t[np.argmax(np.abs(m1))]:.1f}s: {np.max(np.abs(m1))*1000:.4f}mm")
print(f"  Final value: {np.abs(m1[-1])*1000:.6f}mm")
