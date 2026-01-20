"""
Analysis: Why filtered_pd shows more vibration than standard_pd

FINDINGS:
=========

1. POINTING ERROR:
   - standard_pd: 0.0037° (MEETS ≤0.1° requirement)
   - filtered_pd: 0.0168° (MEETS ≤0.1° requirement)  
   - All controllers achieve excellent final pointing!

2. VIBRATION ANALYSIS:
   - standard_pd mode1: 7.8mm peak → 0.06mm at end (DECAYING ✓)
   - filtered_pd mode1: 12.9mm peak → 0.45mm at end (DECAYING but slower ✓)
   
3. ROOT CAUSE OF HIGHER FILTERED_PD VIBRATION:
   
   The filter cutoff at 0.20 Hz introduces excessive phase lag at modal frequencies:
   
   Frequency   | Phase Lag | Gain
   ------------|-----------|------
   0.10 Hz     | 26.6°     | 0.894
   0.20 Hz     | 45.0°     | 0.707
   0.40 Hz     | 63.4°     | 0.447  ← Mode 1
   1.30 Hz     | 81.3°     | 0.152  ← Mode 2
   
   At mode 1 (0.4 Hz), the filter:
   - Delays rate feedback by 63° (0.44 seconds at 0.4 Hz!)
   - Reduces rate feedback gain to 45%
   
   This makes the controller LESS effective at damping modal oscillations.

4. FEEDFORWARD IMPACT:
   During the slew (0-30s), the feedforward torque excites modes.
   The unshaped bang-bang profile excites modes more than ZVD or fourth-order.
   Standard PD's faster rate feedback provides better modal damping during this phase.

5. WHY IS THIS ACTUALLY CORRECT BEHAVIOR?
   
   The filtered_pd is designed to AVOID EXCITING modes from feedback noise,
   NOT to actively damp modes better than standard PD.
   
   Standard PD has HIGHER gain at modal frequencies (infinite in theory),
   which means it responds faster to modal oscillations - both damping them
   AND potentially exciting them from measurement noise.
   
   In a NOISE-FREE simulation like this, standard PD appears better.
   In a REAL system with noisy rate sensors, filtered_pd would prevent
   the noise from being amplified into modal excitation.

6. RECOMMENDATIONS:

   a) For the simulation (noise-free):
      - Standard PD shows lowest vibration during transient
      - This is expected - faster rate feedback = faster damping
      
   b) For real hardware:
      - Use filtered_pd to avoid noise-induced excitation
      - OR use AVC with active PPF damping (currently disabled with gains=[0,0])
      
   c) To improve filtered_pd performance:
      - OPTION 1: Use a notch filter at modal frequencies instead of LP filter
      - OPTION 2: Use a higher cutoff (0.5-1.0 Hz) but add notch at modes
      - OPTION 3: Enable AVC with PPF gains > 0 for active modal damping

SUMMARY:
========
The filtered_pd showing higher vibration than standard_pd is PHYSICALLY CORRECT
for a noise-free simulation. The filter reduces control authority at modal 
frequencies. All controllers meet the pointing requirement of ≤0.1°.
"""

import numpy as np

# Verify all pointing errors
print("=" * 60)
print("FINAL POINTING ERROR VERIFICATION")
print("=" * 60)
print("\nRequirement: ≤ 0.1 degrees")
print("-" * 60)

configs = [
    ('unshaped_standard_pd', 'vizard_demo_unshaped_standard_pd.npz'),
    ('unshaped_filtered_pd', 'vizard_demo_unshaped_filtered_pd.npz'),
    ('unshaped_avc', 'vizard_demo_unshaped_avc.npz'),
    ('zvd_standard_pd', 'vizard_demo_zvd_standard_pd.npz'),
    ('zvd_filtered_pd', 'vizard_demo_zvd_filtered_pd.npz'),
    ('zvd_avc', 'vizard_demo_zvd_avc.npz'),
    ('fourth_standard_pd', 'vizard_demo_fourth_standard_pd.npz'),
    ('fourth_filtered_pd', 'vizard_demo_fourth_filtered_pd.npz'),
    ('fourth_avc', 'vizard_demo_fourth_avc.npz'),
]

target_180deg = np.array([0.0, 0.0, 1.0])

for name, f in configs:
    try:
        d = np.load(f)
        sigma = d['sigma']
        final_err = np.linalg.norm(sigma[-1] - target_180deg)
        final_err_deg = 4 * np.arctan(final_err) * 180 / np.pi
        status = "✓ PASS" if final_err_deg <= 0.1 else "✗ FAIL"
        print(f"  {name:24s}: {final_err_deg:.6f}° {status}")
    except FileNotFoundError:
        print(f"  {name:24s}: FILE NOT FOUND")
