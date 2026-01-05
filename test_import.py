"""Quick test to verify package installation"""

# Test 1: Can we import?
print("Test 1: Importing package...")
from input_shaping import ZV, ZVD, ZVDD, EI, design_shaper
print("✓ Import successful!")

# Test 2: Do the functions work?
print("\nTest 2: Testing ZV shaper...")
import numpy as np

omega_n = np.pi  # 0.5 Hz
zeta = 0.02

A, t, K = ZV(omega_n, zeta)
print(f"✓ ZV shaper: {len(A)} impulses")
print(f"  Amplitudes: {A}")
print(f"  Times: {t}")
print(f"  Damping factor K: {K:.4f}")
print(f"  Sum: {np.sum(A):.6f} (should be 1.0)")

# Test 3: Test all shapers
print("\nTest 3: Testing all shapers via design_shaper()...")
for method in ['ZV', 'ZVD', 'ZVDD', 'EI']:
    if method == 'EI':
        A, t = design_shaper(omega_n, zeta, method=method, Vtol=0.05, tol_band=0.20)
    else:
        A, t = design_shaper(omega_n, zeta, method=method)
    print(f"✓ {method}: {len(A)} impulses, duration = {t[-1]:.3f}s")

print("\n✓ All tests passed! Package is working correctly.")