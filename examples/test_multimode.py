"""
Test multi-mode shaping approaches
"""

import numpy as np
import matplotlib.pyplot as plt
from input_shaping import ZV, ZVD, convolve_shapers, design_multimode_cascaded


def test_basic_convolution():
    """Test basic 2-mode convolution"""
    print("="*60)
    print("Test 1: Basic Convolution (2 modes)")
    print("="*60)
    
    # Two modes: 0.3 Hz and 0.8 Hz
    f1, f2 = 0.3, 0.8
    zeta = 0.02
    
    # Design single-mode shapers
    A1, t1, _ = ZV(2*np.pi*f1, zeta)
    A2, t2, _ = ZV(2*np.pi*f2, zeta)
    
    print(f"\nMode 1 ({f1} Hz) - ZV shaper:")
    print(f"  Impulses: {len(A1)}")
    print(f"  Duration: {t1[-1]:.3f} s")
    
    print(f"\nMode 2 ({f2} Hz) - ZV shaper:")
    print(f"  Impulses: {len(A2)}")
    print(f"  Duration: {t2[-1]:.3f} s")
    
    # Convolve
    A_multi, t_multi = convolve_shapers((A1, t1), (A2, t2))
    
    print(f"\nMulti-mode (cascaded) shaper:")
    print(f"  Impulses: {len(A_multi)} (expected: {len(A1) * len(A2)})")
    print(f"  Duration: {t_multi[-1]:.3f} s (expected: {t1[-1] + t2[-1]:.3f} s)")
    print(f"  Sum of amplitudes: {np.sum(A_multi):.6f}")
    
    # Verify
    assert len(A_multi) == len(A1) * len(A2), "Wrong number of impulses!"
    assert np.isclose(t_multi[-1], t1[-1] + t2[-1], atol=1e-3), "Wrong duration!"
    assert np.isclose(np.sum(A_multi), 1.0, atol=1e-6), "Unity gain violated!"
    
    print("\n✓ All checks passed!")
    
    return A_multi, t_multi


def test_convenience_function():
    """Test design_multimode_cascaded convenience function"""
    print("\n" + "="*60)
    print("Test 2: Convenience Function (3 modes)")
    print("="*60)
    
    # Three modes
    frequencies = [0.3, 0.8, 1.5]  # Hz
    damping = [0.02, 0.02, 0.03]
    
    print(f"\nDesigning multi-mode shaper for:")
    for i, (f, z) in enumerate(zip(frequencies, damping)):
        print(f"  Mode {i+1}: {f} Hz, ζ={z}")
    
    # Design with ZVD
    A, t = design_multimode_cascaded(frequencies, damping, method='ZVD')
    
    print(f"\nResult (ZVD cascaded):")
    print(f"  Total impulses: {len(A)}")
    print(f"  Total duration: {t[-1]:.3f} s")
    print(f"  First 5 amplitudes: {A[:5]}")
    print(f"  Unity gain check: {np.sum(A):.10f}")
    
    print("\n✓ Convenience function works!")
    
    return A, t


def visualize_multimode_shaper():
    """Visualize multi-mode shaper impulse sequence"""
    print("\n" + "="*60)
    print("Test 3: Visualization")
    print("="*60)
    
    frequencies = [0.3, 0.8]
    damping = [0.02, 0.02]
    
    # Compare ZV vs ZVD cascaded
    A_zv, t_zv = design_multimode_cascaded(frequencies, damping, method='ZV')
    A_zvd, t_zvd = design_multimode_cascaded(frequencies, damping, method='ZVD')
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
    
    # ZV cascaded
    ax1.stem(t_zv, A_zv, basefmt=' ', linefmt='r-', markerfmt='ro')
    ax1.set_ylabel('Amplitude')
    ax1.set_title(f'ZV Cascaded: {len(A_zv)} impulses, duration = {t_zv[-1]:.2f}s', 
                  fontweight='bold')
    ax1.grid(True, alpha=0.3)
    
    # ZVD cascaded
    ax2.stem(t_zvd, A_zvd, basefmt=' ', linefmt='b-', markerfmt='bo')
    ax2.set_xlabel('Time (s)')
    ax2.set_ylabel('Amplitude')
    ax2.set_title(f'ZVD Cascaded: {len(A_zvd)} impulses, duration = {t_zvd[-1]:.2f}s',
                  fontweight='bold')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('examples/multimode_cascaded.png', dpi=150)
    print("\n✓ Plot saved to examples/multimode_cascaded.png")
    plt.show()


if __name__ == "__main__":
    print("\nMULTI-MODE SHAPING - CASCADED CONVOLUTION TESTS\n")
    
    test_basic_convolution()
    test_convenience_function()
    visualize_multimode_shaper()
    
    print("\n" + "="*60)
    print("ALL TESTS PASSED! ✓")
    print("="*60)
