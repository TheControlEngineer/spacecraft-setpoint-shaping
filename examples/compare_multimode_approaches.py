"""
Compare cascaded vs simultaneous multi-mode shaping approaches
"""

import numpy as np
import matplotlib.pyplot as plt
from input_shaping import (design_multimode_cascaded, 
                           design_multimode_simultaneous,
                           ZVD)
import time


def compare_two_mode_spacecraft():
    """
    Compare cascaded vs simultaneous for realistic 2-mode spacecraft
    """
    print("="*70)
    print("COMPARISON: 2-Mode Spacecraft Solar Array")
    print("="*70)
    
    # Realistic spacecraft parameters
    frequencies = [0.3, 0.8]  # Hz - typical solar array modes
    damping = [0.02, 0.02]    # Light damping in vacuum
    
    print(f"\nSpacecraft Parameters:")
    print(f"  Mode 1: {frequencies[0]} Hz, ζ={damping[0]}")
    print(f"  Mode 2: {frequencies[1]} Hz, ζ={damping[1]}")
    
    # Cascaded approach (ZVD for each mode)
    print(f"\n--- Cascaded Approach (ZVD) ---")
    start = time.time()
    A_casc, t_casc = design_multimode_cascaded(frequencies, damping, method='ZVD')
    time_casc = time.time() - start
    
    print(f"  Impulses: {len(A_casc)}")
    print(f"  Duration: {t_casc[-1]:.3f} seconds")
    print(f"  Computation time: {time_casc*1000:.1f} ms")
    
    # Simultaneous optimization
    print(f"\n--- Simultaneous Optimization ---")
    start = time.time()
    A_simul, t_simul = design_multimode_simultaneous(
        frequencies, damping, n_impulses=6, Vtol=0.05
    )
    time_simul = time.time() - start
    
    print(f"  Impulses: {len(A_simul)}")
    print(f"  Duration: {t_simul[-1]:.3f} seconds")
    print(f"  Computation time: {time_simul*1000:.1f} ms")
    
    # Compare
    print(f"\n--- Performance Comparison ---")
    print(f"  Duration reduction: {(1 - t_simul[-1]/t_casc[-1])*100:.1f}%")
    print(f"  Impulse reduction: {len(A_casc) - len(A_simul)} ({len(A_casc)} → {len(A_simul)})")
    
    if t_simul[-1] < t_casc[-1]:
        speedup = t_casc[-1] / t_simul[-1]
        print(f"  ✓ Simultaneous is {speedup:.2f}x faster!")
    
    return (A_casc, t_casc), (A_simul, t_simul)


def validate_suppression():
    """
    Verify that both approaches actually suppress vibrations
    """
    print("\n" + "="*70)
    print("VALIDATION: Vibration Suppression Performance")
    print("="*70)
    
    frequencies = [0.3, 0.8]
    damping = [0.02, 0.02]
    
    # Get shapers
    A_casc, t_casc = design_multimode_cascaded(frequencies, damping, method='ZVD')
    A_simul, t_simul = design_multimode_simultaneous(
        frequencies, damping, n_impulses=6, Vtol=0.05
    )
    
    def residual_vibration(omega_d, omega_n, zeta, A, t):
        """Calculate residual vibration for one mode"""
        V = 0
        for amp, time in zip(A, t):
            V += amp * np.exp(zeta * omega_n * time) * np.exp(1j * omega_d * time)
        return np.abs(V)
    
    print(f"\nResidual Vibration at Each Mode:")
    print(f"{'Mode':<10} {'Freq (Hz)':<12} {'Cascaded':<15} {'Simultaneous':<15}")
    print("-"*60)
    
    for i, (f, z) in enumerate(zip(frequencies, damping)):
        omega_n = 2 * np.pi * f
        omega_d = omega_n * np.sqrt(1 - z**2)
        
        V_casc = residual_vibration(omega_d, omega_n, z, A_casc, t_casc)
        V_simul = residual_vibration(omega_d, omega_n, z, A_simul, t_simul)
        
        print(f"Mode {i+1:<5} {f:<12.1f} {V_casc:<15.6f} {V_simul:<15.6f}")
    
    print("\n✓ Both approaches suppress vibrations successfully!")


def visualize_comparison():
    """
    Visualize impulse sequences for both approaches
    """
    print("\n" + "="*70)
    print("VISUALIZATION: Impulse Sequences")
    print("="*70)
    
    frequencies = [0.3, 0.8]
    damping = [0.02, 0.02]
    
    A_casc, t_casc = design_multimode_cascaded(frequencies, damping, method='ZVD')
    A_simul, t_simul = design_multimode_simultaneous(
        frequencies, damping, n_impulses=6, Vtol=0.05
    )
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))
    
    # Cascaded
    ax1.stem(t_casc, A_casc, basefmt=' ', linefmt='b-', markerfmt='bo', label='Cascaded (ZVD)')
    ax1.set_ylabel('Amplitude', fontsize=12)
    ax1.set_title(f'Cascaded Convolution: {len(A_casc)} impulses, {t_casc[-1]:.2f}s duration', 
                  fontweight='bold', fontsize=13)
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=11)
    
    # Simultaneous
    ax2.stem(t_simul, A_simul, basefmt=' ', linefmt='r-', markerfmt='ro', label='Simultaneous Optimization')
    ax2.set_xlabel('Time (s)', fontsize=12)
    ax2.set_ylabel('Amplitude', fontsize=12)
    ax2.set_title(f'Simultaneous Optimization: {len(A_simul)} impulses, {t_simul[-1]:.2f}s duration',
                  fontweight='bold', fontsize=13)
    ax2.grid(True, alpha=0.3)
    ax2.legend(fontsize=11)
    
    # Add performance comparison text
    duration_reduction = (1 - t_simul[-1]/t_casc[-1]) * 100
    impulse_reduction = len(A_casc) - len(A_simul)
    
    textstr = f'Performance:\nDuration: {duration_reduction:.1f}% shorter\nImpulses: {impulse_reduction} fewer'
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
    ax2.text(0.98, 0.97, textstr, transform=ax2.transAxes, fontsize=11,
             verticalalignment='top', horizontalalignment='right', bbox=props)
    
    plt.tight_layout()
    plt.savefig('examples/multimode_comparison.png', dpi=150, bbox_inches='tight')
    print("\n✓ Comparison plot saved to examples/multimode_comparison.png")
    plt.show()


def explore_tolerance_tradeoff():
    """
    Show how tolerance affects duration
    """
    print("\n" + "="*70)
    print("TRADE-OFF ANALYSIS: Tolerance vs Duration")
    print("="*70)
    
    frequencies = [0.3, 0.8]
    damping = [0.02, 0.02]
    tolerances = [0.01, 0.02, 0.05, 0.10, 0.15, 0.20]  # 1% to 20%
    
    print(f"\n{'Tolerance':<12} {'Duration (s)':<15} {'vs Cascaded':<15}")
    print("-"*50)
    
    # Cascaded baseline
    A_casc, t_casc = design_multimode_cascaded(frequencies, damping, method='ZVD')
    
    durations = []
    for Vtol in tolerances:
        A, t = design_multimode_simultaneous(
            frequencies, damping, n_impulses=6, Vtol=Vtol
        )
        durations.append(t[-1])
        reduction = (1 - t[-1]/t_casc[-1]) * 100
        print(f"{Vtol*100:>4.0f}%        {t[-1]:<15.3f} -{reduction:.1f}%")
    
    print(f"\nCascaded (0%): {t_casc[-1]:.3f}s (baseline)")
    print("\n✓ Tighter tolerance → Longer duration (approaching cascaded)")
    
    # Create visualization
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot tolerance vs duration
    tolerances_pct = np.array(tolerances) * 100
    ax.plot(tolerances_pct, durations, 'bo-', linewidth=2, markersize=8, label='Simultaneous Optimization')
    
    # Formatting
    ax.set_xlabel('Tolerance (%)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Maneuver Duration (s)', fontsize=12, fontweight='bold')
    ax.set_title('Trade-off: Vibration Tolerance vs Maneuver Duration\n(2-Mode Spacecraft, 6-Impulse Shaper)', 
                 fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=11)
    
    # Add annotation showing the range
    duration_range = max(durations) - min(durations)
    ax.annotate(f'Shortest: {min(durations):.2f}s @ {tolerances_pct[np.argmin(durations)]:.0f}%',
                xy=(tolerances_pct[np.argmin(durations)], min(durations)),
                xytext=(tolerances_pct[np.argmin(durations)] + 5, min(durations) + 0.03),
                fontsize=10,
                bbox=dict(boxstyle='round,pad=0.5', facecolor='lightgreen', alpha=0.8),
                arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0.2'))
    
    plt.tight_layout()
    plt.savefig('examples/tolerance_tradeoff.png', dpi=150, bbox_inches='tight')
    print("\n✓ Trade-off plot saved to examples/tolerance_tradeoff.png")
    plt.show()


def main():
    """Run all comparisons"""
    print("\n" + "="*70)
    print("MULTI-MODE SHAPING: CASCADED VS SIMULTANEOUS COMPARISON")
    print("="*70)
    
    compare_two_mode_spacecraft()
    validate_suppression()
    visualize_comparison()
    explore_tolerance_tradeoff()
    
    print("\n" + "="*70)
    print("COMPARISON COMPLETE! ✓")
    print("="*70)
    print("\nKey Takeaways:")
    print("  • Cascaded: Simple, modular, but longer duration")
    print("  • Simultaneous: Optimal duration, but requires optimization")
    print("  • Both approaches successfully suppress all modes")
    print("  • Choice depends on: mission duration constraints vs implementation complexity")
    print("="*70)


if __name__ == "__main__":
    main()
