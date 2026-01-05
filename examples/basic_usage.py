"""
Basic Usage Examples for Input Shaping Library

This script demonstrates common use cases for the input shaping library.
"""

import numpy as np
import matplotlib.pyplot as plt
from input_shaping import ZV, ZVD, ZVDD, EI, design_shaper


def example_1_basic_shaper_design():
    """Example 1: Design a basic ZVD shaper"""
    print("="*60)
    print("Example 1: Basic Shaper Design")
    print("="*60)
    
    # System parameters (typical spacecraft solar array)
    frequency = 0.5  # Hz
    omega_n = 2 * np.pi * frequency
    zeta = 0.02  # 2% damping
    
    # Design ZVD shaper
    amplitudes, times, K = ZVD(omega_n, zeta)
    
    print(f"\nSystem Properties:")
    print(f"  Natural frequency: {frequency} Hz")
    print(f"  Damping ratio: {zeta}")
    
    print(f"\nZVD Shaper Design:")
    print(f"  Number of impulses: {len(amplitudes)}")
    print(f"  Amplitudes: {amplitudes}")
    print(f"  Times: {times}")
    print(f"  Total duration: {times[-1]:.3f} seconds")
    print(f"  Sum of amplitudes: {np.sum(amplitudes):.6f} (should be 1.0)")


def example_2_compare_all_shapers():
    """Example 2: Compare all shaper types"""
    print("\n" + "="*60)
    print("Example 2: Comparing All Shapers")
    print("="*60)
    
    omega_n = np.pi
    zeta = 0.02
    
    # Get shaper designs (need to handle different return signatures)
    A_zv, t_zv, K = ZV(omega_n, zeta)
    A_zvd, t_zvd, K = ZVD(omega_n, zeta)
    A_zvdd, t_zvdd = ZVDD(omega_n, zeta)
    A_ei, t_ei = EI(omega_n, zeta, Vtol=0.10)
    
    shapers = {
        'ZV': (A_zv, t_zv),
        'ZVD': (A_zvd, t_zvd),
        'ZVDD': (A_zvdd, t_zvdd),
        'EI': (A_ei, t_ei)
    }
    
    print(f"\n{'Shaper':<8} {'Impulses':<10} {'Duration (s)':<15} {'Amplitudes'}")
    print("-"*80)
    
    for name, (A, t) in shapers.items():
        amp_str = '[' + ', '.join([f'{a:.3f}' for a in A]) + ']'
        print(f"{name:<8} {len(A):<10} {t[-1]:<15.3f} {amp_str}")


def example_3_visualize_impulse_sequence():
    """Example 3: Visualize impulse sequences"""
    print("\n" + "="*60)
    print("Example 3: Visualizing Impulse Sequences")
    print("="*60)
    
    omega_n = np.pi
    zeta = 0.02
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle('Input Shaper Impulse Sequences', fontsize=14, fontweight='bold')
    
    # Get shaper designs
    A_zv, t_zv, K = ZV(omega_n, zeta)
    A_zvd, t_zvd, K = ZVD(omega_n, zeta)
    A_zvdd, t_zvdd = ZVDD(omega_n, zeta)
    A_ei, t_ei = EI(omega_n, zeta, Vtol=0.10)
    
    shapers = [
        ('ZV', (A_zv, t_zv)),
        ('ZVD', (A_zvd, t_zvd)),
        ('ZVDD', (A_zvdd, t_zvdd)),
        ('EI (10%)', (A_ei, t_ei))
    ]
    
    for ax, (name, (A, t)) in zip(axes.flat, shapers):
        # Plot impulses as stem plot
        ax.stem(t, A, basefmt=' ', linefmt='C0-', markerfmt='C0o')
        
        # Formatting
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Amplitude')
        ax.set_title(f'{name} Shaper')
        ax.grid(True, alpha=0.3)
        ax.set_ylim([0, 0.6])
        
        # Add text with shaper info
        info_text = f"Impulses: {len(A)}\nDuration: {t[-1]:.2f}s"
        ax.text(0.95, 0.95, info_text, 
                transform=ax.transAxes,
                verticalalignment='top',
                horizontalalignment='right',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig('examples/shaper_impulses.png', dpi=150, bbox_inches='tight')
    print("\n✓ Plot saved to examples/shaper_impulses.png")
    plt.show()


def example_4_frequency_response():
    """Example 4: Plot frequency response curves"""
    print("\n" + "="*60)
    print("Example 4: Frequency Response Analysis")
    print("="*60)
    
    omega_n = np.pi
    zeta = 0.02
    omega_d = omega_n * np.sqrt(1 - zeta**2)
    
    # Design shapers
    A_zv, t_zv, K = ZV(omega_n, zeta)
    A_zvd, t_zvd, K = ZVD(omega_n, zeta)
    A_zvdd, t_zvdd = ZVDD(omega_n, zeta)
    Vtol = 0.10
    A_ei, t_ei = EI(omega_n, zeta, Vtol=Vtol)
    
    # Frequency sweep - sweep omega_d (damped frequency) since that's what the shaper is designed for
    freq_errors = np.linspace(-0.30, 0.30, 500)
    omega_d_sweep = omega_d * (1 + freq_errors)
    
    def residual_vibration(omega_d_actual, omega_d_design, A, t, zeta):
        """
        Calculate residual vibration at given damped frequency.
        
        Uses Singer & Seering (1990) formula:
            V = |Σ A_i * exp(+ζωn*t_i) * exp(j*ω*t_i)|
        
        Note: The POSITIVE sign in the damping exponent is critical!
        This accounts for how much earlier impulses have decayed relative
        to the final impulse time reference point.
        """
        # Calculate corresponding natural frequency for the actual system
        omega_n_actual = omega_d_actual / np.sqrt(1 - zeta**2)
        
        V = 0
        for amp, time in zip(A, t):
            # POSITIVE exponent per Singer & Seering formula
            damping = np.exp(+zeta * omega_n_actual * time)
            # Oscillation at actual omega_d
            oscillation = np.exp(1j * omega_d_actual * time)
            V += amp * damping * oscillation
        return np.abs(V)
    
    # Calculate responses
    V_zv = [residual_vibration(w, omega_d, A_zv, t_zv, zeta) for w in omega_d_sweep]
    V_zvd = [residual_vibration(w, omega_d, A_zvd, t_zvd, zeta) for w in omega_d_sweep]
    V_zvdd = [residual_vibration(w, omega_d, A_zvdd, t_zvdd, zeta) for w in omega_d_sweep]
    V_ei = [residual_vibration(w, omega_d, A_ei, t_ei, zeta) for w in omega_d_sweep]
    
    # Plot
    plt.figure(figsize=(10, 6))
    plt.plot(freq_errors * 100, V_zv, 'r-', linewidth=2, label='ZV')
    plt.plot(freq_errors * 100, V_zvd, 'b-', linewidth=2, label='ZVD')
    plt.plot(freq_errors * 100, V_zvdd, 'g-', linewidth=2, label='ZVDD')
    plt.plot(freq_errors * 100, V_ei, 'm-', linewidth=2, label=f'EI ({int(Vtol * 100)}%)')
    
    plt.axhline(y=0.05, color='k', linestyle='--', linewidth=1, alpha=0.5, label='5% tolerance')
    plt.axhline(y=0.10, color='k', linestyle=':', linewidth=1, alpha=0.5, label='10% tolerance')
    plt.axvline(x=-20, color='gray', linestyle=':', alpha=0.3)
    plt.axvline(x=20, color='gray', linestyle=':', alpha=0.3)
    
    plt.xlabel('Frequency Error (%)', fontsize=12)
    plt.ylabel('Residual Vibration Amplitude', fontsize=12)
    plt.title('Robustness to Frequency Uncertainty', fontsize=14, fontweight='bold')
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.xlim([-30, 30])
    plt.ylim([0, 0.4])
    
    plt.tight_layout()
    plt.savefig('examples/frequency_response.png', dpi=150, bbox_inches='tight')
    print("\n✓ Plot saved to examples/frequency_response.png")
    
    # Verify zero at nominal
    print(f"\nVerification - Residual vibration at 0% error:")
    print(f"  ZV:   {V_zv[len(V_zv)//2]:.2e} (should be ~0)")
    print(f"  ZVD:  {V_zvd[len(V_zvd)//2]:.2e} (should be ~0)")
    print(f"  ZVDD: {V_zvdd[len(V_zvdd)//2]:.2e} (should be ~0)")
    print(f"  EI:   {V_ei[len(V_ei)//2]:.2e} (should be ~0, target ≤ {Vtol:.0%} at edges)")
    
    plt.show()


def example_5_time_domain_response():
    """Example 5: Simulate time-domain response to shaped input"""
    print("\n" + "="*60)
    print("Example 5: Time Domain Response Simulation")
    print("="*60)
    
    # System parameters
    omega_n = np.pi
    zeta = 0.02
    omega_d = omega_n * np.sqrt(1 - zeta**2)
    
    # Design ALL shapers
    A_unshaped = np.array([1.0])
    t_unshaped = np.array([0.0])
    
    A_zv, t_zv, K = ZV(omega_n, zeta)
    A_zvd, t_zvd, K = ZVD(omega_n, zeta)
    A_zvdd, t_zvdd = ZVDD(omega_n, zeta)
    A_ei, t_ei = EI(omega_n, zeta, Vtol=0.10)
    
    # Time vector
    t_sim = np.linspace(0, 12, 2400)  # Extended to show ZVDD settling
    
    def simulate_response(A, t_impulse, t_sim):
        """Simulate response to impulse sequence"""
        x = np.zeros_like(t_sim)
        for amp, t_i in zip(A, t_impulse):
            mask = t_sim >= t_i
            dt = t_sim[mask] - t_i
            x[mask] += (amp / omega_d) * np.exp(-zeta * omega_n * dt) * np.sin(omega_d * dt)
        return x
    
    # Simulate ALL responses
    x_unshaped = simulate_response(A_unshaped, t_unshaped, t_sim)
    x_zv = simulate_response(A_zv, t_zv, t_sim)
    x_zvd = simulate_response(A_zvd, t_zvd, t_sim)
    x_zvdd = simulate_response(A_zvdd, t_zvdd, t_sim)
    x_ei = simulate_response(A_ei, t_ei, t_sim)
    
    # Plot - now with 5 subplots
    fig, axes = plt.subplots(5, 1, figsize=(12, 14))
    
    responses = [
        (x_unshaped, 'Unshaped Response (Single Impulse)', 'k'),
        (x_zv, 'ZV Shaped Response', 'r'),
        (x_zvd, 'ZVD Shaped Response', 'b'),
        (x_zvdd, 'ZVDD Shaped Response', 'g'),
        (x_ei, 'EI Shaped Response (10% tol)', 'm')
    ]
    
    for ax, (x, title, color) in zip(axes, responses):
        ax.plot(t_sim, x, color=color, linewidth=1.5)
        ax.set_ylabel('Displacement')
        ax.set_title(title, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.set_xlim([0, 12])
    
    axes[-1].set_xlabel('Time (s)')
    
    plt.tight_layout()
    plt.savefig('examples/time_domain_response.png', dpi=150, bbox_inches='tight')
    print("\n✓ Plot saved to examples/time_domain_response.png")
    plt.show()
    
    # Print settling metrics
    print("\nSettling Time Analysis (to 2% of peak):")
    threshold = 0.02 * np.max(np.abs(x_unshaped))
    
    def settling_time(x, t):
        """Find time when response settles below threshold"""
        # Look after the shaper duration
        start_idx = len(t) // 3  # Start checking after 1/3 of simulation
        settled = np.abs(x[start_idx:]) < threshold
        if np.any(settled):
            return t[start_idx:][np.where(settled)[0][0]]
        return np.inf
    
    print(f"  Unshaped: Never settles")
    print(f"  ZV:       {settling_time(x_zv, t_sim):.2f} seconds")
    print(f"  ZVD:      {settling_time(x_zvd, t_sim):.2f} seconds")
    print(f"  ZVDD:     {settling_time(x_zvdd, t_sim):.2f} seconds")
    print(f"  EI:       {settling_time(x_ei, t_sim):.2f} seconds")


def example_6_convenience_function():
    """Example 6: Using the design_shaper convenience function"""
    print("\n" + "="*60)
    print("Example 6: Convenience Function Usage")
    print("="*60)
    
    omega_n = np.pi
    zeta = 0.02
    
    print("\nUsing design_shaper() function:")
    print("-" * 40)
    
    # Design different shapers with one function
    for method in ['ZV', 'ZVD', 'ZVDD']:
        A, t = design_shaper(omega_n, zeta, method=method)
        print(f"\n{method}:")
        print(f"  Impulses: {len(A)}")
        print(f"  Duration: {t[-1]:.3f}s")
        print(f"  First 3 amplitudes: {A[:3]}")
    
    # EI with custom parameters
    A, t = design_shaper(omega_n, zeta, method='EI', Vtol=0.15, tol_band=0.25)
    print(f"\nEI (custom: Vtol=15%, band=±25%):")
    print(f"  Impulses: {len(A)}")
    print(f"  Duration: {t[-1]:.3f}s")


def main():
    """Run all examples"""
    print("\n" + "="*60)
    print("INPUT SHAPING LIBRARY - USAGE EXAMPLES")
    print("="*60)
    
    example_1_basic_shaper_design()
    example_2_compare_all_shapers()
    example_3_visualize_impulse_sequence()
    example_4_frequency_response()
    example_5_time_domain_response()
    example_6_convenience_function()
    
    print("\n" + "="*60)
    print("All examples completed successfully! ✓")
    print("="*60)


if __name__ == "__main__":
    main()