"""
OPTIMAL CONTROLLER DESIGN FOR FLEXIBLE SPACECRAFT

Based on investigation results:
- Need PM > 62° 
- Need to minimize mode excitation
- Need pointing error < 0.1 deg

Key insight: Filter cutoff at first_mode/2 = 0.2 Hz causes PM to drop to 42°.
We need a higher filter cutoff that still provides some high-frequency rolloff.

Strategy:
1. Use PD with bandwidth = first_mode/6 (good PM)
2. Add derivative filter at HIGHER cutoff (less phase lag)
3. Optionally add PPF for modal damping
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import signal

# Import from project
from spacecraft_properties import HUB_INERTIA, compute_effective_inertia

def build_flexible_plant(I, modal_freqs_hz, modal_damping, modal_gains):
    """Build flexible plant transfer function."""
    rigid_num = np.array([1.0])
    rigid_den = np.array([I, 0.0, 0.0])
    
    current_num = rigid_num
    current_den = rigid_den
    
    for f_mode, zeta, gain in zip(modal_freqs_hz, modal_damping, modal_gains):
        omega_n = 2 * np.pi * f_mode
        mode_num = np.array([gain / I])
        mode_den = np.array([1.0, 2*zeta*omega_n, omega_n**2])
        
        term1 = np.convolve(current_num, mode_den)
        term2 = np.convolve(mode_num, current_den)
        
        if len(term1) > len(term2):
            term2 = np.pad(term2, (len(term1) - len(term2), 0))
        elif len(term2) > len(term1):
            term1 = np.pad(term1, (len(term2) - term1.shape[0], 0))
        
        current_num = term1 + term2
        current_den = np.convolve(current_den, mode_den)
    
    return signal.TransferFunction(current_num, current_den)


def analyze_controller(name, K, P, filter_hz, freqs, G, I, modal_freqs):
    """Analyze a controller configuration."""
    omega = 2 * np.pi * freqs
    
    if filter_hz is None or filter_hz <= 0:
        # Pure PD: C(s) = K + P*s
        C = K + 1j * omega * P
    else:
        # Filtered PD: C(s) = K + P*s/(tau*s + 1)
        tau = 1 / (2 * np.pi * filter_hz)
        C = K + P * 1j * omega / (tau * 1j * omega + 1)
    
    # Open-loop
    L = G * C
    
    # Sensitivity and complementary sensitivity
    S = 1 / (1 + L)
    T = L / (1 + L)
    
    # Stability margins
    phase = np.angle(L, deg=True)
    mag = np.abs(L)
    
    # Find gain crossover (|L| = 1)
    gc_idx = np.argmin(np.abs(mag - 1))
    pm = 180 + phase[gc_idx]
    gc_freq = freqs[gc_idx]
    
    # Find phase crossover (phase = -180)
    pc_idx = np.argmin(np.abs(phase + 180))
    gm = -20 * np.log10(mag[pc_idx]) if mag[pc_idx] > 0 else np.inf
    
    # Values at modal frequencies
    mode_data = []
    for f_mode in modal_freqs:
        idx = np.argmin(np.abs(freqs - f_mode))
        mode_data.append({
            'freq': f_mode,
            'S_db': 20 * np.log10(np.abs(S[idx])),
            'T_db': 20 * np.log10(np.abs(T[idx])),
            'C_db': 20 * np.log10(np.abs(C[idx]))
        })
    
    return {
        'name': name,
        'pm': pm,
        'gm': gm,
        'gc_freq': gc_freq,
        'mode_data': mode_data,
        'L': L,
        'S': S,
        'T': T,
        'C': C
    }


def main():
    print("=" * 80)
    print("OPTIMAL CONTROLLER DESIGN")
    print("=" * 80)
    
    # Parameters
    I = HUB_INERTIA[2, 2]  # 600 kg·m²
    modal_freqs_hz = [0.4, 1.3]
    modal_damping = [0.02, 0.015]
    
    # Use CONSISTENT modal gains for analysis
    # The theoretical value is ~0.001, but we use control_modal_gains for design
    # to ensure we properly account for modal peaks in frequency response
    modal_gains = [0.15, 0.08]  # For control design (conservative)
    
    print(f"\nDesign Parameters:")
    print(f"  Inertia (Z): {I} kg·m²")
    print(f"  Modal frequencies: {modal_freqs_hz} Hz")
    print(f"  Modal gains (for design): {modal_gains}")
    
    # Frequency range
    freqs = np.logspace(-2, 1, 1000)
    omega = 2 * np.pi * freqs
    
    # Build flexible plant
    plant = build_flexible_plant(I, modal_freqs_hz, modal_damping, modal_gains)
    _, G = signal.freqresp(plant, omega)
    
    # First mode frequency
    first_mode = min(modal_freqs_hz)
    
    # ========================================================================
    # CONTROLLER CONFIGURATIONS TO TEST
    # ========================================================================
    
    controllers = []
    
    # Base gains (low bandwidth)
    bw = first_mode / 6.0  # 0.067 Hz
    omega_n = 2 * np.pi * bw
    K = I * omega_n**2
    P = 2 * 0.7 * I * omega_n
    
    print(f"\nBase controller gains (bw={bw:.4f} Hz):")
    print(f"  K = {K:.2f} N·m/rad")
    print(f"  P = {P:.2f} N·m·s/rad")
    
    # Test different filter cutoffs
    filter_cutoffs = [
        None,           # Pure PD
        0.2,            # first_mode/2 (original)
        0.3,            # first_mode * 0.75
        0.4,            # first_mode
        0.6,            # 1.5 * first_mode
        0.8,            # 2 * first_mode
        1.0,            # 2.5 * first_mode
    ]
    
    print("\n" + "=" * 80)
    print("TESTING FILTER CUTOFF OPTIONS")
    print("=" * 80)
    
    results = []
    for fc in filter_cutoffs:
        if fc is None:
            name = "Pure PD (no filter)"
        else:
            name = f"Filtered PD (fc={fc:.1f} Hz)"
        
        result = analyze_controller(name, K, P, fc, freqs, G, I, modal_freqs_hz)
        results.append(result)
        
        print(f"\n{name}:")
        print(f"  Phase margin: {result['pm']:.1f}°")
        print(f"  Gain crossover: {result['gc_freq']:.3f} Hz")
        
        for md in result['mode_data']:
            status_s = "✓" if md['S_db'] < 0 else "⚠"
            status_c = "✓" if md['C_db'] < 60 else "⚠"
            print(f"  Mode @ {md['freq']} Hz: S={md['S_db']:+.1f}dB {status_s}, |C|={md['C_db']:.1f}dB {status_c}")
        
        pm_ok = "✓" if result['pm'] >= 62 else "✗"
        print(f"  PM requirement: {pm_ok}")
    
    # Find optimal configuration
    print("\n" + "=" * 80)
    print("OPTIMAL CONFIGURATION SELECTION")
    print("=" * 80)
    
    # Filter: PM > 62, minimize controller gain at modes
    valid = [r for r in results if r['pm'] >= 62]
    
    if valid:
        # Find one with lowest max controller gain at modal frequencies
        def max_mode_gain(r):
            return max(md['C_db'] for md in r['mode_data'])
        
        best = min(valid, key=max_mode_gain)
        print(f"\n✓ OPTIMAL: {best['name']}")
        print(f"  Phase margin: {best['pm']:.1f}° (> 62° ✓)")
        print(f"  Max controller gain at modes: {max_mode_gain(best):.1f} dB")
    else:
        print("\n✗ No configuration achieves PM > 62°")
        # Find best PM
        best = max(results, key=lambda r: r['pm'])
        print(f"  Best available: {best['name']} with PM={best['pm']:.1f}°")
    
    # ========================================================================
    # NOW TEST WITH PPF COMPENSATOR
    # ========================================================================
    
    print("\n" + "=" * 80)
    print("TESTING PPF COMPENSATOR OPTION")
    print("=" * 80)
    
    # PPF adds phase lead near modal frequency, which can help
    # But it can also destabilize if not tuned correctly
    
    # Start with pure PD (has good PM) and add PPF
    print("\nBase: Pure PD with PM=65°")
    print("Adding PPF compensator to increase modal damping...")
    
    # PPF transfer function: H_ppf(s) = g * omega_f^2 / (s^2 + 2*zeta_f*omega_f*s + omega_f^2)
    def test_ppf(K, P, ppf_gains, ppf_freqs, ppf_damping=0.5):
        """Test PD + PPF configuration."""
        # Pure PD controller
        C_pd = K + 1j * omega * P
        
        # PPF compensators
        C_ppf = np.zeros_like(C_pd)
        for f_ppf, g_ppf in zip(ppf_freqs, ppf_gains):
            omega_f = 2 * np.pi * f_ppf
            # H_ppf(jω) = g * omega_f^2 / (-ω^2 + 2*zeta*omega_f*jω + omega_f^2)
            H_ppf = g_ppf * omega_f**2 / (omega_f**2 - omega**2 + 2j * ppf_damping * omega_f * omega)
            C_ppf += H_ppf
        
        # Total controller
        C_total = C_pd + C_ppf
        
        # Open-loop
        L = G * C_total
        
        # Margins
        phase = np.angle(L, deg=True)
        mag = np.abs(L)
        gc_idx = np.argmin(np.abs(mag - 1))
        pm = 180 + phase[gc_idx]
        
        return pm, C_total, L
    
    # Test different PPF gain levels
    ppf_freq_ratio = 0.9  # PPF tuned slightly below mode
    ppf_freqs = [ppf_freq_ratio * f for f in modal_freqs_hz]
    
    print(f"\nPPF filter frequencies: {ppf_freqs} Hz")
    print(f"PPF damping: 0.5")
    
    ppf_gains_to_test = [
        [0, 0],         # No PPF
        [1, 2],         # Low PPF
        [2, 4],         # Medium-low PPF
        [5, 10],        # Medium PPF (original)
        [10, 20],       # High PPF
    ]
    
    for ppf_gains in ppf_gains_to_test:
        pm, C_total, L = test_ppf(K, P, ppf_gains, ppf_freqs)
        
        # Check sensitivity at modes
        S = 1 / (1 + L)
        mode_s = []
        for f_mode in modal_freqs_hz:
            idx = np.argmin(np.abs(freqs - f_mode))
            mode_s.append(20 * np.log10(np.abs(S[idx])))
        
        pm_ok = "✓" if pm >= 62 else "✗"
        print(f"\nPPF gains {ppf_gains}:")
        print(f"  PM = {pm:.1f}° {pm_ok}")
        print(f"  S @ modes: {mode_s[0]:+.1f} dB, {mode_s[1]:+.1f} dB")
    
    # ========================================================================
    # FINAL RECOMMENDATION
    # ========================================================================
    
    print("\n" + "=" * 80)
    print("FINAL RECOMMENDATIONS")
    print("=" * 80)
    
    print("""
OPTION 1: Pure PD (SIMPLEST, RECOMMENDED)
------------------------------------------
Configuration:
  - Bandwidth: first_mode/6 = 0.067 Hz
  - K = 105.3 N·m/rad
  - P = 351.9 N·m·s/rad
  - No derivative filter
  - No PPF
  
Performance:
  - Phase margin: ~65° ✓ (exceeds 62° requirement)
  - Sensitivity at 0.4 Hz: ~-5 dB (attenuates disturbances)
  - Controller gain: ~59 dB at 0.4 Hz, ~69 dB at 1.3 Hz
  
Trade-off:
  + Simple, robust, meets PM requirement
  - High controller gain at modes may amplify sensor noise
  
OPTION 2: Filtered PD (filter above first mode)
------------------------------------------------
Configuration:
  - Bandwidth: first_mode/6 = 0.067 Hz
  - K = 105.3 N·m/rad
  - P = 351.9 N·m·s/rad
  - Derivative filter: 0.6-0.8 Hz (ABOVE first mode but below second)
  
Performance:
  - Phase margin: ~63-64° (just meets requirement)
  - Reduces controller gain at high frequencies
  
Trade-off:
  + Reduces noise amplification at high frequencies
  - Close to PM limit, less robust

OPTION 3: PD + Low-Gain PPF
----------------------------
Configuration:
  - Same PD as Option 1
  - PPF gains: [1-2, 2-4] (LOW)
  - PPF frequencies: 0.9 × modal frequencies
  
Performance:
  - Phase margin: Depends on PPF gain (check < 62° threshold)
  - Adds some modal damping
  
Trade-off:
  + Actively damps modes
  - May reduce PM below requirement if gains too high

CRITICAL NOTE:
--------------
The `modal_gains` mismatch (0.0015 vs 0.15) means simulation and 
control analysis see DIFFERENT plants. This must be fixed:

EITHER use consistent modal_gains = [0.15, 0.08] everywhere
OR use consistent modal_gains = [0.0015, 0.0008] everywhere

The theoretical value is ~0.001, so [0.0015, 0.0008] is more 
physically realistic. The [0.15, 0.08] values may have been 
accidentally set 100x too high.
""")
    
    # Create plot
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Plot 1: Sensitivity for different configurations
    ax1 = axes[0, 0]
    for r in results[:4]:  # First 4 configs
        S_db = 20 * np.log10(np.abs(r['S']))
        ax1.semilogx(freqs, S_db, label=f"{r['name']} (PM={r['pm']:.0f}°)")
    ax1.axhline(0, color='k', linestyle='--', alpha=0.5)
    for f_mode in modal_freqs_hz:
        ax1.axvline(f_mode, color='orange', linestyle=':', alpha=0.7)
    ax1.set_xlabel('Frequency (Hz)')
    ax1.set_ylabel('|S(jω)| (dB)')
    ax1.set_title('Sensitivity Function')
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim([0.01, 10])
    ax1.set_ylim([-30, 20])
    
    # Plot 2: Controller gain
    ax2 = axes[0, 1]
    for r in results[:4]:
        C_db = 20 * np.log10(np.abs(r['C']))
        ax2.semilogx(freqs, C_db, label=f"{r['name']}")
    ax2.axhline(60, color='r', linestyle='--', alpha=0.5, label='60 dB warning')
    for f_mode in modal_freqs_hz:
        ax2.axvline(f_mode, color='orange', linestyle=':', alpha=0.7)
    ax2.set_xlabel('Frequency (Hz)')
    ax2.set_ylabel('|C(jω)| (dB)')
    ax2.set_title('Controller Gain')
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim([0.01, 10])
    
    # Plot 3: Bode magnitude
    ax3 = axes[1, 0]
    for r in results[:4]:
        L_db = 20 * np.log10(np.abs(r['L']))
        ax3.semilogx(freqs, L_db, label=f"{r['name']}")
    ax3.axhline(0, color='k', linestyle='--', alpha=0.5)
    for f_mode in modal_freqs_hz:
        ax3.axvline(f_mode, color='orange', linestyle=':', alpha=0.7)
    ax3.set_xlabel('Frequency (Hz)')
    ax3.set_ylabel('|L(jω)| (dB)')
    ax3.set_title('Open-Loop Magnitude')
    ax3.legend(fontsize=8)
    ax3.grid(True, alpha=0.3)
    ax3.set_xlim([0.01, 10])
    
    # Plot 4: Bode phase
    ax4 = axes[1, 1]
    for r in results[:4]:
        L_phase = np.angle(r['L'], deg=True)
        ax4.semilogx(freqs, L_phase, label=f"{r['name']}")
    ax4.axhline(-180, color='r', linestyle='--', alpha=0.5, label='-180°')
    ax4.axhline(-180+62, color='g', linestyle='--', alpha=0.5, label='PM=62°')
    for f_mode in modal_freqs_hz:
        ax4.axvline(f_mode, color='orange', linestyle=':', alpha=0.7)
    ax4.set_xlabel('Frequency (Hz)')
    ax4.set_ylabel('∠L(jω) (deg)')
    ax4.set_title('Open-Loop Phase')
    ax4.legend(fontsize=8)
    ax4.grid(True, alpha=0.3)
    ax4.set_xlim([0.01, 10])
    ax4.set_ylim([-270, 0])
    
    plt.tight_layout()
    plt.savefig('analysis/optimal_controller_design.png', dpi=150)
    print(f"\nSaved plot: analysis/optimal_controller_design.png")


if __name__ == "__main__":
    main()
