"""
Input Shaping Module
====================

This module provides various input shaping techniques for vibration suppression
in flexible spacecraft systems. Input shapers are used to reduce residual vibrations
by convolving a command signal with a sequence of impulses.

Available Shapers:
------------------
- ZV (Zero Vibration): 2-impulse shaper that eliminates vibration at nominal frequency
- ZVD (Zero Vibration Derivative): 3-impulse shaper with robustness to frequency uncertainty
- ZVDD (Zero Vibration Double Derivative): 4-impulse shaper with enhanced robustness
- EI (Extra-Insensitive): 3-impulse shaper optimized for specific frequency uncertainty band

References:
-----------
Singer, N. C., & Seering, W. P. (1990). Preshaping command inputs to reduce system 
vibration. Journal of Dynamic Systems, Measurement, and Control, 112(1), 76-82.

Author: Refactored for spacecraft_input_shaping project
Date: January 2026
"""

import numpy as np
from scipy.optimize import minimize
from typing import Tuple


# ==============================================================================
# BASIC INPUT SHAPERS
# ==============================================================================

def ZV(omega_n, zeta):
    """
    Zero Vibration (ZV) input shaper.
    
    The ZV shaper is the simplest input shaper, using two impulses to cancel
    vibration at the nominal frequency. It provides no robustness to modeling
    errors but has the shortest duration.
    
    Parameters
    ----------
    omega_n : float
        Natural frequency of the system (rad/s).
    zeta : float
        Damping ratio of the system (dimensionless).
    
    Returns
    -------
    A : ndarray
        Amplitudes of the impulses, normalized such that sum(A) = 1.
    t : ndarray
        Times of the impulses (seconds).
    K : float
        Damping factor exp(-ζπ/√(1-ζ²)), useful for shaper analysis.
    
    Notes
    -----
    The ZV shaper places impulses at t = [0, π/ωd] where ωd is the damped
    natural frequency. The amplitudes are weighted to satisfy the zero
    vibration condition at the nominal frequency.
    
    Examples
    --------
    >>> A, t, K = ZV(omega_n=1.0, zeta=0.05)
    >>> print(f"Impulse amplitudes: {A}")
    >>> print(f"Impulse times: {t}")
    """
    # Calculate damped natural frequency
    omega_d = omega_n * np.sqrt(1 - zeta**2)
    
    # Calculate damping factor
    K = np.exp(-zeta * np.pi / np.sqrt(1 - zeta**2))
    
    # Calculate normalized amplitudes
    A = np.array([1 / (1 + K), K / (1 + K)])
    
    # Calculate impulse times
    t = np.array([0, np.pi / omega_d])
    
    return A, t, K


def ZVD(omega_n, zeta):
    """
    Zero Vibration Derivative (ZVD) input shaper.
    
    The ZVD shaper uses three impulses to cancel both vibration and its
    derivative with respect to frequency at the nominal frequency. This
    provides increased robustness to frequency uncertainty compared to ZV.
    
    Parameters
    ----------
    omega_n : float
        Natural frequency of the system (rad/s).
    zeta : float
        Damping ratio of the system (dimensionless).
    
    Returns
    -------
    A : ndarray
        Amplitudes of the impulses, normalized such that sum(A) = 1.
    t : ndarray
        Times of the impulses (seconds).
    K : float
        Damping factor exp(-ζπ/√(1-ζ²)), useful for shaper analysis.
    
    Notes
    -----
    The ZVD shaper places impulses at t = [0, π/ωd, 2π/ωd] where ωd is
    the damped natural frequency. The middle impulse has twice the amplitude
    of the outer impulses due to the derivative constraint.
    
    The ZVD shaper is more robust than ZV but has a longer duration, which
    may result in slower system response.
    
    Examples
    --------
    >>> A, t, K = ZVD(omega_n=1.0, zeta=0.05)
    >>> print(f"Impulse amplitudes: {A}")
    >>> print(f"Impulse times: {t}")
    """
    # Calculate damped natural frequency
    omega_d = omega_n * np.sqrt(1 - zeta**2)
    
    # Calculate damping factor
    K = np.exp(-zeta * np.pi / np.sqrt(1 - zeta**2))
    
    # Calculate normalized amplitudes
    # Pattern: [1, 2K, K²] / (1 + 2K + K²)
    denominator = 1 + 2 * K + K**2
    A = np.array([
        1 / denominator,
        2 * K / denominator,
        K**2 / denominator
    ])
    
    # Calculate impulse times
    t = np.array([0, np.pi / omega_d, 2 * np.pi / omega_d])
    
    return A, t, K


def ZVDD(omega_n, zeta):
    """
    Zero Vibration Double Derivative (ZVDD) input shaper.
    
    The ZVDD shaper uses four impulses to cancel vibration and its first
    and second derivatives with respect to frequency. This provides the
    highest robustness to frequency uncertainty among the standard shapers.
    
    Parameters
    ----------
    omega_n : float
        Natural frequency of the system (rad/s).
    zeta : float
        Damping ratio of the system (dimensionless).
    
    Returns
    -------
    A : ndarray
        Amplitudes of the impulses, normalized such that sum(A) = 1.
    t : ndarray
        Times of the impulses (seconds).
    
    Notes
    -----
    The ZVDD shaper places impulses at t = [0, π/ωd, 2π/ωd, 3π/ωd] where
    ωd is the damped natural frequency. The amplitudes follow a binomial
    pattern weighted by powers of the damping factor K.
    
    Pattern: [1, 3K, 3K², K³] / (1 + 3K + 3K² + K³)
    
    The ZVDD shaper provides excellent robustness but has the longest duration
    among the standard shapers, which may significantly slow system response.
    
    Examples
    --------
    >>> A, t = ZVDD(omega_n=1.0, zeta=0.05)
    >>> print(f"Impulse amplitudes: {A}")
    >>> print(f"Impulse times: {t}")
    """
    # Calculate damped natural frequency
    omega_d = omega_n * np.sqrt(1 - zeta**2)
    
    # Calculate damping factor
    K = np.exp(-zeta * np.pi / np.sqrt(1 - zeta**2))
    
    # Calculate impulse times (evenly spaced at half-periods)
    t1 = np.pi / omega_d
    t2 = 2 * t1
    t3 = 3 * t1
    t = np.array([0, t1, t2, t3])
    
    # Calculate normalized amplitudes following binomial pattern
    # Pattern: [1, 3K, 3K², K³] / (1 + 3K + 3K² + K³)
    denominator = 1 + 3*K + 3*K**2 + K**3
    A = np.array([
        1 / denominator,
        3*K / denominator,
        3*K**2 / denominator,
        K**3 / denominator
    ])
    
    return A, t


# ==============================================================================
# ADVANCED INPUT SHAPERS
# ==============================================================================

def EI(omega_n, zeta, Vtol, tol_band=0.20):
    """
    Extra-Insensitive (EI) input shaper.
    
    The EI shaper is designed using optimization to maintain residual vibration
    below a specified tolerance (Vtol) across a frequency uncertainty band of
    ±tol_band. Unlike the ZV family, the EI shaper directly targets a specific
    uncertainty range, allowing for customization based on system requirements.
    
    Parameters
    ----------
    omega_n : float
        Nominal natural frequency of the system (rad/s).
    zeta : float
        Damping ratio of the system (dimensionless).
    Vtol : float
        Maximum allowable residual vibration amplitude (e.g., 0.05 for 5%).
        This defines the vibration suppression requirement at the edges of
        the frequency uncertainty band.
    tol_band : float, optional
        Fractional frequency uncertainty as a decimal (default: 0.20 for ±20%).
        The shaper will maintain V < Vtol over [ωn(1-tol_band), ωn(1+tol_band)].
    
    Returns
    -------
    amplitudes : ndarray
        Impulse amplitudes [A1, A2, A3], normalized such that sum = 1.
    times : ndarray
        Impulse times [0, t2, t3] in seconds.
    
    Notes
    -----
    The EI shaper is computed by solving an optimization problem that:
    1. Minimizes the shaper duration (t3)
    2. Ensures unity gain (sum of amplitudes = 1)
    3. Achieves zero vibration at nominal frequency
    4. Maintains vibration = Vtol at frequency band edges
    
    The optimization uses SLSQP (Sequential Least Squares Programming) to
    solve this constrained nonlinear problem. Initial guess is based on
    ZVD-like spacing.
    
    Optimization Constraints:
    - Unity gain: A1 + A2 + A3 = 1
    - Zero vibration at nominal: V(ωn) = 0
    - Tolerance at lower edge: V(ωn(1-tol_band)) = Vtol
    - Tolerance at upper edge: V(ωn(1+tol_band)) = Vtol
    
    Examples
    --------
    >>> # Design shaper for 5% max vibration over ±20% frequency uncertainty
    >>> A, t = EI(omega_n=1.0, zeta=0.05, Vtol=0.05, tol_band=0.20)
    >>> print(f"Impulse amplitudes: {A}")
    >>> print(f"Impulse times: {t}")
    
    >>> # Design for tighter frequency band
    >>> A, t = EI(omega_n=1.0, zeta=0.05, Vtol=0.05, tol_band=0.10)
    
    References
    ----------
    Singer, N. C., & Seering, W. P. (1990). Preshaping command inputs to 
    reduce system vibration. ASME Journal of Dynamic Systems, Measurement, 
    and Control.
    """
    # Calculate damped natural frequency
    omega_d = omega_n * np.sqrt(1 - zeta**2)
    
    # Define frequency points for constraints
    omega_low = omega_n * (1 - tol_band)   # Lower edge of uncertainty band
    omega_nom = omega_n                     # Nominal frequency
    omega_high = omega_n * (1 + tol_band)  # Upper edge of uncertainty band
    
    def residual_vibration(omega, A1, A2, A3, t2, t3):
        """
        Calculate residual vibration amplitude at a given frequency.
        
        The residual vibration for a 3-impulse shaper is given by:
        V(ω) = |∑ Ai * exp(-ζωti) * exp(jωti)|
        
        For impulses at times [0, t2, t3], this becomes:
        V(ω) = |A1 + A2*exp(-ζωt2)*exp(jωt2) + A3*exp(-ζωt3)*exp(jωt3)|
        
        Parameters
        ----------
        omega : float
            Frequency at which to evaluate vibration (rad/s).
        A1, A2, A3 : float
            Impulse amplitudes.
        t2, t3 : float
            Times of second and third impulses (first is at t=0).
        
        Returns
        -------
        float
            Magnitude of residual vibration at frequency omega.
        """
        # Complex exponentials with damping term
        z1 = A1  # First impulse at t=0
        z2 = A2 * np.exp(-zeta * omega * t2) * np.exp(1j * omega * t2)
        z3 = A3 * np.exp(-zeta * omega * t3) * np.exp(1j * omega * t3)
        
        # Return magnitude of complex sum
        return np.abs(z1 + z2 + z3)
    
    def objective(x):
        """
        Objective function: minimize shaper duration.
        
        Parameters
        ----------
        x : array-like
            Design variables [A1, A2, A3, t2, t3].
        
        Returns
        -------
        float
            Shaper duration (time of last impulse).
        """
        A1, A2, A3, t2, t3 = x
        return t3
    
    def constraint_unity_gain(x):
        """
        Unity gain constraint: sum of amplitudes must equal 1.
        
        This ensures the shaper doesn't amplify or attenuate the input signal.
        """
        A1, A2, A3, t2, t3 = x
        return A1 + A2 + A3 - 1.0
    
    def constraint_zero_at_nominal(x):
        """
        Zero vibration constraint at nominal frequency.
        
        This ensures complete vibration cancellation when the system frequency
        matches the design frequency exactly.
        """
        A1, A2, A3, t2, t3 = x
        return residual_vibration(omega_nom, A1, A2, A3, t2, t3)
    
    def constraint_vtol_at_low(x):
        """
        Vibration tolerance constraint at lower frequency edge.
        
        Ensures V(ωn(1-tol_band)) = Vtol, defining the robustness envelope.
        """
        A1, A2, A3, t2, t3 = x
        return residual_vibration(omega_low, A1, A2, A3, t2, t3) - Vtol
    
    def constraint_vtol_at_high(x):
        """
        Vibration tolerance constraint at upper frequency edge.
        
        Ensures V(ωn(1+tol_band)) = Vtol, completing the robustness envelope.
        """
        A1, A2, A3, t2, t3 = x
        return residual_vibration(omega_high, A1, A2, A3, t2, t3) - Vtol
    
    # Initial guess: Use ZVD-like spacing and amplitudes
    t2_init = np.pi / omega_d
    t3_init = 2 * np.pi / omega_d
    A1_init, A2_init, A3_init = 0.25, 0.5, 0.25
    x0 = [A1_init, A2_init, A3_init, t2_init, t3_init]
    
    # Define bounds for design variables
    # Amplitudes: [0, 1], Times: reasonable positive values
    bounds = [
        (0, 1),      # A1
        (0, 1),      # A2
        (0, 1),      # A3
        (0.1, 10),   # t2
        (0.2, 20)    # t3
    ]
    
    # Define equality constraints
    constraints = [
        {'type': 'eq', 'fun': constraint_unity_gain},
        {'type': 'eq', 'fun': constraint_zero_at_nominal},
        {'type': 'eq', 'fun': constraint_vtol_at_low},
        {'type': 'eq', 'fun': constraint_vtol_at_high}
    ]
    
    # Callback for monitoring optimization progress
    iteration_count = [0]
    
    def callback(x):
        """Print progress every 50 iterations."""
        iteration_count[0] += 1
        if iteration_count[0] % 50 == 0:
            A1, A2, A3, t2, t3 = x
            V_nom = residual_vibration(omega_nom, A1, A2, A3, t2, t3)
            V_low = residual_vibration(omega_low, A1, A2, A3, t2, t3)
            V_high = residual_vibration(omega_high, A1, A2, A3, t2, t3)
            print(f"Iter {iteration_count[0]}: V_nom={V_nom:.4f}, "
                  f"V_low={V_low:.4f}, V_high={V_high:.4f}")
    
    # Solve the optimization problem
    result = minimize(
        objective,
        x0,
        method='SLSQP',
        bounds=bounds,
        constraints=constraints,
        options={'maxiter': 1000, 'ftol': 1e-9},
        callback=callback
    )
    
    # Check convergence
    if not result.success:
        print(f"Warning: EI optimization did not converge. "
              f"Message: {result.message}")
    
    # Extract optimized parameters
    A1, A2, A3, t2, t3 = result.x
    
    # Format output
    amplitudes = np.array([A1, A2, A3])
    times = np.array([0, t2, t3])
    
    return amplitudes, times


# ==============================================================================
# UTILITY FUNCTIONS
# ==============================================================================

def design_shaper(omega_n: float,
                  zeta: float,
                  method: str = 'ZVD',
                  **kwargs) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convenience function to design an input shaper.
    
    Parameters
    ----------
    omega_n : float
        Natural frequency (rad/s)
    zeta : float
        Damping ratio
    method : str
        Shaper type: 'ZV', 'ZVD', 'ZVDD', or 'EI'
    **kwargs
        Additional arguments passed to EI (Vtol, tol_band)
    
    Returns
    -------
    amplitudes : np.ndarray
        Impulse amplitudes
    times : np.ndarray
        Impulse times
        
    Examples
    --------
    >>> A, t = design_shaper(omega_n=np.pi, zeta=0.02, method='ZVD')
    """
    method = method.upper()
    
    if method == 'ZV':
        A, t, K = ZV(omega_n, zeta)
        return A, t
    elif method == 'ZVD':
        A, t, K = ZVD(omega_n, zeta)
        return A, t
    elif method == 'ZVDD':
        return ZVDD(omega_n, zeta)
    elif method == 'EI':
        return EI(omega_n, zeta, **kwargs)
    else:
        raise ValueError(f"Unknown shaper method: {method}. Use 'ZV', 'ZVD', 'ZVDD', or 'EI'")


def get_shaper_info():
    """
    Return a dictionary with information about all available shapers.
    
    Returns
    -------
    dict
        Dictionary mapping shaper names to their descriptions and properties.
    
    Examples
    --------
    >>> info = get_shaper_info()
    >>> for name, details in info.items():
    ...     print(f"{name}: {details['description']}")
    """
    return {
        'ZV': {
            'name': 'Zero Vibration',
            'impulses': 2,
            'description': 'Simplest shaper, no robustness',
            'parameters': ['omega_n', 'zeta']
        },
        'ZVD': {
            'name': 'Zero Vibration Derivative',
            'impulses': 3,
            'description': 'Moderate robustness to frequency uncertainty',
            'parameters': ['omega_n', 'zeta']
        },
        'ZVDD': {
            'name': 'Zero Vibration Double Derivative',
            'impulses': 4,
            'description': 'High robustness to frequency uncertainty',
            'parameters': ['omega_n', 'zeta']
        },
        'EI': {
            'name': 'Extra-Insensitive',
            'impulses': 3,
            'description': 'Optimized for specific frequency uncertainty band',
            'parameters': ['omega_n', 'zeta', 'Vtol', 'tol_band']
        }
    }


if __name__ == '__main__':
    """
    Example usage and basic testing of all shapers.
    """
    print("=" * 70)
    print("INPUT SHAPING MODULE - EXAMPLE USAGE")
    print("=" * 70)
    
    # System parameters
    omega_n = 1.0  # rad/s
    zeta = 0.05    # 5% damping
    
    print(f"\nSystem Parameters:")
    print(f"  Natural Frequency: {omega_n} rad/s")
    print(f"  Damping Ratio: {zeta}")
    print()
    
    # Test ZV shaper
    print("-" * 70)
    print("ZV Shaper (Zero Vibration)")
    print("-" * 70)
    A_zv, t_zv, K = ZV(omega_n, zeta)
    print(f"Amplitudes: {A_zv}")
    print(f"Times: {t_zv}")
    print(f"Duration: {t_zv[-1]:.4f} s")
    print()
    
    # Test ZVD shaper
    print("-" * 70)
    print("ZVD Shaper (Zero Vibration Derivative)")
    print("-" * 70)
    A_zvd, t_zvd, K = ZVD(omega_n, zeta)
    print(f"Amplitudes: {A_zvd}")
    print(f"Times: {t_zvd}")
    print(f"Duration: {t_zvd[-1]:.4f} s")
    print()
    
    # Test ZVDD shaper
    print("-" * 70)
    print("ZVDD Shaper (Zero Vibration Double Derivative)")
    print("-" * 70)
    A_zvdd, t_zvdd = ZVDD(omega_n, zeta)
    print(f"Amplitudes: {A_zvdd}")
    print(f"Times: {t_zvdd}")
    print(f"Duration: {t_zvdd[-1]:.4f} s")
    print()
    
    # Test EI shaper
    print("-" * 70)
    print("EI Shaper (Extra-Insensitive)")
    print("-" * 70)
    print("Note: This may take a few moments to optimize...")
    A_ei, t_ei = EI(omega_n, zeta, Vtol=0.05, tol_band=0.20)
    print(f"Amplitudes: {A_ei}")
    print(f"Times: {t_ei}")
    print(f"Duration: {t_ei[-1]:.4f} s")
    print()
    
    print("=" * 70)
    print("All shapers computed successfully!")
    print("=" * 70)
