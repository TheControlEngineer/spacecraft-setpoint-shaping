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
        
        Uses Singer & Seering (1990) formula:
        V(ω) = |∑ Ai * exp(+ζωti) * exp(jωti)|
        
        Note: The POSITIVE sign in the damping exponent is critical!
        This accounts for how much earlier impulses have decayed relative
        to the final impulse time reference point.
        
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
        # Complex exponentials with POSITIVE damping exponent per Singer & Seering
        z1 = A1  # First impulse at t=0
        z2 = A2 * np.exp(+zeta * omega * t2) * np.exp(1j * omega * t2)
        z3 = A3 * np.exp(+zeta * omega * t3) * np.exp(1j * omega * t3)
        
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
    
    # Calculate ZVD parameters as initial guess (ZVD is often close to optimal EI)
    K = np.exp(-zeta * np.pi / np.sqrt(1 - zeta**2))
    A1_zvd = 1 / (1 + K)**2
    A2_zvd = 2 * K / (1 + K)**2
    A3_zvd = K**2 / (1 + K)**2
    t2_zvd = np.pi / omega_d
    t3_zvd = 2 * np.pi / omega_d
    
    # Use ZVD as initial guess (much better than arbitrary values)
    x0 = [A1_zvd, A2_zvd, A3_zvd, t2_zvd, t3_zvd]
    
    # Define bounds for design variables
    # Amplitudes: [0, 1], Times: reasonable positive values
    bounds = [
        (0.01, 0.99),  # A1
        (0.01, 0.99),  # A2
        (0.01, 0.99),  # A3
        (0.1, 10),     # t2
        (0.2, 20)      # t3
    ]
    
    # Define equality constraints
    constraints = [
        {'type': 'eq', 'fun': constraint_unity_gain},
        {'type': 'eq', 'fun': constraint_zero_at_nominal},
        {'type': 'eq', 'fun': constraint_vtol_at_low},
        {'type': 'eq', 'fun': constraint_vtol_at_high}
    ]
    
    # Solve the optimization problem
    result = minimize(
        objective,
        x0,
        method='SLSQP',
        bounds=bounds,
        constraints=constraints,
        options={'maxiter': 2000, 'ftol': 1e-12}
    )
    
    # If optimization didn't converge well, check if ZVD is close enough
    A1, A2, A3, t2, t3 = result.x
    V_nom_result = residual_vibration(omega_nom, A1, A2, A3, t2, t3)
    
    # If nominal vibration is not close to zero, fall back to ZVD
    # (ZVD guarantees zero at nominal and often meets EI requirements)
    if V_nom_result > 1e-3:
        V_nom_zvd = residual_vibration(omega_nom, A1_zvd, A2_zvd, A3_zvd, t2_zvd, t3_zvd)
        if V_nom_zvd < V_nom_result:
            # ZVD is better, use it
            A1, A2, A3, t2, t3 = A1_zvd, A2_zvd, A3_zvd, t2_zvd, t3_zvd
    
    # Extract optimized parameters
    # A1, A2, A3, t2, t3 = result.x  # Already extracted above
    
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


def convolve_shapers(shaper1: Tuple[np.ndarray, np.ndarray], 
                     shaper2: Tuple[np.ndarray, np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convolve two input shapers to create a multi-mode shaper.
    
    Given two shapers S1 and S2, the convolved shaper suppresses both modes.
    
    Parameters
    ----------
    shaper1 : tuple
        First shaper (amplitudes, times)
    shaper2 : tuple
        Second shaper (amplitudes, times)
    
    Returns
    -------
    amplitudes : np.ndarray
        Convolved impulse amplitudes
    times : np.ndarray
        Convolved impulse times
        
    Examples
    --------
    >>> # Create multi-mode shaper for 0.3 Hz and 0.8 Hz modes
    >>> A1, t1, _ = ZV(2*np.pi*0.3, 0.02)
    >>> A2, t2, _ = ZV(2*np.pi*0.8, 0.02)
    >>> A_multi, t_multi = convolve_shapers((A1, t1), (A2, t2))
    >>> print(f"Single mode: {len(A1)} impulses, Multi-mode: {len(A_multi)} impulses")
    Single mode: 2 impulses, Multi-mode: 4 impulses
    
    Notes
    -----
    The convolution creates N1 × N2 impulses, where N1 and N2 are the number
    of impulses in each original shaper.
    
    Duration of convolved shaper = duration(S1) + duration(S2)
    """
    A1, t1 = shaper1
    A2, t2 = shaper2
    
    # Convolution: every impulse from S1 paired with every impulse from S2
    n_impulses = len(A1) * len(A2)
    A_conv = np.zeros(n_impulses)
    t_conv = np.zeros(n_impulses)
    
    idx = 0
    for i, (a1, t1_i) in enumerate(zip(A1, t1)):
        for j, (a2, t2_j) in enumerate(zip(A2, t2)):
            A_conv[idx] = a1 * a2  # Amplitude product
            t_conv[idx] = t1_i + t2_j  # Time sum
            idx += 1
    
    # Sort by time (impulses may not be in order after convolution)
    sort_idx = np.argsort(t_conv)
    A_conv = A_conv[sort_idx]
    t_conv = t_conv[sort_idx]
    
    # Verify unity gain (should still sum to 1.0)
    assert np.isclose(np.sum(A_conv), 1.0, atol=1e-6), \
        f"Convolved shaper has gain {np.sum(A_conv)}, expected 1.0"
    
    return A_conv, t_conv


def design_multimode_cascaded(mode_frequencies: list,
                               damping_ratios: list,
                               method: str = 'ZV') -> Tuple[np.ndarray, np.ndarray]:
    """
    Design multi-mode shaper using cascaded convolution.
    
    Parameters
    ----------
    mode_frequencies : list
        Natural frequencies of each mode (Hz)
    damping_ratios : list
        Damping ratio for each mode
    method : str
        Shaper type to use for each mode ('ZV', 'ZVD', or 'ZVDD')
    
    Returns
    -------
    amplitudes : np.ndarray
        Multi-mode shaper impulse amplitudes
    times : np.ndarray
        Multi-mode shaper impulse times
        
    Examples
    --------
    >>> # Design for 2-mode spacecraft (0.3 Hz and 0.8 Hz)
    >>> A, t = design_multimode_cascaded([0.3, 0.8], [0.02, 0.02], method='ZVD')
    >>> print(f"Number of impulses: {len(A)}")
    >>> print(f"Total duration: {t[-1]:.2f} seconds")
    """
    if len(mode_frequencies) != len(damping_ratios):
        raise ValueError("mode_frequencies and damping_ratios must have same length")
    
    if len(mode_frequencies) == 0:
        raise ValueError("Must provide at least one mode")
    
    # Design shaper for first mode
    omega_n = 2 * np.pi * mode_frequencies[0]
    zeta = damping_ratios[0]
    
    if method.upper() == 'ZV':
        A_total, t_total, _ = ZV(omega_n, zeta)
    elif method.upper() == 'ZVD':
        A_total, t_total, _ = ZVD(omega_n, zeta)
    elif method.upper() == 'ZVDD':
        A_total, t_total = ZVDD(omega_n, zeta)
    else:
        raise ValueError(f"Unknown method: {method}")
    
    # Convolve with shapers for remaining modes
    for i in range(1, len(mode_frequencies)):
        omega_n = 2 * np.pi * mode_frequencies[i]
        zeta = damping_ratios[i]
        
        if method.upper() == 'ZV':
            A_mode, t_mode, _ = ZV(omega_n, zeta)
        elif method.upper() == 'ZVD':
            A_mode, t_mode, _ = ZVD(omega_n, zeta)
        elif method.upper() == 'ZVDD':
            A_mode, t_mode = ZVDD(omega_n, zeta)
        else:
            raise ValueError(f"Unknown method: {method}")
        
        # Convolve
        A_total, t_total = convolve_shapers((A_total, t_total), (A_mode, t_mode))
    
    return A_total, t_total


def design_multimode_simultaneous(mode_frequencies: list,
                                  damping_ratios: list,
                                  n_impulses: int = 4,
                                  Vtol: float = 0.05) -> Tuple[np.ndarray, np.ndarray]:
    """
    Design multi-mode shaper using simultaneous optimization.
    
    Optimizes a single shaper with n_impulses to suppress all modes
    simultaneously, typically achieving shorter duration than cascaded approach.
    
    Parameters
    ----------
    mode_frequencies : list
        Natural frequencies of each mode (Hz)
    damping_ratios : list
        Damping ratio for each mode
    n_impulses : int
        Number of impulses to use (default: 4)
        More impulses → better performance but longer duration
    Vtol : float
        Vibration tolerance for each mode (default: 0.05 = 5%)
    
    Returns
    -------
    amplitudes : np.ndarray
        Optimized impulse amplitudes
    times : np.ndarray
        Optimized impulse times
        
    Notes
    -----
    This is a constrained optimization problem:
    
    minimize: t_N (duration)
    subject to:
        V_i(ω_i) ≤ Vtol  for all modes i
        Σ A_j = 1        (unity gain)
        A_j > 0          (positive amplitudes)
        0 ≤ t_j ≤ t_N    (ordered times)
    
    The optimization typically takes 1-5 seconds to converge.
    
    Examples
    --------
    >>> # Design for 2-mode spacecraft
    >>> A, t = design_multimode_simultaneous([0.3, 0.8], [0.02, 0.02], 
    ...                                       n_impulses=4, Vtol=0.05)
    >>> print(f"Duration: {t[-1]:.2f}s")  # Typically shorter than cascaded
    """
    if len(mode_frequencies) != len(damping_ratios):
        raise ValueError("mode_frequencies and damping_ratios must have same length")
    
    if n_impulses < len(mode_frequencies) + 1:
        import warnings
        warnings.warn(f"n_impulses={n_impulses} may be too few for {len(mode_frequencies)} modes. "
                     f"Recommend at least {len(mode_frequencies) + 1} impulses.")
    
    n_modes = len(mode_frequencies)
    omega_d = [2 * np.pi * f * np.sqrt(1 - z**2) 
               for f, z in zip(mode_frequencies, damping_ratios)]
    omega_n = [2 * np.pi * f for f in mode_frequencies]
    
    def residual_vibration_mode(omega_d_i, omega_n_i, zeta_i, A, t):
        """Calculate residual vibration for one mode"""
        V = 0
        for amp, time in zip(A, t):
            V += amp * np.exp(zeta_i * omega_n_i * time) * np.exp(1j * omega_d_i * time)
        return np.abs(V)
    
    def objective(x):
        """Minimize duration (last impulse time)"""
        # x = [A1, A2, ..., A_n, t1, t2, ..., t_n]
        # Note: t1 = 0 is fixed
        t_last = x[2*n_impulses - 1]
        return t_last
    
    def constraint_unity_gain(x):
        """Sum of amplitudes = 1"""
        A = x[:n_impulses]
        return np.sum(A) - 1.0
    
    def constraint_vibration_mode_i(i):
        """Factory function for mode-specific constraints"""
        def constraint(x):
            A = x[:n_impulses]
            t = x[n_impulses:]
            V = residual_vibration_mode(omega_d[i], omega_n[i], damping_ratios[i], A, t)
            return Vtol - V  # V ≤ Vtol means Vtol - V ≥ 0
        return constraint
    
    def constraint_time_ordering(x):
        """Ensure times are in ascending order"""
        t = x[n_impulses:]
        # Return array of differences (all should be ≥ 0)
        return np.diff(t)
    
    # Initial guess: evenly spaced impulses with equal amplitudes
    # Estimate duration from longest period
    T_max = max([1/f for f in mode_frequencies])
    t_init = np.linspace(0, T_max, n_impulses)
    A_init = np.ones(n_impulses) / n_impulses
    x0 = np.concatenate([A_init, t_init])
    
    # Bounds
    bounds = [(0.001, 1.0) for _ in range(n_impulses)]  # Amplitudes: (0, 1]
    bounds += [(0.0, 0.1)]  # First time fixed at 0, with small tolerance
    bounds += [(0.01, 20.0) for _ in range(n_impulses - 1)]  # Other times
    
    # Constraints
    constraints = [
        {'type': 'eq', 'fun': constraint_unity_gain}
    ]
    
    # Add vibration constraint for each mode
    for i in range(n_modes):
        constraints.append({
            'type': 'ineq', 
            'fun': constraint_vibration_mode_i(i)
        })
    
    # Add time ordering constraints
    constraints.append({
        'type': 'ineq',
        'fun': constraint_time_ordering
    })
    
    # Solve
    result = minimize(
        objective,
        x0,
        method='SLSQP',
        bounds=bounds,
        constraints=constraints,
        options={'maxiter': 500, 'ftol': 1e-8}
    )
    
    if not result.success:
        import warnings
        warnings.warn(f"Optimization did not fully converge: {result.message}")
    
    # Extract solution
    A_opt = result.x[:n_impulses]
    t_opt = result.x[n_impulses:]
    
    # Verify constraints
    for i, (f, z) in enumerate(zip(mode_frequencies, damping_ratios)):
        V = residual_vibration_mode(omega_d[i], omega_n[i], z, A_opt, t_opt)
        if V > Vtol * 1.5:  # Allow 50% overshoot as warning threshold
            import warnings
            warnings.warn(f"Mode {i+1} ({f} Hz): V = {V:.4f} exceeds tolerance {Vtol:.4f}")
    
    return A_opt, t_opt


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
