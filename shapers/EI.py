import numpy as np
from scipy.optimize import minimize, NonlinearConstraint

def EI(omega_n, zeta, Vtol, tol_band=0.20):
    """
    Extra-Insensitive (EI) input shaper
    
    Designed to maintain residual vibration below Vtol across 
    a frequency uncertainty band of ±tol_band (default ±20%)
    
    Parameters:
    -----------
    omega_n : float
        Natural frequency (rad/s)
    zeta : float
        Damping ratio
    Vtol : float
        Vibration tolerance (e.g., 0.05 for 5%)
    tol_band : float
        Fractional frequency uncertainty (default 0.20 for ±20%)
    
    Returns:
    --------
    amplitudes : np.array
        Impulse amplitudes [A1, A2, A3]
    times : np.array
        Impulse times [0, t2, t3]
    """
    
    omega_d = omega_n * np.sqrt(1 - zeta**2)
    
    # Frequency points for constraints
    omega_low = omega_n * (1 - tol_band)  # Lower edge
    omega_nom = omega_n                    # Nominal
    omega_high = omega_n * (1 + tol_band)  # Upper edge
    
    def residual_vibration(omega, A1, A2, A3, t2, t3):
        """
        Calculate residual vibration amplitude at given frequency
        
        V(ω) = |A1*exp(-ζωt1)*exp(jωt1) + A2*exp(-ζωt2)*exp(jωt2) + A3*exp(-ζωt3)*exp(jωt3)|
        
        With t1 = 0, this simplifies to:
        V(ω) = |A1 + A2*exp(-ζωt2)*exp(jωt2) + A3*exp(-ζωt3)*exp(jωt3)|
        """
        # Complex exponentials with damping
        z1 = A1  # First impulse at t=0
        z2 = A2 * np.exp(-zeta * omega * t2) * np.exp(1j * omega * t2)
        z3 = A3 * np.exp(-zeta * omega * t3) * np.exp(1j * omega * t3)
        
        return np.abs(z1 + z2 + z3)
    
    def objective(x):
        """Minimize duration (t3)"""
        A1, A2, A3, t2, t3 = x
        return t3
    
    def constraint_unity_gain(x):
        """A1 + A2 + A3 = 1"""
        A1, A2, A3, t2, t3 = x
        return A1 + A2 + A3 - 1.0
    
    def constraint_zero_at_nominal(x):
        """V(ω_nominal) = 0"""
        A1, A2, A3, t2, t3 = x
        return residual_vibration(omega_nom, A1, A2, A3, t2, t3)
    
    def constraint_vtol_at_low(x):
        """V(ω_low) = Vtol"""
        A1, A2, A3, t2, t3 = x
        return residual_vibration(omega_low, A1, A2, A3, t2, t3) - Vtol
    
    def constraint_vtol_at_high(x):
        """V(ω_high) = Vtol"""
        A1, A2, A3, t2, t3 = x
        return residual_vibration(omega_high, A1, A2, A3, t2, t3) - Vtol
    # Initial guess: Use ZVD-like spacing and amplitudes
    t2_init = np.pi / omega_d
    t3_init = 2 * np.pi / omega_d
    A1_init, A2_init, A3_init = 0.25, 0.5, 0.25
    x0 = [A1_init, A2_init, A3_init, t2_init, t3_init]
    
    # Bounds: amplitudes [0, 1], times positive
    bounds = [(0, 1), (0, 1), (0, 1), (0.1, 10), (0.2, 20)]
    
    # Constraints
    constraints = [
        {'type': 'eq', 'fun': constraint_unity_gain},
        {'type': 'eq', 'fun': constraint_zero_at_nominal},
        {'type': 'eq', 'fun': constraint_vtol_at_low},
        {'type': 'eq', 'fun': constraint_vtol_at_high}
    ]
    
    iteration_count = [0]
    def callback(x):
        iteration_count[0] += 1
        if iteration_count [0] % 50 == 0:
            A1, A2, A3, t2, t3 = x
            V_nom = residual_vibration(omega_nom, A1, A2, A3, t2, t3)
            V_low = residual_vibration(omega_low, A1, A2, A3, t2, t3)
            V_high = residual_vibration(omega_high, A1, A2, A3, t2, t3)
            print(f"Iter {iteration_count[0]}: V_nom={V_nom:.4f}, V_low={V_low:.4f}, V_high={V_high:.4f}")

    # Solve optimization
    result = minimize(
        objective,
        x0,
        method='SLSQP',
        bounds=bounds,
        constraints=constraints,
        options={'maxiter': 1000, 'ftol': 1e-9},
        callback=callback
    )
    
    if not result.success:
        print(f"Warning: EI optimization did not converge. Message: {result.message}")
    
    A1, A2, A3, t2, t3 = result.x
    
    amplitudes = np.array([A1, A2, A3])
    times = np.array([0, t2, t3])
    
    return amplitudes, times

