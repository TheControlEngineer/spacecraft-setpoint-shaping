import numpy as np
def ZVDD(omega_n, zeta):
    """
    Zero Vibration Derivative Derivative (ZVDD) input shaper.

    Parameters:
    omega_n : float
        Natural frequency of the system (rad/s).
    zeta : float
        Damping ratio of the system.

    Returns:
    tuple
        A tuple containing two numpy arrays:
        - times: The time delays for each impulse.
        - amplitudes: The amplitudes for each impulse.
    """
    # Calculate the damped natural frequency
    omega_d = omega_n * np.sqrt(1 - zeta**2)

    # Calculate time delays
    t1 = np.pi / omega_d
    t2 = 2 * t1
    t3 = 3 * t1

    # Calculate amplitudes
    K = np.exp(-zeta * np.pi / np.sqrt(1 - zeta**2))
    A1 = 1 / (1 + 3*K+ 3*K**2 + K**3)
    A2 = 3*K / (1 + 3*K+ 3*K**2 + K**3)
    A3 = 3*K**2 / (1 + 3*K+ 3*K**2 + K**3)
    A4 = K**3 / (1 + 3*K+ 3*K**2 + K**3)

    t = np.array([0, t1, t2, t3])
    A = np.array([A1, A2, A3, A4])

    return A, t
    


