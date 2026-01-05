import numpy as np 
def ZVD(omega_n, zeta):
    """
    Zero Vibration Derivative (ZVD) input shaper.

    Parameters
    ----------
    omega_n : float
        Natural frequency (rad/s).
    zeta : float
        Damping ratio.

    Returns
    -------
    A : ndarray
        Amplitudes of the impulses.
    t : ndarray
        Times of the impulses.
    """
    omega_d = omega_n * np.sqrt(1 - zeta**2)
    K = np.exp(-zeta * np.pi / np.sqrt(1 - zeta**2)) # K values
    A = np.array([1 / (1 + 2 * K + K**2), 2 * K / (1 + 2 * K + K**2), K**2 / (1 + 2 * K + K**2)]) # Amplitudes
    t = np.array([0, np.pi / omega_d, 2 * np.pi / omega_d])# Times
    return A, t, K




