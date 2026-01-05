import numpy as np 
def ZV(omega_n, zeta):
    """
    Zero Vibration (ZV) input shaper.

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
    K = np.exp(-zeta * np.pi / np.sqrt(1 - zeta**2))#
    A = np.array([1 / (1 + K), K / (1 + K)]) # Amplitudes
    t = np.array([0, np.pi / omega_d])# Times
    return A, t, K 



