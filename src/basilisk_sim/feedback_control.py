"""
Feedback Control Module.

Provides MRP-based feedback attitude control for spacecraft pointing.
This module encapsulates the feedback control logic used in vizard_demo.py
for fine pointing after feedforward slews.

Control Law (MRP Feedback):
    tau = -K * sigma_error - P * omega_error + Ki * integral(sigma_error dt)

Where:
    - K: proportional gain on attitude error (MRP).
    - P: derivative gain on rate error.
    - Ki: integral gain for steady-state error correction.
    - sigma_error: MRP attitude error vector.
    - omega_error: angular velocity error vector.

The gains are typically tuned based on the spacecraft inertia and the
desired closed-loop bandwidth and damping.

"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy import signal

from .spacecraft_properties import compute_modal_gains, FLEX_MODE_MASS


def _mrp_shadow(sigma: np.ndarray) -> np.ndarray:
    """Return the shadow MRP if |sigma| > 1 to ensure shortest rotation."""
    sigma = np.array(sigma, dtype=float).flatten()
    sigma_norm_sq = float(np.dot(sigma, sigma))
    if sigma_norm_sq > 1.0 + 1e-12:
        return -sigma / sigma_norm_sq
    return sigma


def _mrp_subtract(sigma_body: np.ndarray, sigma_ref: np.ndarray) -> np.ndarray:
    """
    Compute MRP attitude error sigma_BR = sigma_BN (-) sigma_RN (body relative to reference).

    This follows the standard MRP subtraction formula and applies the shadow set
    to return the shorter rotation.
    """
    sigma_body = np.array(sigma_body, dtype=float).flatten()
    sigma_ref = np.array(sigma_ref, dtype=float).flatten()

    sigma_body_sq = float(np.dot(sigma_body, sigma_body))
    sigma_ref_sq = float(np.dot(sigma_ref, sigma_ref))
    dot_product = float(np.dot(sigma_ref, sigma_body))

    denom = 1.0 + sigma_body_sq * sigma_ref_sq + 2.0 * dot_product
    if abs(denom) < 1e-12:
        return np.zeros(3)

    sigma_error = (
        (1.0 - sigma_ref_sq) * sigma_body
        - (1.0 - sigma_body_sq) * sigma_ref
        + 2.0 * np.cross(sigma_body, sigma_ref)
    ) / denom

    return _mrp_shadow(sigma_error)


def _tf_add(num1: np.ndarray, den1: np.ndarray, num2: np.ndarray, den2: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Add two transfer functions: num1/den1 + num2/den2."""
    num1 = np.atleast_1d(np.squeeze(num1))
    den1 = np.atleast_1d(np.squeeze(den1))
    num2 = np.atleast_1d(np.squeeze(num2))
    den2 = np.atleast_1d(np.squeeze(den2))

    term1 = np.convolve(num1, den2)
    term2 = np.convolve(num2, den1)

    if len(term1) > len(term2):
        term2 = np.pad(term2, (len(term1) - len(term2), 0), mode='constant')
    elif len(term2) > len(term1):
        term1 = np.pad(term1, (len(term2) - len(term1), 0), mode='constant')

    num = term1 + term2
    den = np.convolve(den1, den2)

    return num, den


def _build_flexible_plant_tf(inertia: np.ndarray,
                             axis: int,
                             modal_freqs_hz: List[float],
                             modal_damping: List[float],
                             modal_gains: List[float]) -> signal.TransferFunction:
    """
    Build a coupled rigid-flex plant (torque -> sigma) using a modal mass model.

    The state-space model includes the rigid angle and modal coordinates, avoiding
    mixing rigid attitude output with modal displacement outputs.
    """
    I_axis = float(inertia[axis, axis])
    sigma_scale = 4.0

    if not modal_freqs_hz or not modal_gains:
        return signal.TransferFunction([1.0], [sigma_scale * I_axis, 0.0, 0.0])

    n_modes = min(len(modal_freqs_hz), len(modal_gains))
    zetas = list(modal_damping[:n_modes]) if modal_damping else [0.01] * n_modes
    if len(zetas) < n_modes:
        zetas.extend([zetas[-1]] * (n_modes - len(zetas)))

    # Infer lever arms from modal gains: gain = r / I_axis to r = gain * I_axis
    lever_arms = [float(g) * I_axis for g in modal_gains[:n_modes]]
    masses = [float(FLEX_MODE_MASS)] * n_modes

    # Assemble mass, damping, stiffness matrices for coupled model
    sum_mr2 = sum(m * r * r for m, r in zip(masses, lever_arms))
    # Avoid double counting if inertia already includes modal masses.
    base_inertia = I_axis - sum_mr2 if I_axis > sum_mr2 else I_axis

    m_mat = np.zeros((n_modes + 1, n_modes + 1))
    m_mat[0, 0] = base_inertia + sum_mr2
    for idx, (m_i, r_i) in enumerate(zip(masses, lever_arms), start=1):
        m_mat[0, idx] = m_i * r_i
        m_mat[idx, 0] = m_i * r_i
        m_mat[idx, idx] = m_i

    d_mat = np.zeros_like(m_mat)
    k_mat = np.zeros_like(m_mat)
    for idx, (freq_hz, zeta, m_i) in enumerate(zip(modal_freqs_hz[:n_modes], zetas, masses), start=1):
        omega = 2.0 * np.pi * float(freq_hz)
        d_mat[idx, idx] = 2.0 * float(zeta) * omega * m_i
        k_mat[idx, idx] = omega**2 * m_i

    m_inv = np.linalg.inv(m_mat)
    n_state = n_modes + 1
    a = np.zeros((2 * n_state, 2 * n_state))
    a[:n_state, n_state:] = np.eye(n_state)
    a[n_state:, :n_state] = -m_inv @ k_mat
    a[n_state:, n_state:] = -m_inv @ d_mat

    b = np.zeros((2 * n_state, 1))
    b[n_state:, 0] = (m_inv @ np.array([1.0] + [0.0] * n_modes)).ravel()

    # Output is sigma = theta / 4
    c_sigma = np.zeros((1, 2 * n_state))
    c_sigma[0, 0] = 1.0 / sigma_scale
    d = np.zeros((1, 1))

    ss = signal.StateSpace(a, b, c_sigma, d)
    num, den = signal.ss2tf(ss.A, ss.B, ss.C, ss.D)
    return signal.TransferFunction(np.squeeze(num), np.squeeze(den))


class MRPFeedbackController:
    """
    MRP-based feedback attitude controller.
    
    Implements a PID-like controller using Modified Rodrigues Parameters (MRPs)
    for attitude error representation. This is the same control law used in
    Basilisk's mrpFeedback module.
    
    Attributes:
        K: Proportional gain (attitude stiffness)
        P: Derivative gain (rate damping)
        Ki: Integral gain (steady-state correction)
        inertia: Spacecraft inertia matrix (3x3)
    """
    
    def __init__(self, 
                 inertia: np.ndarray,
                 K: float = 30.0,
                 P: float = 60.0,
                 Ki: float = -1.0):
        """
        Initialize the feedback controller.
        
        Args:
            inertia: 3x3 spacecraft inertia matrix [kg*m^2]
            K: Proportional gain (default: 30.0)
            P: Derivative gain (default: 60.0)  
            Ki: Integral gain (default: -1.0, negative disables integral)
        """
        self.inertia = np.array(inertia)
        self.K = K
        self.P = P
        self.Ki = Ki
        
        # Integral error accumulator
        self.sigma_integral = np.zeros(3)
        self.last_time = None
        
        # Target attitude
        self.sigma_target = np.zeros(3)
        self.omega_target = np.zeros(3)
        
        print(f"MRP Feedback Controller initialized:")
        print(f"  Gains: K={K}, P={P}, Ki={Ki}")
        print(f"  Inertia diagonal: [{inertia[0,0]:.0f}, {inertia[1,1]:.0f}, {inertia[2,2]:.0f}] kg*m^2")
    
    def set_target(self, sigma_target: np.ndarray, omega_target: np.ndarray = None):
        """
        Set the target attitude and angular velocity.
        
        Args:
            sigma_target: Target MRP attitude (3,)
            omega_target: Target angular velocity [rad/s] (3,), default is zero
        """
        self.sigma_target = np.array(sigma_target).flatten()
        self.omega_target = np.array(omega_target).flatten() if omega_target is not None else np.zeros(3)
        
        # Reset integral on target change
        self.sigma_integral = np.zeros(3)
        self.last_time = None
    
    def reset_integral(self):
        """Reset the integral error accumulator."""
        self.sigma_integral = np.zeros(3)
        self.last_time = None
    
    def compute_mrp_error(self, sigma_current: np.ndarray) -> np.ndarray:
        """
        Compute MRP attitude error with shadow set handling.

        Basilisk convention: sigma_BR = sigma_BN (-) sigma_RN (body relative to reference).
        For small angles this reduces to sigma_current - sigma_target.

        Args:
            sigma_current: Current MRP attitude (3,)

        Returns:
            sigma_error: MRP attitude error (3,)
        """
        sigma_current = np.array(sigma_current).flatten()
        return _mrp_subtract(sigma_current, self.sigma_target)

    def compute_torque(self, 
                       sigma_current: np.ndarray,
                       omega_current: np.ndarray,
                       current_time: float = None) -> np.ndarray:
        """
        Compute feedback control torque.
        
        Implements: tau = -K * sigma_error - P * omega_error + Ki * integral(sigma_error dt)
        
        Args:
            sigma_current: Current MRP attitude (3,)
            omega_current: Current angular velocity [rad/s] (3,)
            current_time: Current time for integral calculation [s]
            
        Returns:
            torque: Control torque command [N*m] (3,)
        """
        sigma_current = np.array(sigma_current).flatten()
        omega_current = np.array(omega_current).flatten()
        
        # Compute errors
        sigma_error = self.compute_mrp_error(sigma_current)
        omega_error = omega_current - self.omega_target
        
        # Update integral term
        if self.Ki > 0 and current_time is not None:
            if self.last_time is not None:
                dt = current_time - self.last_time
                if dt > 0:
                    self.sigma_integral += sigma_error * dt
            self.last_time = current_time
        
        # Compute control torque
        # Control torque combines proportional, derivative, and optional integral action
        torque = -self.K * sigma_error - self.P * omega_error
        
        if self.Ki > 0:
            torque += self.Ki * self.sigma_integral
        
        return torque
    
    def get_closed_loop_params(self) -> Dict[str, float]:
        """
        Compute approximate closed-loop parameters.
        
        For a linearized attitude system about small angles:
            (4I) * sigma_ddot + P * sigma_dot + K * sigma = 0

        This gives natural frequency omega_n = sqrt(K/(4I)) and damping
        zeta = P / (2 * sqrt(K * 4I)).
        
        Returns:
            Dictionary with natural frequency, damping ratio, and bandwidth
        """
        # Use average principal inertia
        I_avg = np.mean(np.diag(self.inertia))
        
        sigma_scale = 4.0

        # Natural frequency [rad/s]
        omega_n = np.sqrt(self.K / (sigma_scale * I_avg))

        # Damping ratio (sigma_dot ≈ 0.25 * omega)
        zeta = self.P / np.sqrt(self.K * I_avg)
        
        # Closed loop bandwidth (approximately)
        omega_bw = omega_n * np.sqrt(1 - 2*zeta**2 + np.sqrt(4*zeta**4 - 4*zeta**2 + 2))
        
        # Settling time (2% criterion)
        if zeta < 1:
            t_settle = 4 / (zeta * omega_n)
        elif np.isclose(zeta, 1.0):
            t_settle = 4 / omega_n
        else:
            slower_pole_mag = omega_n * (zeta - np.sqrt(zeta**2 - 1.0))
            t_settle = 4 / slower_pole_mag if slower_pole_mag > 0 else np.inf
        
        return {
            'omega_n': omega_n,
            'omega_n_hz': omega_n / (2 * np.pi),
            'zeta': zeta,
            'omega_bw': omega_bw,
            'omega_bw_hz': omega_bw / (2 * np.pi),
            't_settle': t_settle,
            'I_avg': I_avg
        }
    
    def get_transfer_function(self, axis: int = 2) -> Tuple[signal.TransferFunction, signal.TransferFunction]:
        """
        Get linearized transfer functions for the feedback loop.
        
        For single-axis rotation about principal axis:
            Plant: G(s) = 1 / (I * s^2)
            Controller: C(s) = K + P*s (+ Ki/s if integral enabled)
            
        Returns open-loop and closed-loop transfer functions.
        
        Args:
            axis: Principal axis (0=X, 1=Y, 2=Z)
            
        Returns:
            Tuple of (open_loop_tf, closed_loop_tf)
        """
        I = self.inertia[axis, axis]
        
        # Plant (torque to sigma): 1/(4*I*s^2)
        sigma_scale = 4.0
        plant_num = [1.0]
        plant_den = [sigma_scale * I, 0, 0]
        
        # Controller in sigma domain: K + 4*P*s (PD on omega)
        if self.Ki > 0:
            # PID: (Ki + K*s + 4*P*s^2) / s
            ctrl_num = [4.0 * self.P, self.K, self.Ki]
            ctrl_den = [1, 0]
        else:
            # PD: 4*P*s + K
            ctrl_num = [4.0 * self.P, self.K]
            ctrl_den = [1]
        
        # Open loop: G*C
        G = signal.TransferFunction(plant_num, plant_den)
        C = signal.TransferFunction(ctrl_num, ctrl_den)
        
        # For transfer function multiplication, convert to zpk
        G_zpk = signal.tf2zpk(plant_num, plant_den)
        C_zpk = signal.tf2zpk(ctrl_num, ctrl_den)
        
        # Open loop = G * C
        ol_zeros = np.concatenate([G_zpk[0], C_zpk[0]])
        ol_poles = np.concatenate([G_zpk[1], C_zpk[1]])
        ol_gain = G_zpk[2] * C_zpk[2]
        
        open_loop = signal.ZerosPolesGain(ol_zeros, ol_poles, ol_gain)
        open_loop_tf = open_loop.to_tf()
        
        # Closed loop: GC / (1 + GC)
        # For simple PD on double integrator: 4*I*s^2 + 4*P*s + K
        if self.Ki > 0:
            cl_num = [4.0 * self.P, self.K, self.Ki]
            cl_den = [sigma_scale * I, 4.0 * self.P, self.K, self.Ki]
        else:
            cl_num = [4.0 * self.P, self.K]
            cl_den = [sigma_scale * I, 4.0 * self.P, self.K]
        
        closed_loop_tf = signal.TransferFunction(cl_num, cl_den)
        
        return open_loop_tf, closed_loop_tf
    
    def get_open_loop_tf(self,
                         axis: int = 2,
                         modal_freqs_hz: List[float] = None,
                         modal_damping: List[float] = None,
                         modal_gains: List[float] = None,
                         include_flexibility: bool = True) -> signal.TransferFunction:
        """
        Get open-loop transfer function L(s) = G(s) * C(s).
        
        Args:
            axis: Principal axis (0=X, 1=Y, 2=Z)
            modal_freqs_hz: Modal frequencies [Hz] for flexible plant
            modal_damping: Modal damping ratios
            modal_gains: Modal coupling gains
            include_flexibility: If True, include flexible modes in plant
            
        Returns:
            Open-loop transfer function L(s)
        """
        # Build plant
        if include_flexibility and modal_freqs_hz:
            plant = _build_flexible_plant_tf(
                self.inertia,
                axis,
                modal_freqs_hz,
                modal_damping,
                modal_gains
            )
        else:
            I = self.inertia[axis, axis]
            sigma_scale = 4.0
            plant = signal.TransferFunction([1.0], [sigma_scale * I, 0, 0])
        
        # Controller in sigma domain: K + 4*P*s
        if self.Ki > 0:
            ctrl_num = [4.0 * self.P, self.K, self.Ki]
            ctrl_den = [1, 0]
        else:
            ctrl_num = [4.0 * self.P, self.K]
            ctrl_den = [1]
        
        controller = signal.TransferFunction(ctrl_num, ctrl_den)
        
        # Open loop = Plant * Controller
        plant_num = np.atleast_1d(np.squeeze(plant.num))
        plant_den = np.atleast_1d(np.squeeze(plant.den))
        ctrl_num = np.atleast_1d(np.squeeze(controller.num))
        ctrl_den = np.atleast_1d(np.squeeze(controller.den))
        
        ol_num = np.convolve(plant_num, ctrl_num)
        ol_den = np.convolve(plant_den, ctrl_den)
        
        return signal.TransferFunction(ol_num, ol_den)


class FilteredDerivativeController:
    """
    PD controller with low-pass filtered derivative term.
    
    Standard PD control has infinite gain at high frequencies due to the
    derivative term. This excites high-frequency structural modes.
    
    Filtered PD uses:
        C(s) = K + P*s / (tau*s + 1)
    
    The filter time constant tau sets the rolloff frequency:
        f_rolloff = 1 / (2*pi*tau)
    
    Choose f_rolloff below the first structural mode for best vibration
    suppression, but above the desired bandwidth for good tracking.
    """
    
    def __init__(self,
                 inertia: np.ndarray,
                 K: float = 30.0,
                 P: float = 60.0,
                 filter_freq_hz: float = 0.2):
        """
        Initialize filtered derivative controller.
        
        Args:
            inertia: 3x3 spacecraft inertia matrix [kg*m^2]
            K: Proportional gain
            P: Derivative gain (before filtering)
            filter_freq_hz: Derivative filter cutoff frequency [Hz]
                           Should be above closed-loop bandwidth but below
                           first structural mode
        """
        self.inertia = np.array(inertia)
        self.K = K
        self.P = P
        self.filter_freq_hz = filter_freq_hz
        
        # Filter time constant
        self.tau = 1.0 / (2 * np.pi * filter_freq_hz)
        
        # Target attitude
        self.sigma_target = np.zeros(3)
        self.omega_target = np.zeros(3)
        
        # Filter state (3 axis)
        self.filtered_rate = np.zeros(3)
        self.last_time = None
        
        print(f"Filtered Derivative Controller initialized:")
        print(f"  Gains: K={K}, P={P}")
        print(f"  Filter cutoff: {filter_freq_hz:.3f} Hz (tau={self.tau:.3f} s)")
        print(f"  Inertia diagonal: [{inertia[0,0]:.0f}, {inertia[1,1]:.0f}, {inertia[2,2]:.0f}] kg*m^2")
    
    def set_target(self, sigma_target: np.ndarray, omega_target: np.ndarray = None):
        """Set target attitude and angular velocity."""
        self.sigma_target = np.array(sigma_target).flatten()
        self.omega_target = np.array(omega_target).flatten() if omega_target is not None else np.zeros(3)
    
    def reset(self):
        """Reset filter states."""
        self.filtered_rate = np.zeros(3)
        self.last_time = None
    
    def compute_mrp_error(self, sigma_current: np.ndarray) -> np.ndarray:
        """
        Compute MRP error using Basilisk convention: sigma_BR = sigma_BN (-) sigma_RN.

        This computes the rotation FROM reference TO body (current relative to target).
        When body is at origin and target is positive, sigma_BR is negative.
        Combined with tau = -K * sigma_BR, this gives positive torque toward target.
        """
        sigma_current = np.array(sigma_current).flatten()
        return _mrp_subtract(sigma_current, self.sigma_target)
    
    def compute_torque(self,
                       sigma_current: np.ndarray,
                       omega_current: np.ndarray,
                       current_time: float) -> np.ndarray:
        """
        Compute control torque with filtered derivative.
        
        Implements: tau = -K * sigma_error - P * (filtered omega_error)
        
        Where the filtered rate is computed by:
            tau * (d/dt)omega_filtered + omega_filtered = omega
        """
        sigma_current = np.array(sigma_current).flatten()
        omega_current = np.array(omega_current).flatten()
        
        # Compute attitude error
        sigma_error = self.compute_mrp_error(sigma_current)
        omega_error = omega_current - self.omega_target
        
        # Update filtered rate (first order low pass filter)
        if self.last_time is not None:
            dt = current_time - self.last_time
            if dt > 0:
                # Low pass filter: tau*omega_f' + omega_f = omega
                # Discrete form updates omega_f with the difference between omega and omega_f
                alpha = dt / (self.tau + dt)  # Bilinear approximation
                self.filtered_rate = (1 - alpha) * self.filtered_rate + alpha * omega_error
        else:
            self.filtered_rate = omega_error
        
        self.last_time = current_time
        
        # Compute torque with filtered rate
        torque = -self.K * sigma_error - self.P * self.filtered_rate
        
        return torque
    
    def get_transfer_function(self, axis: int = 2) -> signal.TransferFunction:
        """
        Get the filtered controller transfer function.
        
        Returns:
            C(s) = K + 4*P*s / (tau*s + 1)
                 = ((K*tau + 4P)*s + K) / (tau*s + 1)
        """
        # C(s) = K + 4*P*s/(tau*s + 1)
        # Combined: ((K*tau + 4P)*s + K) / (tau*s + 1)
        num = [self.K * self.tau + 4.0 * self.P, self.K]
        den = [self.tau, 1.0]
        
        return signal.TransferFunction(num, den)
    
    def get_closed_loop_params(self) -> Dict[str, float]:
        """Compute approximate closed-loop parameters."""
        I_avg = np.mean(np.diag(self.inertia))
        sigma_scale = 4.0
        omega_n = np.sqrt(self.K / (sigma_scale * I_avg))
        zeta = self.P / np.sqrt(self.K * I_avg)
        
        # Modified bandwidth estimate due to filtering
        omega_bw = omega_n * np.sqrt(1 - 2*zeta**2 + np.sqrt(4*zeta**4 - 4*zeta**2 + 2))
        
        return {
            'omega_n': omega_n,
            'omega_n_hz': omega_n / (2 * np.pi),
            'zeta': zeta,
            'omega_bw': omega_bw,
            'omega_bw_hz': omega_bw / (2 * np.pi),
            'filter_freq_hz': self.filter_freq_hz,
            'I_avg': I_avg
        }
    
    def get_open_loop_tf(self,
                         axis: int = 2,
                         modal_freqs_hz: List[float] = None,
                         modal_damping: List[float] = None,
                         modal_gains: List[float] = None,
                         include_flexibility: bool = True) -> signal.TransferFunction:
        """
        Get open-loop transfer function L(s) = G(s) * C(s).

        Args:
            axis: Principal axis (0=X, 1=Y, 2=Z)
            modal_freqs_hz: Modal frequencies [Hz] for flexible plant
            modal_damping: Modal damping ratios
            modal_gains: Modal coupling gains
            include_flexibility: If True, include flexible modes in plant

        Returns:
            Open-loop transfer function L(s)
        """
        # Build plant
        if include_flexibility and modal_freqs_hz:
            plant = _build_flexible_plant_tf(
                self.inertia,
                axis,
                modal_freqs_hz,
                modal_damping,
                modal_gains
            )
        else:
            I = self.inertia[axis, axis]
            sigma_scale = 4.0
            plant = signal.TransferFunction([1.0], [sigma_scale * I, 0, 0])

        # Filtered controller: C(s) = K + P*s/(tau*s + 1)
        controller = self.get_transfer_function(axis)

        # Open loop = Plant * Controller
        plant_num = np.atleast_1d(np.squeeze(plant.num))
        plant_den = np.atleast_1d(np.squeeze(plant.den))
        ctrl_num = np.atleast_1d(np.squeeze(controller.num))
        ctrl_den = np.atleast_1d(np.squeeze(controller.den))

        ol_num = np.convolve(plant_num, ctrl_num)
        ol_den = np.convolve(plant_den, ctrl_den)

        return signal.TransferFunction(ol_num, ol_den)


class NotchFilterController:
    """
    PD controller with notch filters at modal frequencies.

    Notch filters provide sharp attenuation at specific frequencies, preventing
    the controller from exciting structural modes. This is an alternative to
    low-pass filtering the derivative term.

    Transfer function for each notch:
        H_notch(s) = (s^2 + 2*zeta_z*omega_n*s + omega_n^2) / (s^2 + 2*zeta_p*omega_n*s + omega_n^2)

    Where:
        - zeta_z: Zero damping (typically small, ~0.01-0.05 for deep notch)
        - zeta_p: Pole damping (typically larger, ~0.1-0.5 for wider notch)
        - omega_n: Notch center frequency

    Robustness consideration: Notch filters are sensitive to frequency uncertainty.
    If the actual modal frequency differs from the design frequency, the notch
    may miss the resonance entirely. Use wider notches (larger zeta_p) for
    better robustness at the cost of phase margin.
    """

    def __init__(self,
                 inertia: np.ndarray,
                 K: float = 30.0,
                 P: float = 60.0,
                 notch_freqs_hz: List[float] = None,
                 notch_depth_db: float = 20.0,
                 notch_width: float = 0.3):
        """
        Initialize notch filter controller.

        Args:
            inertia: 3x3 spacecraft inertia matrix [kg*m^2]
            K: Proportional gain
            P: Derivative gain
            notch_freqs_hz: Frequencies to notch out [Hz]
            notch_depth_db: Notch depth in dB (default: 20 dB = 10x attenuation)
            notch_width: Relative width of notch (0.1=narrow, 0.5=wide)
        """
        self.inertia = np.array(inertia)
        self.K = K
        self.P = P
        self.notch_freqs_hz = list(notch_freqs_hz) if notch_freqs_hz else []
        self.notch_depth_db = notch_depth_db
        self.notch_width = notch_width

        # Target attitude
        self.sigma_target = np.zeros(3)
        self.omega_target = np.zeros(3)

        # Notch filter states (2nd order state per notch per axis)
        self.notch_states = []
        for _ in self.notch_freqs_hz:
            self.notch_states.append(np.zeros((3, 2)))  # 3 axes, 2 states each

        self.last_time = None

        # Compute notch filter coefficients
        self._compute_notch_coefficients()

        print(f"Notch Filter Controller initialized:")
        print(f"  Gains: K={K}, P={P}")
        print(f"  Notch frequencies: {self.notch_freqs_hz} Hz")
        print(f"  Notch depth: {notch_depth_db} dB, width: {notch_width}")
        print(f"  Inertia diagonal: [{inertia[0,0]:.0f}, {inertia[1,1]:.0f}, {inertia[2,2]:.0f}] kg*m^2")

    def _compute_notch_coefficients(self):
        """Compute discrete notch filter coefficients using bilinear transform."""
        self.notch_coeffs = []

        # Depth factor: 10^( depth_dB/20) for zero damping
        depth_factor = 10 ** (-self.notch_depth_db / 20.0)
        zeta_z = depth_factor * 0.5  # Zero damping for depth
        zeta_p = self.notch_width  # Pole damping for width

        for f_notch in self.notch_freqs_hz:
            omega_n = 2 * np.pi * f_notch
            # Store continuous time parameters for transfer function analysis
            self.notch_coeffs.append({
                'omega_n': omega_n,
                'zeta_z': zeta_z,
                'zeta_p': zeta_p,
                'freq_hz': f_notch
            })

    def set_target(self, sigma_target: np.ndarray, omega_target: np.ndarray = None):
        """Set target attitude and angular velocity."""
        self.sigma_target = np.array(sigma_target).flatten()
        self.omega_target = np.array(omega_target).flatten() if omega_target is not None else np.zeros(3)

    def reset(self):
        """Reset filter states."""
        for i in range(len(self.notch_states)):
            self.notch_states[i] = np.zeros((3, 2))
        self.last_time = None

    def _apply_notch_filter(self, input_signal: np.ndarray, notch_idx: int, dt: float) -> np.ndarray:
        """
        Apply notch filter to 3-axis signal using state-space form.

        Transfer function:
            H(s) = (s^2 + 2*zeta_z*omega_n*s + omega_n^2) / (s^2 + 2*zeta_p*omega_n*s + omega_n^2)

        Controllable canonical form:
            x' = A*x + B*u
            y  = C*x + D*u
        """
        coeffs = self.notch_coeffs[notch_idx]
        omega_n = coeffs['omega_n']
        zeta_z = coeffs['zeta_z']
        zeta_p = coeffs['zeta_p']

        output = np.zeros(3)
        for axis in range(3):
            x1, x2 = self.notch_states[notch_idx][axis]
            u = input_signal[axis]

            # Semi implicit Euler integration for stability
            x2_new = x2 + dt * (-omega_n**2 * x1 - 2 * zeta_p * omega_n * x2 + u)
            x1_new = x1 + dt * x2_new

            # y = C*x + D*u, where C = [0, 2*omega_n*(zeta_z zeta_p)], D = 1
            output[axis] = 2.0 * omega_n * (zeta_z - zeta_p) * x2_new + u

            self.notch_states[notch_idx][axis] = [x1_new, x2_new]

        return output

    def compute_mrp_error(self, sigma_current: np.ndarray) -> np.ndarray:
        """Compute MRP attitude error."""
        sigma_current = np.array(sigma_current).flatten()
        return _mrp_subtract(sigma_current, self.sigma_target)

    def compute_torque(self,
                       sigma_current: np.ndarray,
                       omega_current: np.ndarray,
                       current_time: float) -> np.ndarray:
        """
        Compute control torque with notch-filtered rate feedback.

        The rate signal passes through notch filters before the derivative gain,
        attenuating controller response at modal frequencies.
        """
        sigma_current = np.array(sigma_current).flatten()
        omega_current = np.array(omega_current).flatten()

        # Compute errors
        sigma_error = self.compute_mrp_error(sigma_current)
        omega_error = omega_current - self.omega_target

        # Apply notch filters to rate error
        omega_filtered = omega_error.copy()
        if self.last_time is not None:
            dt = current_time - self.last_time
            if dt > 0:
                for i in range(len(self.notch_freqs_hz)):
                    omega_filtered = self._apply_notch_filter(omega_filtered, i, dt)

        self.last_time = current_time

        # Compute torque with notch filtered rate
        torque = -self.K * sigma_error - self.P * omega_filtered

        return torque

    def get_transfer_function(self, axis: int = 2) -> signal.TransferFunction:
        """
        Get the notch controller transfer function.

        Returns C(s) = K + P * s * H_notch1(s) * H_notch2(s) * ...
        """
        # Base PD in sigma domain: 4*P*s + K
        num = np.array([4.0 * self.P, self.K])
        den = np.array([1.0])

        # Multiply by each notch filter
        for coeffs in self.notch_coeffs:
            omega_n = coeffs['omega_n']
            zeta_z = coeffs['zeta_z']
            zeta_p = coeffs['zeta_p']

            # Notch: (s^2 + 2*zeta_z*omega_n*s + omega_n^2) / (s^2 + 2*zeta_p*omega_n*s + omega_n^2)
            notch_num = np.array([1.0, 2*zeta_z*omega_n, omega_n**2])
            notch_den = np.array([1.0, 2*zeta_p*omega_n, omega_n**2])

            num = np.convolve(num, notch_num)
            den = np.convolve(den, notch_den)

        return signal.TransferFunction(num, den)

    def get_open_loop_tf(self,
                         axis: int = 2,
                         modal_freqs_hz: List[float] = None,
                         modal_damping: List[float] = None,
                         modal_gains: List[float] = None,
                         include_flexibility: bool = True) -> signal.TransferFunction:
        """Get open-loop transfer function L(s) = G(s) * C(s)."""
        if include_flexibility and modal_freqs_hz:
            plant = _build_flexible_plant_tf(
                self.inertia, axis, modal_freqs_hz, modal_damping, modal_gains
            )
        else:
            I = self.inertia[axis, axis]
            sigma_scale = 4.0
            plant = signal.TransferFunction([1.0], [sigma_scale * I, 0, 0])

        controller = self.get_transfer_function(axis)

        plant_num = np.atleast_1d(np.squeeze(plant.num))
        plant_den = np.atleast_1d(np.squeeze(plant.den))
        ctrl_num = np.atleast_1d(np.squeeze(controller.num))
        ctrl_den = np.atleast_1d(np.squeeze(controller.den))

        ol_num = np.convolve(plant_num, ctrl_num)
        ol_den = np.convolve(plant_den, ctrl_den)

        return signal.TransferFunction(ol_num, ol_den)

    def get_closed_loop_params(self) -> Dict[str, float]:
        """Compute approximate closed-loop parameters."""
        I_avg = np.mean(np.diag(self.inertia))
        sigma_scale = 4.0
        omega_n = np.sqrt(self.K / (sigma_scale * I_avg))
        zeta = self.P / np.sqrt(self.K * I_avg)
        omega_bw = omega_n * np.sqrt(1 - 2*zeta**2 + np.sqrt(4*zeta**4 - 4*zeta**2 + 2))

        return {
            'omega_n': omega_n,
            'omega_n_hz': omega_n / (2 * np.pi),
            'zeta': zeta,
            'omega_bw': omega_bw,
            'omega_bw_hz': omega_bw / (2 * np.pi),
            'notch_freqs_hz': self.notch_freqs_hz,
            'I_avg': I_avg
        }


class TrajectoryTrackingController:
    """
    Feedback controller that tracks instantaneous feedforward trajectory.

    This is the CORRECT way to combine feedforward and feedback control.
    The feedback controller receives time-varying reference signals from
    the feedforward trajectory, so it only corrects for:
    1. Model mismatch (actual vs. designed inertia)
    2. External disturbances (gravity gradient, etc.)
    3. Unmodeled dynamics

    The feedback does NOT fight the feedforward trajectory motion.

    Architecture:
        sigma_ref(t) ──┐
                       ├──> [sigma_error] ──> [-K] ──┐
        sigma_meas ────┘                             ├──> tau_fb
                                                     │
        omega_ref(t) ──┐                             │
                       ├──> [omega_error] ──> [-P] ──┘
        omega_meas ────┘

    Where sigma_ref(t) and omega_ref(t) come from the feedforward trajectory.
    """

    def __init__(self,
                 inertia: np.ndarray,
                 K: float = 30.0,
                 P: float = 60.0,
                 controller_type: str = 'filtered_pd',
                 filter_freq_hz: float = 0.5,
                 notch_freqs_hz: List[float] = None):
        """
        Initialize trajectory tracking controller.

        Args:
            inertia: 3x3 spacecraft inertia matrix [kg*m^2]
            K: Proportional gain
            P: Derivative gain
            controller_type: 'standard_pd', 'filtered_pd', or 'notch'
            filter_freq_hz: Filter cutoff for filtered_pd [Hz]
            notch_freqs_hz: Notch frequencies for notch controller [Hz]
        """
        self.inertia = np.array(inertia)
        self.K = K
        self.P = P
        self.controller_type = controller_type

        # Current reference (updated each timestep from feedforward)
        self.sigma_ref = np.zeros(3)
        self.omega_ref = np.zeros(3)

        # Final target (for after feedforward completes)
        self.sigma_final = np.zeros(3)
        self.omega_final = np.zeros(3)

        # Feedforward trajectory interpolators
        self.trajectory_interp = None
        self.trajectory_duration = 0.0
        self.rotation_axis = np.array([0.0, 0.0, 1.0])

        # Create the underlying controller based on type
        if controller_type == 'standard_pd':
            self.controller = MRPFeedbackController(inertia, K, P, Ki=-1.0)
        elif controller_type == 'filtered_pd':
            self.controller = FilteredDerivativeController(inertia, K, P, filter_freq_hz)
        elif controller_type == 'notch':
            self.controller = NotchFilterController(
                inertia, K, P,
                notch_freqs_hz=notch_freqs_hz or [0.4, 1.3]
            )
        else:
            raise ValueError(f"Unknown controller type: {controller_type}")

        print(f"Trajectory Tracking Controller initialized:")
        print(f"  Type: {controller_type}")
        print(f"  Gains: K={K}, P={P}")

    def set_feedforward_trajectory(self, ff_controller, rotation_axis: np.ndarray):
        """
        Set the feedforward trajectory to track.

        Args:
            ff_controller: FeedforwardController instance with designed trajectory
            rotation_axis: Rotation axis for MRP conversion
        """
        from scipy import interpolate

        self.rotation_axis = np.array(rotation_axis) / np.linalg.norm(rotation_axis)

        if ff_controller.trajectory is None:
            print("Warning: Feedforward controller has no trajectory!")
            return

        t = ff_controller.t_profile
        theta = ff_controller.trajectory['theta']
        omega = ff_controller.trajectory['omega']

        self.trajectory_duration = t[-1]

        # Create interpolators for theta(t) and omega(t)
        self.theta_interp = interpolate.interp1d(
            t, theta, kind='linear', bounds_error=False,
            fill_value=(theta[0], theta[-1])
        )
        self.omega_interp = interpolate.interp1d(
            t, omega, kind='linear', bounds_error=False,
            fill_value=(omega[0], omega[-1])
        )

        # Compute final target MRP
        theta_final = theta[-1]
        self.sigma_final = self.rotation_axis * np.tan(theta_final / 4.0)
        self.omega_final = np.zeros(3)

        print(f"  Trajectory loaded: {self.trajectory_duration:.1f}s, "
              f"final angle {np.degrees(theta_final):.1f} deg")

    def set_final_target(self, sigma_target: np.ndarray, omega_target: np.ndarray = None):
        """Set the final target attitude (for after feedforward completes)."""
        self.sigma_final = np.array(sigma_target).flatten()
        self.omega_final = np.array(omega_target).flatten() if omega_target is not None else np.zeros(3)

    def _get_reference_at_time(self, current_time: float) -> Tuple[np.ndarray, np.ndarray]:
        """Get instantaneous reference from feedforward trajectory."""
        if self.theta_interp is None or current_time > self.trajectory_duration:
            # After feedforward ends, track final target
            return self.sigma_final.copy(), self.omega_final.copy()

        # Get trajectory values
        theta_ref = float(self.theta_interp(current_time))
        omega_scalar = float(self.omega_interp(current_time))

        # Convert to MRP and angular velocity vector
        sigma_ref = self.rotation_axis * np.tan(theta_ref / 4.0)
        omega_ref = self.rotation_axis * omega_scalar

        return sigma_ref, omega_ref

    def compute_torque(self,
                       sigma_current: np.ndarray,
                       omega_current: np.ndarray,
                       current_time: float) -> np.ndarray:
        """
        Compute feedback torque tracking the instantaneous reference.

        Args:
            sigma_current: Current MRP attitude (3,)
            omega_current: Current angular velocity [rad/s] (3,)
            current_time: Current simulation time [s]
        Returns:
            torque: Feedback control torque [N*m] (3,)
        """
        # Get instantaneous reference from feedforward trajectory
        sigma_ref, omega_ref = self._get_reference_at_time(current_time)

        # Update controller target to instantaneous reference
        self.controller.set_target(sigma_ref, omega_ref)

        # Compute torque using underlying controller
        if hasattr(self.controller, 'compute_torque'):
            torque = self.controller.compute_torque(sigma_current, omega_current, current_time)
        else:
            torque = np.zeros(3)

        return torque

    def reset(self):
        """Reset controller states."""
        if hasattr(self.controller, 'reset'):
            self.controller.reset()


class HybridController:
    """
    Hybrid feedforward + feedback controller.
    
    Combines open-loop feedforward trajectory tracking with closed-loop
    feedback for disturbance rejection and fine pointing.
    
    Control modes:
        1. Feedforward only (during slew)
        2. Feedback only (fine pointing)
        3. Combined FF + FB
    """
    
    def __init__(self,
                 inertia: np.ndarray,
                 Gs_matrix: np.ndarray,
                 ff_controller,
                 fb_controller: MRPFeedbackController):
        """
        Initialize hybrid controller.
        
        Args:
            inertia: Spacecraft inertia matrix
            Gs_matrix: RW configuration matrix (3x3)
            ff_controller: Feedforward controller instance
            fb_controller: MRPFeedbackController instance
        """
        self.inertia = np.array(inertia)
        self.Gs_matrix = np.array(Gs_matrix)
        self.ff = ff_controller
        self.fb = fb_controller
        
        # Control mode
        self.mode = 'feedforward'  # 'feedforward', 'feedback', 'hybrid'
        
        # Transition parameters
        self.ff_end_time = None
        self.transition_duration = 1.0  # seconds
        
    def set_mode(self, mode: str, ff_end_time: float = None):
        """
        Set control mode.
        
        Args:
            mode: 'feedforward', 'feedback', or 'hybrid'
            ff_end_time: Time when feedforward ends (for transition)
        """
        self.mode = mode
        self.ff_end_time = ff_end_time
        
    def compute_torque(self,
                       current_time: float,
                       sigma_current: np.ndarray,
                       omega_current: np.ndarray) -> Tuple[np.ndarray, str]:
        """
        Compute control torque based on current mode.
        
        Args:
            current_time: Current simulation time [s]
            sigma_current: Current MRP attitude
            omega_current: Current angular velocity [rad/s]
            
        Returns:
            Tuple of (body_torque, active_mode_str)
        """
        if self.mode == 'feedforward':
            # Pure feedforward
            rw_torques = self.ff.get_torque(current_time)
            body_torque = self.Gs_matrix @ (-rw_torques)
            return body_torque, 'FF'
            
        elif self.mode == 'feedback':
            # Pure feedback
            body_torque = self.fb.compute_torque(sigma_current, omega_current, current_time)
            return body_torque, 'FB'
            
        elif self.mode == 'hybrid':
            # Combined with smooth transition
            ff_torque = np.zeros(3)
            fb_torque = self.fb.compute_torque(sigma_current, omega_current, current_time)
            
            if self.ff_end_time is not None and current_time < self.ff_end_time:
                rw_torques = self.ff.get_torque(current_time)
                ff_torque = self.Gs_matrix @ (-rw_torques)
                
                # Blend factor (1 = full FF, 0 = full FB)
                if current_time > self.ff_end_time - self.transition_duration:
                    blend = (self.ff_end_time - current_time) / self.transition_duration
                else:
                    blend = 1.0
                    
                body_torque = blend * ff_torque + (1 - blend) * fb_torque
                return body_torque, f'HYB({blend:.1f})'
            else:
                return fb_torque, 'FB'
        
        else:
            return np.zeros(3), 'IDLE'


def design_gains_from_bandwidth(inertia: np.ndarray,
                                 bandwidth_hz: float,
                                 damping_ratio: float = 0.7) -> Tuple[float, float]:
    """
    Design feedback gains to achieve desired closed-loop bandwidth.
    
    For a double-integrator plant in sigma (1/(4I s^2)) with PD control on omega:
        omegan = sqrt(K/(4I))
        zeta = P / sqrt(K*I)
        
    Args:
        inertia: Spacecraft inertia matrix
        bandwidth_hz: Desired closed-loop bandwidth [Hz]
        damping_ratio: Desired damping ratio (default: 0.7 for good transient)
        
    Returns:
        Tuple of (K, P) gains
    """
    I_avg = np.mean(np.diag(inertia))
    omega_n = 2 * np.pi * bandwidth_hz
    
    # From omegan = sqrt(K/(4I))
    K = 4.0 * omega_n**2 * I_avg

    # From zeta = P / sqrt(K*I) (sigma_dot ≈ 0.25*omega)
    P = 2.0 * damping_ratio * omega_n * I_avg
    
    print(f"Gain design for {bandwidth_hz:.2f} Hz bandwidth, zeta={damping_ratio}:")
    print(f"  K = {K:.1f}")
    print(f"  P = {P:.1f}")
    print(f"  I_avg = {I_avg:.1f} kg*m^2")
    
    return K, P


if __name__ == "__main__":
    print("="*60)
    print("Feedback Control Module - Test")
    print("="*60)
    
    # Test with spacecraft inertia from vizard_demo.py
    inertia = np.array([
        [900.0, 0.0, 0.0],
        [0.0, 800.0, 0.0],
        [0.0, 0.0, 600.0]
    ])
    
    # Create controller with default gains
    controller = MRPFeedbackController(inertia, K=30.0, P=60.0, Ki=-1.0)
    
    # Set target
    controller.set_target(np.array([0.0, 0.0, 1.0]))  # 180 deg yaw
    
    # Get closed loop parameters
    params = controller.get_closed_loop_params()
    print(f"\nClosed-loop parameters:")
    print(f"  Natural frequency: {params['omega_n_hz']:.3f} Hz")
    print(f"  Damping ratio: {params['zeta']:.3f}")
    print(f"  Bandwidth: {params['omega_bw_hz']:.3f} Hz")
    print(f"  Settling time: {params['t_settle']:.1f} s")
    
    # Test torque computation
    sigma_current = np.array([0.1, 0.0, 0.0])  # Small error
    omega_current = np.array([0.0, 0.0, 0.01])  # Small rate
    
    torque = controller.compute_torque(sigma_current, omega_current)
    print(f"\nTest torque computation:")
    print(f"  sigma_current = {sigma_current}")
    print(f"  omega_current = {omega_current}")
    print(f"  tau_command = {torque}")
    
    # Design gains for different bandwidth
    print("\n" + "="*60)
    print("Gain design example:")
    K_new, P_new = design_gains_from_bandwidth(inertia, bandwidth_hz=0.1, damping_ratio=0.7)
