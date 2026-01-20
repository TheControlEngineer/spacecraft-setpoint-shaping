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

Active Vibration Control (AVC):
    This module also includes active vibration suppression using:
    1. Low-pass filtered derivative feedback (reduces high-frequency gain).
    2. Positive Position Feedback (PPF) compensators for modal damping.
    3. Optional modal velocity feedback if modal states are available.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy import signal


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
    Build a single-axis flexible plant: rigid body + modal resonances in parallel.

    G_total(s) = 1/(I*s^2) + Sum k_i/(s^2 + 2*zeta_i*omega_i*s + omega_i^2)
    """
    I = float(inertia[axis, axis])
    rigid_num = np.array([1.0])
    rigid_den = np.array([I, 0.0, 0.0])

    if not modal_freqs_hz:
        return signal.TransferFunction(rigid_num, rigid_den)

    current_num = rigid_num
    current_den = rigid_den

    for f_mode, zeta, gain in zip(modal_freqs_hz, modal_damping, modal_gains):
        omega_n = 2 * np.pi * float(f_mode)
        mode_num = np.array([float(gain) / I])
        mode_den = np.array([1.0, 2.0 * float(zeta) * omega_n, omega_n**2])

        term1 = np.convolve(current_num, mode_den)
        term2 = np.convolve(mode_num, current_den)

        if len(term1) > len(term2):
            term2 = np.pad(term2, (len(term1) - len(term2), 0), mode='constant')
        elif len(term2) > len(term1):
            term1 = np.pad(term1, (len(term2) - len(term1), 0), mode='constant')

        current_num = term1 + term2
        current_den = np.convolve(current_den, mode_den)

    return signal.TransferFunction(current_num, current_den)


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
        # tau = -K * sigma_error - P * omega_error + Ki * integral(sigma_error dt)
        torque = -self.K * sigma_error - self.P * omega_error
        
        if self.Ki > 0:
            torque += self.Ki * self.sigma_integral
        
        return torque
    
    def get_closed_loop_params(self) -> Dict[str, float]:
        """
        Compute approximate closed-loop parameters.
        
        For a linearized attitude system about small angles:
            I * sigma_ddot + P * sigma_dot + K * sigma = 0

        This gives natural frequency omega_n = sqrt(K/I) and damping
        zeta = P / (2 * sqrt(K * I)).
        
        Returns:
            Dictionary with natural frequency, damping ratio, and bandwidth
        """
        # Use average principal inertia
        I_avg = np.mean(np.diag(self.inertia))
        
        # Natural frequency [rad/s]
        omega_n = np.sqrt(self.K / I_avg)
        
        # Damping ratio
        zeta = self.P / (2 * np.sqrt(self.K * I_avg))
        
        # Closed-loop bandwidth (approximately)
        omega_bw = omega_n * np.sqrt(1 - 2*zeta**2 + np.sqrt(4*zeta**4 - 4*zeta**2 + 2))
        
        # Settling time (2% criterion)
        if zeta < 1:
            t_settle = 4 / (zeta * omega_n)
        else:
            t_settle = 4 * zeta / omega_n
        
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
        
        # Plant: 1/(I*s^2)
        plant_num = [1.0]
        plant_den = [I, 0, 0]
        
        # Controller: P*s + K (PD controller)
        if self.Ki > 0:
            # PID: (Ki + K*s + P*s^2) / s
            ctrl_num = [self.P, self.K, self.Ki]
            ctrl_den = [1, 0]
        else:
            # PD: P*s + K
            ctrl_num = [self.P, self.K]
            ctrl_den = [1]
        
        # Open-loop: G*C
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
        
        # Closed-loop: GC / (1 + GC)
        # For simple PD on double integrator: I*s^2 + P*s + K
        if self.Ki > 0:
            cl_num = [self.P, self.K, self.Ki]
            cl_den = [I, self.P, self.K, self.Ki]
        else:
            cl_num = [self.P, self.K]
            cl_den = [I, self.P, self.K]
        
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
            plant = signal.TransferFunction([1.0], [I, 0, 0])
        
        # Controller: P*s + K (PD controller)
        if self.Ki > 0:
            ctrl_num = [self.P, self.K, self.Ki]
            ctrl_den = [1, 0]
        else:
            ctrl_num = [self.P, self.K]
            ctrl_den = [1]
        
        controller = signal.TransferFunction(ctrl_num, ctrl_den)
        
        # Open-loop = Plant * Controller
        plant_num = np.atleast_1d(np.squeeze(plant.num))
        plant_den = np.atleast_1d(np.squeeze(plant.den))
        ctrl_num = np.atleast_1d(np.squeeze(controller.num))
        ctrl_den = np.atleast_1d(np.squeeze(controller.den))
        
        ol_num = np.convolve(plant_num, ctrl_num)
        ol_den = np.convolve(plant_den, ctrl_den)
        
        return signal.TransferFunction(ol_num, ol_den)


class PPFCompensator:
    """
    Positive Position Feedback (PPF) compensator for active vibration control.
    
    PPF is a second-order compensator that adds damping to structural modes
    while maintaining guaranteed stability for collocated sensors/actuators.
    
    Transfer function:
        H_ppf(s) = g * omega_f^2 / (s^2 + 2*zeta_f*omega_f*s + omega_f^2)
    
    Key properties:
        - Adds positive phase (lead) near the tuned frequency
        - Rolls off at high frequencies (won't excite higher modes)
        - Unconditionally stable with collocated control
        
    References:
        - Goh & Caughey (1985) "On the Stability Problem Caused by Finite 
          Actuator Dynamics in the Control of Large Space Structures"
        - Fanson & Caughey (1990) "Positive Position Feedback Control for 
          Large Space Structures"
    """
    
    def __init__(self, 
                 modal_freq_hz: float,
                 filter_freq_hz: float = None,
                 damping: float = 0.5,
                 gain: float = 1.0):
        """
        Initialize PPF compensator.
        
        Args:
            modal_freq_hz: Target modal frequency to damp [Hz]
            filter_freq_hz: PPF filter frequency [Hz] (default: 0.9 * modal_freq)
                           Set slightly below modal frequency for best damping
            damping: PPF filter damping ratio (default: 0.5)
            gain: Compensator gain (default: 1.0)
        """
        self.modal_freq_hz = modal_freq_hz
        self.filter_freq_hz = filter_freq_hz if filter_freq_hz else 0.9 * modal_freq_hz
        self.damping = damping
        self.gain = gain
        
        # Convert to rad/s
        self.omega_f = 2 * np.pi * self.filter_freq_hz
        
        # State-space representation for real-time implementation
        # x'' + 2*zeta*omega*x' + omega^2*x = g*omega^2*u
        # Using states: [x, x']
        self.state = np.zeros(2)  # [position, velocity]
        
        print(f"PPF Compensator initialized:")
        print(f"  Target mode: {modal_freq_hz:.2f} Hz")
        print(f"  Filter frequency: {self.filter_freq_hz:.2f} Hz")
        print(f"  Filter damping: {damping}")
        print(f"  Gain: {gain}")
    
    def reset(self):
        """Reset internal state."""
        self.state = np.zeros(2)
    
    def get_transfer_function(self) -> signal.TransferFunction:
        """
        Get the PPF transfer function.
        
        Returns:
            H_ppf(s) = g * omega_f^2 / (s^2 + 2*zeta_f*omega_f*s + omega_f^2)
        """
        omega_f = self.omega_f
        zeta_f = self.damping
        g = self.gain
        
        num = [g * omega_f**2]
        den = [1.0, 2*zeta_f*omega_f, omega_f**2]
        
        return signal.TransferFunction(num, den)
    
    def update(self, input_signal: float, dt: float) -> float:
        """
        Update PPF state and compute output.
        
        Uses trapezoidal integration for numerical stability.
        
        Args:
            input_signal: Input (typically attitude error or modal displacement)
            dt: Time step [s]
            
        Returns:
            PPF output (additive torque component)
        """
        omega_f = self.omega_f
        zeta_f = self.damping
        g = self.gain
        
        # State-space: x' = A*x + B*u, y = C*x
        # x = [eta, eta'], where eta is the filter state
        # eta'' + 2*zeta*omega*eta' + omega^2*eta = g*omega^2*u
        
        x, xdot = self.state
        
        # Compute acceleration
        xddot = g * omega_f**2 * input_signal - 2*zeta_f*omega_f*xdot - omega_f**2*x
        
        # Integrate (trapezoidal for xdot, forward Euler for x)
        x_new = x + xdot * dt + 0.5 * xddot * dt**2
        xdot_new = xdot + xddot * dt
        
        self.state = np.array([x_new, xdot_new])
        
        return x_new


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
        
        # Filter state (3-axis)
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
        
        # Update filtered rate (first-order low-pass filter)
        if self.last_time is not None:
            dt = current_time - self.last_time
            if dt > 0:
                # Low-pass filter: tau*omega_f' + omega_f = omega
                # Discretized: omega_f[k+1] = omega_f[k] + (dt/tau)*(omega[k] - omega_f[k])
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
            C(s) = K + P*s / (tau*s + 1) = (K*tau*s + K + P*s) / (tau*s + 1)
                 = ((K*tau + P)*s + K) / (tau*s + 1)
        """
        # C(s) = K + P*s/(tau*s + 1)
        # Combined: ((K*tau + P)*s + K) / (tau*s + 1)
        num = [self.K * self.tau + self.P, self.K]
        den = [self.tau, 1.0]
        
        return signal.TransferFunction(num, den)
    
    def get_closed_loop_params(self) -> Dict[str, float]:
        """Compute approximate closed-loop parameters."""
        I_avg = np.mean(np.diag(self.inertia))
        omega_n = np.sqrt(self.K / I_avg)
        zeta = self.P / (2 * np.sqrt(self.K * I_avg))
        
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
            plant = signal.TransferFunction([1.0], [I, 0, 0])
        
        # Filtered controller: C(s) = K + P*s/(tau*s + 1)
        controller = self.get_transfer_function(axis)
        
        # Open-loop = Plant * Controller
        plant_num = np.atleast_1d(np.squeeze(plant.num))
        plant_den = np.atleast_1d(np.squeeze(plant.den))
        ctrl_num = np.atleast_1d(np.squeeze(controller.num))
        ctrl_den = np.atleast_1d(np.squeeze(controller.den))
        
        ol_num = np.convolve(plant_num, ctrl_num)
        ol_den = np.convolve(plant_den, ctrl_den)
        
        return signal.TransferFunction(ol_num, ol_den)
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


class ActiveVibrationController:
    """
    Complete active vibration control system.
    
    Combines:
    1. Filtered derivative feedback (reduces high-freq excitation)
    2. PPF compensators for each structural mode (adds modal damping)
    
    The controller architecture is:
    
        sigma_ref -->[+]--> [K + P*s/(tau*s+1)] -->[+]--> tau_cmd
                 |-                        |+
                 |                         |
                 <-- sigma_meas                +-- PPF_1(s) * sigma_meas
                                           |
                                           +-- PPF_2(s) * sigma_meas
    
    The PPF compensators feed back on attitude (or modal displacement if 
    available) and add torques that effectively increase modal damping.
    
    Design Guidelines:
        - Set filter_freq_hz between closed-loop bandwidth and first mode
        - Tune PPF gains to add 5-10% effective damping to each mode
        - Higher PPF gain = more damping but higher control effort
    """
    
    def __init__(self,
                 inertia: np.ndarray,
                 K: float = 30.0,
                 P: float = 60.0,
                 filter_freq_hz: float = 0.2,
                 modal_freqs_hz: List[float] = None,
                 modal_damping: List[float] = None,
                 modal_gains: List[float] = None,
                 ppf_damping: float = 0.5,
                 ppf_gains: List[float] = None):
        """
        Initialize active vibration controller.
        
        Args:
            inertia: 3x3 spacecraft inertia matrix [kg*m^2]
            K: Proportional gain
            P: Derivative gain
            filter_freq_hz: Cutoff for derivative filter [Hz]
            modal_freqs_hz: List of modal frequencies to damp [Hz]
            modal_damping: Modal damping ratios for flexible plant modeling
            modal_gains: Modal coupling gains for flexible plant modeling
            ppf_damping: Damping ratio for PPF filters
            ppf_gains: Gains for each PPF compensator (default: auto-computed)
        """
        self.inertia = np.array(inertia)
        self.K = K
        self.P = P

        self.modal_freqs_hz = list(modal_freqs_hz) if modal_freqs_hz is not None else []
        if modal_damping is None:
            if len(self.modal_freqs_hz) == 2:
                self.modal_damping = [0.02, 0.015]
            else:
                self.modal_damping = [0.02] * len(self.modal_freqs_hz)
        else:
            self.modal_damping = list(modal_damping)

        if modal_gains is None:
            if len(self.modal_freqs_hz) == 2:
                self.modal_gains = [0.15, 0.08]
            else:
                self.modal_gains = [0.1] * len(self.modal_freqs_hz)
        else:
            self.modal_gains = list(modal_gains)

        if len(self.modal_damping) < len(self.modal_freqs_hz):
            self.modal_damping.extend([self.modal_damping[-1]] * (len(self.modal_freqs_hz) - len(self.modal_damping)))
        if len(self.modal_gains) < len(self.modal_freqs_hz):
            self.modal_gains.extend([self.modal_gains[-1]] * (len(self.modal_freqs_hz) - len(self.modal_gains)))
        
        # Create filtered derivative controller
        self.pd_controller = FilteredDerivativeController(
            inertia, K, P, filter_freq_hz
        )
        
        # Create PPF compensators for each modal frequency
        self.ppf_compensators = []
        if modal_freqs_hz is not None:
            if ppf_gains is None:
                # Default: gain proportional to modal frequency squared
                # This gives roughly equal damping contribution to each mode
                ppf_gains = [1.0] * len(modal_freqs_hz)
            
            for i, f_mode in enumerate(modal_freqs_hz):
                gain = ppf_gains[i] if i < len(ppf_gains) else 1.0
                ppf = PPFCompensator(
                    modal_freq_hz=f_mode,
                    damping=ppf_damping,
                    gain=gain
                )
                self.ppf_compensators.append(ppf)
        
        # Active axis (Z for yaw)
        self.active_axis = 2
        self.default_ppf_dt = 0.01
        self.ppf_last_time = None
        
        print(f"\n{'='*60}")
        print("Active Vibration Controller Summary:")
        print(f"  Base controller: Filtered PD (K={K}, P={P})")
        print(f"  Derivative filter: {filter_freq_hz:.3f} Hz")
        print(f"  PPF compensators: {len(self.ppf_compensators)}")
        for i, ppf in enumerate(self.ppf_compensators):
            print(f"    Mode {i+1}: {ppf.modal_freq_hz:.2f} Hz, gain={ppf.gain}")
        print(f"{'='*60}\n")
    
    def set_target(self, sigma_target: np.ndarray, omega_target: np.ndarray = None):
        """Set target attitude."""
        self.pd_controller.set_target(sigma_target, omega_target)
    
    def reset(self):
        """Reset all controller states."""
        self.pd_controller.reset()
        for ppf in self.ppf_compensators:
            ppf.reset()
        self.ppf_last_time = None
    
    def compute_torque(self,
                       sigma_current: np.ndarray,
                       omega_current: np.ndarray,
                       current_time: float,
                       modal_displacements: np.ndarray = None) -> np.ndarray:
        """
        Compute active vibration control torque.
        
        Args:
            sigma_current: Current MRP attitude (3,)
            omega_current: Current angular velocity [rad/s] (3,)
            current_time: Current simulation time [s]
            modal_displacements: Optional modal displacement measurements
                                If None, uses attitude error for PPF input
                                
        Returns:
            torque: Control torque command [N*m] (3,)
        """
        sigma_current = np.array(sigma_current).flatten()
        omega_current = np.array(omega_current).flatten()
        
        # Base PD torque (with filtered derivative)
        torque = self.pd_controller.compute_torque(
            sigma_current, omega_current, current_time
        )
        
        # Add PPF contributions
        if self.ppf_compensators:
            # Time step for PPF update
            if self.ppf_last_time is None:
                dt = self.default_ppf_dt
            else:
                dt = current_time - self.ppf_last_time
                if dt <= 0:
                    dt = self.default_ppf_dt
            self.ppf_last_time = current_time
            
            # PPF input: either modal displacement or attitude error on active axis
            if modal_displacements is not None:
                # Use actual modal measurements
                for i, ppf in enumerate(self.ppf_compensators):
                    if i < len(modal_displacements):
                        ppf_output = ppf.update(modal_displacements[i], dt)
                        torque[self.active_axis] += ppf_output
            else:
                # Use attitude error as proxy for modal excitation
                sigma_error = self.pd_controller.compute_mrp_error(sigma_current)
                for ppf in self.ppf_compensators:
                    ppf_output = ppf.update(sigma_error[self.active_axis], dt)
                    torque[self.active_axis] += ppf_output
        
        return torque
    
    def get_open_loop_tf(self,
                         axis: int = 2,
                         include_flexibility: bool = True,
                         include_ppf: bool = True) -> signal.TransferFunction:
        """
        Get combined open-loop transfer function L(s) = G(s) * C_total(s).
        
        Where C_total includes the filtered PD and (optionally) all PPF compensators.
        Flexible modes are included in G(s) if include_flexibility=True.
        """
        if include_flexibility:
            plant = _build_flexible_plant_tf(
                self.inertia,
                axis,
                self.modal_freqs_hz,
                self.modal_damping,
                self.modal_gains
            )
        else:
            I = self.inertia[axis, axis]
            plant = signal.TransferFunction([1.0], [I, 0, 0])

        C_pd = self.pd_controller.get_transfer_function(axis)

        if include_ppf and self.ppf_compensators:
            controller_num = np.atleast_1d(np.squeeze(C_pd.num))
            controller_den = np.atleast_1d(np.squeeze(C_pd.den))

            for ppf in self.ppf_compensators:
                ppf_tf = ppf.get_transfer_function()
                controller_num, controller_den = _tf_add(
                    controller_num, controller_den, ppf_tf.num, ppf_tf.den
                )
            controller = signal.TransferFunction(controller_num, controller_den)
        else:
            controller = C_pd

        plant_num = np.atleast_1d(np.squeeze(plant.num))
        plant_den = np.atleast_1d(np.squeeze(plant.den))
        ctrl_num = np.atleast_1d(np.squeeze(controller.num))
        ctrl_den = np.atleast_1d(np.squeeze(controller.den))

        ol_num = np.convolve(plant_num, ctrl_num)
        ol_den = np.convolve(plant_den, ctrl_den)

        return signal.TransferFunction(ol_num, ol_den)
    
    def analyze_vibration_suppression(self, 
                                       modal_freqs_hz: List[float] = None,
                                       modal_damping: List[float] = None):
        """
        Analyze the vibration suppression capability.
        
        Computes:
        1. Controller gain at each modal frequency (before and after filtering)
        2. Effective damping added by PPF compensators
        3. Closed-loop modal damping estimate
        """
        if modal_freqs_hz is None:
            modal_freqs_hz = self.modal_freqs_hz if self.modal_freqs_hz else [0.4, 1.3]
        if modal_damping is None:
            modal_damping = self.modal_damping if self.modal_damping else [0.02, 0.015]
        
        print("\n" + "="*60)
        print("VIBRATION SUPPRESSION ANALYSIS")
        print("="*60)
        
        # Frequency vector for analysis
        freqs = np.logspace(-2, 1, 1000)  # 0.01 to 10 Hz
        omega = 2 * np.pi * freqs
        s = 1j * omega
        
        # Unfiltered PD response: C(s) = P*s + K
        C_unfiltered = self.P * s + self.K
        C_unfiltered_dB = 20 * np.log10(np.abs(C_unfiltered))
        
        # Filtered PD response: C(s) = K + P*s/(tau*s + 1)
        tau = self.pd_controller.tau
        C_filtered = self.K + self.P * s / (tau * s + 1)
        C_filtered_dB = 20 * np.log10(np.abs(C_filtered))
        
        print(f"\nController Gain at Modal Frequencies:")
        print(f"{'Mode':<10} {'Freq [Hz]':<12} {'Unfiltered [dB]':<18} {'Filtered [dB]':<18} {'Reduction [dB]':<15}")
        print("-" * 75)
        
        for i, f_mode in enumerate(modal_freqs_hz):
            idx = np.argmin(np.abs(freqs - f_mode))
            unf_dB = C_unfiltered_dB[idx]
            filt_dB = C_filtered_dB[idx]
            reduction = unf_dB - filt_dB
            print(f"Mode {i+1:<5} {f_mode:<12.2f} {unf_dB:<18.1f} {filt_dB:<18.1f} {reduction:<15.1f}")
        
        # PPF damping analysis
        print(f"\nPPF Compensator Damping Contribution:")
        print(f"{'Mode':<10} {'Open-loop zeta':<15} {'PPF Added zeta':<15} {'Effective zeta':<15}")
        print("-" * 55)
        
        for i, (f_mode, zeta_open) in enumerate(zip(modal_freqs_hz, modal_damping)):
            if i < len(self.ppf_compensators):
                ppf = self.ppf_compensators[i]
                # Approximate added damping from PPF
                # For PPF tuned at omega_f ~= omega_mode, added damping ~= g/(2*omega_mode)
                # This is a simplified estimate
                zeta_added = ppf.gain * ppf.damping * 0.1  # Approximate
                zeta_eff = zeta_open + zeta_added
                print(f"Mode {i+1:<5} {zeta_open:<15.3f} {zeta_added:<15.3f} {zeta_eff:<15.3f}")
            else:
                print(f"Mode {i+1:<5} {zeta_open:<15.3f} {'N/A':<15} {zeta_open:<15.3f}")
        
        print("\n" + "="*60)


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
    
    For a double-integrator plant (1/Is^2) with PD control:
        omegan = sqrt(K/I)
        zeta = P / (2*sqrt(K*I))
        
    Args:
        inertia: Spacecraft inertia matrix
        bandwidth_hz: Desired closed-loop bandwidth [Hz]
        damping_ratio: Desired damping ratio (default: 0.7 for good transient)
        
    Returns:
        Tuple of (K, P) gains
    """
    I_avg = np.mean(np.diag(inertia))
    omega_n = 2 * np.pi * bandwidth_hz
    
    # From omegan = sqrt(K/I)
    K = omega_n**2 * I_avg
    
    # From zeta = P / (2*sqrt(K*I))
    P = 2 * damping_ratio * np.sqrt(K * I_avg)
    
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
    
    # Get closed-loop parameters
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
