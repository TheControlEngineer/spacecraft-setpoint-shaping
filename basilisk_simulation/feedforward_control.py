"""
Feedforward Control Module.

Provides pure feedforward torque profiles for attitude slew maneuvers.
No feedback is used, which isolates the effect of input shaping on
flexible mode excitation.

For yaw (Z-axis) maneuvers on this spacecraft:
- rotation_axis = [0, 0, 1]
- This excites Y-positioned solar arrays bending in Z.
- Input shaping reduces post-slew residual vibration.

Simplified Euler equation: torque = J * alpha (principal axis rotation).
"""

from __future__ import annotations

import numpy as np
from scipy import interpolate


def compute_bang_bang_trajectory(theta_final, duration, dt=0.001):
    """
    Compute a bang-bang trajectory with constant accel then decel.

    Inputs:
    - theta_final: target rotation angle [rad].
    - duration: total maneuver duration [s].
    - dt: sample time [s].

    Outputs:
    - (t, theta, omega, alpha) arrays.

    Process:
    - Apply constant acceleration for the first half and constant deceleration
      for the second half to hit the target angle in the given duration.
    """
    t = np.arange(0, duration + dt, dt)
    n = len(t)
    t_half = duration / 2
    
    # Maximum acceleration for bang-bang: alpha_max = 4 * theta_final / T^2
    alpha_max = 4 * theta_final / duration**2
    
    theta = np.zeros(n)
    omega = np.zeros(n)
    alpha = np.zeros(n)
    
    for i, ti in enumerate(t):
        if ti <= t_half:
            # Acceleration phase
            alpha[i] = alpha_max
            omega[i] = alpha_max * ti
            theta[i] = 0.5 * alpha_max * ti**2
        else:
            # Deceleration phase
            t_dec = ti - t_half
            alpha[i] = -alpha_max
            omega[i] = alpha_max * t_half - alpha_max * t_dec
            theta[i] = (0.5 * alpha_max * t_half**2 + 
                       alpha_max * t_half * t_dec - 
                       0.5 * alpha_max * t_dec**2)
    
    return t, theta, omega, alpha


def compute_smooth_trajectory(theta_final, duration, dt=0.001, smooth_fraction=0.2):
    """
    Compute a smooth trajectory using cosine ramps to reduce high-frequency content.

    Inputs:
    - theta_final: target rotation angle [rad].
    - duration: total maneuver duration [s].
    - dt: sample time [s].
    - smooth_fraction: fraction of duration used for acceleration ramps.

    Outputs:
    - (t, theta, omega, alpha) arrays.
    """
    t = np.arange(0, duration + dt, dt)
    n = len(t)
    
    # Acceleration is symmetric: accelerate for half the maneuver, then decelerate.
    t_acc = duration / 2.0
    t_ramp = smooth_fraction * duration
    max_ramp = 0.5 * t_acc
    if t_ramp > max_ramp:
        t_ramp = max_ramp
    t_const = max(t_acc - 2 * t_ramp, 0.0)

    theta = np.zeros(n)
    omega = np.zeros(n)
    alpha = np.zeros(n)

    def accel_profile(time_in_accel: float) -> float:
        if t_ramp <= 0:
            return 1.0
        if time_in_accel <= t_ramp:
            return 0.5 * (1 - np.cos(np.pi * time_in_accel / t_ramp))
        if time_in_accel <= t_ramp + t_const:
            return 1.0
        if time_in_accel <= t_acc:
            t_dec = time_in_accel - t_ramp - t_const
            return 0.5 * (1 + np.cos(np.pi * t_dec / t_ramp))
        return 0.0

    for i, ti in enumerate(t):
        if ti <= t_acc:
            alpha[i] = accel_profile(ti)
        else:
            t_mirror = duration - ti
            alpha[i] = -accel_profile(t_mirror)

    # Integrate to get omega and theta, then scale to hit target angle.
    omega = np.cumsum(alpha) * dt
    theta = np.cumsum(omega) * dt
    scale = theta_final / (theta[-1] + 1e-12)
    alpha *= scale
    omega *= scale
    theta *= scale
    
    return t, theta, omega, alpha


def compute_step_command_torque(theta_final, axis, inertia, duration, dt=0.001, 
                                 trajectory_type='bang-bang'):
    """
    Compute the body torque profile for an unshaped step command.

    Inputs:
    - theta_final: target rotation angle [rad].
    - axis: rotation axis (3,).
    - inertia: inertia matrix (3x3).
    - duration: maneuver duration [s].
    - dt: sample time [s].
    - trajectory_type: 'bang-bang' or 'smooth'.

    Outputs:
    - (t, torque, trajectory_dict) where torque is Nx3.
    """
    if trajectory_type == 'bang-bang':
        t, theta, omega, alpha = compute_bang_bang_trajectory(theta_final, duration, dt)
    else:
        t, theta, omega, alpha = compute_smooth_trajectory(theta_final, duration, dt)
    
    # Body torque: tau = J * alpha * axis
    # For single-axis rotation about body axis
    axis = np.array(axis) / np.linalg.norm(axis)
    
    # Get moment of inertia about rotation axis: I_axis = axis^T * J * axis
    I_axis = axis @ inertia @ axis
    
    # Torque magnitude about axis
    torque_mag = I_axis * alpha
    
    # Torque vector
    torque = np.outer(torque_mag, axis)
    
    trajectory = {
        'theta': theta,
        'omega': omega,
        'alpha': alpha
    }
    
    return t, torque, trajectory


def apply_input_shaper_to_torque(t, torque, shaper_amplitudes, shaper_times, trajectory=None, inertia=None, rotation_axis=None):
    """
    Apply an input shaper to a torque profile via discrete convolution.

    Inputs:
    - t: base time array.
    - torque: base torque profile (Nx3).
    - shaper_amplitudes: shaper impulse amplitudes.
    - shaper_times: shaper impulse times [s].
    - trajectory: optional base trajectory dict (theta, omega, alpha).
    - inertia: inertia matrix for trajectory recomputation.
    - rotation_axis: axis used for torque/trajectory mapping.

    Outputs:
    - (t_shaped, torque_shaped, trajectory_shaped).
    """
    dt = t[1] - t[0]
    
    # Extended time to account for shaper duration
    shaper_duration = shaper_times[-1]
    t_shaped = np.arange(0, t[-1] + shaper_duration + dt, dt)
    n_shaped = len(t_shaped)
    
    torque_shaped = np.zeros((n_shaped, 3))
    
    # Convolve: tau_shaped(t) = Sum A_i * tau(t - t_i)
    for amp, t_imp in zip(shaper_amplitudes, shaper_times):
        # Shift original torque by t_imp
        shift_idx = int(round(t_imp / dt))
        
        for i in range(len(torque)):
            idx_shaped = i + shift_idx
            if idx_shaped < n_shaped:
                torque_shaped[idx_shaped] += amp * torque[i]
    
    # Compute shaped trajectory if inputs provided
    trajectory_shaped = None
    if trajectory is not None and inertia is not None and rotation_axis is not None:
        # Also convolve the trajectory components
        axis = np.array(rotation_axis) / np.linalg.norm(rotation_axis)
        I_axis = axis @ inertia @ axis
        
        # Method 1: Convolve original trajectory (more accurate)
        alpha_orig = trajectory['alpha']
        omega_orig = trajectory['omega']
        theta_orig = trajectory['theta']
        
        # Extend and convolve
        alpha_shaped = np.zeros(n_shaped)
        for amp, t_imp in zip(shaper_amplitudes, shaper_times):
            shift_idx = int(round(t_imp / dt))
            for i in range(len(alpha_orig)):
                idx = i + shift_idx
                if idx < n_shaped:
                    alpha_shaped[idx] += amp * alpha_orig[i]
        
        # Integrate to get omega and theta
        omega_shaped = np.cumsum(alpha_shaped) * dt
        theta_shaped = np.cumsum(omega_shaped) * dt
        
        trajectory_shaped = {
            'theta': theta_shaped,
            'omega': omega_shaped,
            'alpha': alpha_shaped
        }
    
    return t_shaped, torque_shaped, trajectory_shaped


def body_torque_to_rw_torque(body_torque, Gs_matrix):
    """
    Map body torque to RW motor torques using the pseudo-inverse.

    Input:
    - body_torque: array of body torques (Nx3).
    - Gs_matrix: RW spin-axis matrix (3 x N).

    Output:
    - rw_torque: array of wheel torques (NxNw).
    """
    Gs_pinv = np.linalg.pinv(Gs_matrix)
    
    # tau_motor = -Gs_pinv @ tau_body (reaction torque opposes motor torque)
    rw_torque = -body_torque @ Gs_pinv.T
    
    return rw_torque


def compute_minimum_duration(theta_final, rotation_axis, inertia, Gs_matrix, max_rw_torque):
    """
    Compute the minimum maneuver duration to avoid RW torque saturation.

    Inputs:
    - theta_final: target rotation angle [rad].
    - rotation_axis: unit rotation axis (3,).
    - inertia: inertia matrix (3x3).
    - Gs_matrix: RW spin-axis matrix (3 x N).
    - max_rw_torque: per-wheel torque limit [Nm].

    Output:
    - Minimum feasible duration [s] for a bang-bang profile.
    """
    axis = np.array(rotation_axis) / np.linalg.norm(rotation_axis)
    
    # Moment of inertia about rotation axis
    I_axis = axis @ inertia @ axis
    
    # Maximum body torque we can command
    # For 3 RWs in pyramid config, effective torque is approximately:
    # tau_body_max ~= sqrt(2) * max_rw_torque for axes in the plane
    # For single-axis (z), it's just max_rw_torque
    # Conservative estimate: use worst case
    
    # Compute using pseudo-inverse to find what body torque we can achieve
    Gs_pinv = np.linalg.pinv(Gs_matrix)
    # Test with unit body torque along axis to find scaling
    # tau_motor = -Gs_pinv @ tau_body, so for unit body torque:
    unit_body_torque = axis
    rw_for_unit = -unit_body_torque @ Gs_pinv.T  # Negative for reaction physics
    max_rw_needed = np.max(np.abs(rw_for_unit))
    
    # Scale factor to keep within limits
    torque_scale = max_rw_torque / max_rw_needed if max_rw_needed > 0 else max_rw_torque
    
    # Maximum body torque achievable
    tau_body_max = torque_scale  # Since we computed for unit torque
    
    # For bang-bang: T_min = sqrt(4 * I * theta / tau_max)
    T_min = np.sqrt(4 * I_axis * abs(theta_final) / tau_body_max)
    
    return T_min


def create_feedforward_torque_profile(theta_final, rotation_axis, inertia, 
                                       Gs_matrix, maneuver_duration,
                                       shaper_amplitudes=None, shaper_times=None,
                                       dt=0.001, trajectory_type='bang-bang',
                                       max_torque=None):
    """
    Create a complete feedforward torque profile for a slew maneuver.

    Inputs:
    - theta_final: target rotation angle [rad].
    - rotation_axis: unit rotation axis (3,).
    - inertia: inertia matrix (3x3).
    - Gs_matrix: RW spin-axis matrix (3 x N).
    - maneuver_duration: base maneuver duration [s].
    - shaper_amplitudes/shaper_times: optional input shaper.
    - dt: sample time [s].
    - trajectory_type: 'bang-bang' or 'smooth'.
    - max_torque: optional wheel torque limit [Nm].

    Outputs:
    - (t, rw_torque, body_torque, trajectory_dict).
    """
    # Compute base torque profile
    t, body_torque, trajectory = compute_step_command_torque(
        theta_final, rotation_axis, inertia, maneuver_duration, dt, trajectory_type
    )
    
    # Apply input shaper if provided
    if shaper_amplitudes is not None and shaper_times is not None:
        t, body_torque, trajectory_shaped = apply_input_shaper_to_torque(
            t, body_torque, shaper_amplitudes, shaper_times,
            trajectory=trajectory, inertia=inertia, rotation_axis=rotation_axis
        )
        # Use the shaped trajectory for reference
        if trajectory_shaped is not None:
            trajectory = trajectory_shaped
    
    # Map to RW torques
    rw_torque = body_torque_to_rw_torque(body_torque, Gs_matrix)
    
    # Check for saturation
    if max_torque is not None:
        rw_torque_clipped = np.clip(rw_torque, -max_torque, max_torque)
        if not np.allclose(rw_torque, rw_torque_clipped):
            print(f"Warning: RW torque saturated at {max_torque} Nm")
            rw_torque = rw_torque_clipped
    
    return t, rw_torque, body_torque, trajectory


class FeedforwardController:
    """
    Feedforward controller for Basilisk simulation.

    Inputs:
    - inertia: spacecraft inertia matrix (3x3).
    - Gs_matrix: RW spin-axis matrix (3 x N).
    - max_torque: per-wheel torque limit [Nm].

    Output:
    - Provides interpolated RW torque commands via get_torque().

    Process:
    - Pre-compute a torque profile for the requested maneuver and build
      interpolators for use in a time-stepped simulation.
    """
    
    def __init__(self, inertia, Gs_matrix, max_torque=0.2):
        """
        Initialize the feedforward controller.

        Inputs:
        - inertia: spacecraft inertia matrix (3x3).
        - Gs_matrix: RW spin-axis matrix (3 x N).
        - max_torque: per-wheel torque limit [Nm].
        """
        self.inertia = np.array(inertia).reshape(3, 3)
        self.Gs_matrix = np.array(Gs_matrix).reshape(3, -1)
        self.max_torque = max_torque
        
        self.t_profile = None
        self.rw_torque_profile = None
        self.torque_interp = None
        
    def design_maneuver(self, theta_final, rotation_axis, duration=None,
                        shaper_amplitudes=None, shaper_times=None,
                        trajectory_type='bang-bang'):
        """
        Design a feedforward torque profile for a maneuver.

        Inputs:
        - theta_final: target rotation angle [rad].
        - rotation_axis: unit rotation axis (3,).
        - duration: optional duration [s]; if None, auto-sized.
        - shaper_amplitudes/shaper_times: optional input shaper.
        - trajectory_type: 'bang-bang' or 'smooth'.
        """
        # Compute minimum duration
        T_min = compute_minimum_duration(
            theta_final, rotation_axis, self.inertia, 
            self.Gs_matrix, self.max_torque
        )
        
        if duration is None:
            duration = T_min * 1.1  # Add 10% margin
            print(f"  Auto-computed duration: {duration:.2f}s (min={T_min:.2f}s)")
        elif duration < T_min:
            print(f"  WARNING: Duration {duration:.1f}s < minimum {T_min:.1f}s")
            print(f"           Extending to {T_min*1.1:.1f}s to avoid saturation")
            duration = T_min * 1.1
        
        t, rw_torque, body_torque, traj = create_feedforward_torque_profile(
            theta_final=theta_final,
            rotation_axis=rotation_axis,
            inertia=self.inertia,
            Gs_matrix=self.Gs_matrix,
            maneuver_duration=duration,
            shaper_amplitudes=shaper_amplitudes,
            shaper_times=shaper_times,
            dt=0.001,
            trajectory_type=trajectory_type,
            max_torque=self.max_torque
        )
        
        self.t_profile = t
        self.rw_torque_profile = rw_torque
        self.trajectory = traj
        self.is_shaped = shaper_amplitudes is not None
        
        # Create interpolators for each wheel
        n_wheels = rw_torque.shape[1]
        self.torque_interp = []
        for i in range(n_wheels):
            interp = interpolate.interp1d(
                t, rw_torque[:, i], 
                kind='linear', 
                bounds_error=False, 
                fill_value=0.0
            )
            self.torque_interp.append(interp)
        
        total_duration = t[-1]
        print(f"Feedforward maneuver: {np.degrees(theta_final):.1f} deg, {total_duration:.2f}s, peak torque {np.max(np.abs(rw_torque)):.4f} Nm")
        
    def get_torque(self, t):
        """
        Get RW torque command at time t.

        Input:
        - t: time in seconds.

        Output:
        - torque command array (Nw,).
        """
        if self.torque_interp is None:
            return np.zeros(3)
        
        torque = np.array([interp(t) for interp in self.torque_interp])
        return torque
    
    def load_fourth_order_trajectory(self, trajectory_file, rotation_axis):
        """
        Load a precomputed fourth-order trajectory from file.

        Inputs:
        - trajectory_file: NPZ file with time/theta/omega/alpha arrays.
        - rotation_axis: unit rotation axis (3,).

        Output:
        - Populates internal torque profile and interpolators.
        """
        # Load trajectory
        traj_data = np.load(trajectory_file, allow_pickle=True)
        
        t = traj_data['time']
        theta = traj_data['theta']
        omega = traj_data['omega']
        alpha = traj_data['alpha']
        
        print(f"Loaded 4th-order trajectory: {t[-1]:.1f}s, final angle {np.degrees(theta[-1]):.1f} deg")
        
        # Compute body torques from acceleration
        rotation_axis = np.array(rotation_axis) / np.linalg.norm(rotation_axis)
        I_axis = rotation_axis @ self.inertia @ rotation_axis
        
        # Body torque magnitude
        torque_mag = I_axis * alpha
        
        # Body torque vector (Nx3)
        body_torque = np.outer(torque_mag, rotation_axis)
        
        # Map to RW torques
        rw_torque = body_torque_to_rw_torque(body_torque, self.Gs_matrix)
        
        # Store
        self.t_profile = t
        self.rw_torque_profile = rw_torque
        self.trajectory = {
            'theta': theta,
            'omega': omega,
            'alpha': alpha
        }
        self.is_shaped = True
        
        # Create interpolators
        n_wheels = rw_torque.shape[1]
        self.torque_interp = []
        for i in range(n_wheels):
            interp = interpolate.interp1d(
                t, rw_torque[:, i],
                kind='linear',
                bounds_error=False,
                fill_value=0.0
            )
            self.torque_interp.append(interp)
    
    def get_profile(self):
        """
        Return the stored torque profile for plotting.

        Output:
        - (t_profile, rw_torque_profile)
        """
        return self.t_profile, self.rw_torque_profile


if __name__ == "__main__":
    # Test the feedforward controller
    import matplotlib.pyplot as plt
    
    # Spacecraft parameters (must match spacecraft_model.py)
    # Flexible modes are modeled separately; use hub inertia for rigid-body slews.
    inertia = np.diag([900.0, 800.0, 600.0])  # kg*m^2 [I_xx, I_yy, I_zz]
    # Gs matrix: columns are wheel spin axes
    Gs_matrix = np.array([
        [np.sqrt(2)/2, -np.sqrt(2)/2, 0.0],
        [np.sqrt(2)/2,  np.sqrt(2)/2, 0.0],
        [0.0,           0.0,          1.0]
    ])
    
    # Test 180-degree YAW (Z-axis) - matches spacecraft_model.py configuration
    theta = np.radians(180.0)
    axis = np.array([0.0, 0.0, 1.0])  # Z-axis (YAW) for strong flex coupling
    duration = 30.0  # seconds
    
    # Create unshaped profile
    ff_unshaped = FeedforwardController(inertia, Gs_matrix)
    ff_unshaped.design_maneuver(theta, axis, duration, trajectory_type='bang-bang')
    
    t1, torque1 = ff_unshaped.get_profile()
    
    # Create shaped profile (load shaper if available)
    try:
        shaper = np.load('spacecraft_shaper.npz')
        ff_shaped = FeedforwardController(inertia, Gs_matrix)
        ff_shaped.design_maneuver(theta, axis, duration,
                                   shaper_amplitudes=shaper['amplitudes'],
                                   shaper_times=shaper['times'],
                                   trajectory_type='bang-bang')
        t2, torque2 = ff_shaped.get_profile()
        has_shaped = True
    except:
        has_shaped = False
        print("No shaper file found - only showing unshaped profile")
    
    # Plot
    fig, axes = plt.subplots(2, 1, figsize=(12, 8))
    
    ax1 = axes[0]
    for i in range(3):
        ax1.plot(t1, torque1[:, i], label=f'RW {i+1}')
    ax1.set_ylabel('RW Torque (Nm)')
    ax1.set_xlabel('Time (s)')
    ax1.set_title('Unshaped Feedforward Torque Profile', fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    if has_shaped:
        ax2 = axes[1]
        for i in range(3):
            ax2.plot(t2, torque2[:, i], label=f'RW {i+1}')
        ax2.set_ylabel('RW Torque (Nm)')
        ax2.set_xlabel('Time (s)')
        ax2.set_title('Shaped Feedforward Torque Profile', fontweight='bold')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('feedforward_profiles.png', dpi=150)
    plt.show()
