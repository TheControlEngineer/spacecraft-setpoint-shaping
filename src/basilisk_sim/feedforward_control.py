"""
Feedforward Control Module

Provides pure feedforward torque profiles for attitude slew maneuvers.
No feedback is used, which isolates the effect of input shaping on
flexible mode excitation.

For yaw (Z axis) maneuvers on this spacecraft:
    rotation_axis = [0, 0, 1]
    This excites Y positioned solar arrays bending in Z.
    Input shaping reduces post slew residual vibration.

The fundamental relationship used throughout this module:
    torque = J * alpha
where J is the moment of inertia about the rotation axis and alpha
is the angular acceleration, valid for principal axis rotation.
"""

from __future__ import annotations

import numpy as np
from scipy import interpolate

# Sampling period that matches the Basilisk simulation integration step.
# All trajectory profiles are discretized at this rate.
DEFAULT_SAMPLE_DT = 0.01  # 100 Hz


def compute_bang_bang_trajectory(theta_final, duration, dt=DEFAULT_SAMPLE_DT):
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

    # For a bang bang profile the peak acceleration is derived from
    # theta_final = 0.5 * alpha_max * (T/2)^2 * 2, giving:
    alpha_max = 4 * theta_final / duration**2

    theta = np.zeros(n)
    omega = np.zeros(n)
    alpha = np.zeros(n)

    for i, ti in enumerate(t):
        if ti <= t_half:
            # First half: constant positive acceleration
            alpha[i] = alpha_max
            omega[i] = alpha_max * ti
            theta[i] = 0.5 * alpha_max * ti**2
        else:
            # Second half: constant negative acceleration to brake
            t_dec = ti - t_half
            alpha[i] = -alpha_max
            omega[i] = alpha_max * t_half - alpha_max * t_dec
            theta[i] = (0.5 * alpha_max * t_half**2 +
                       alpha_max * t_half * t_dec -
                       0.5 * alpha_max * t_dec**2)

    return t, theta, omega, alpha


def compute_smooth_trajectory(theta_final, duration, dt=DEFAULT_SAMPLE_DT, smooth_fraction=0.2):
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
    
    # Symmetric profile: accelerate for half the maneuver, then decelerate.
    # The ramp duration limits how fast the acceleration transitions
    # from zero to its peak, using a cosine taper for smooth onset.
    t_acc = duration / 2.0
    t_ramp = smooth_fraction * duration
    max_ramp = 0.5 * t_acc
    if t_ramp > max_ramp:
        t_ramp = max_ramp
    # Constant acceleration plateau between the two ramps
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

    # Numerical integration via cumulative sum, then rescale all
    # kinematic quantities so the final position matches theta_final exactly.
    omega = np.cumsum(alpha) * dt
    theta = np.cumsum(omega) * dt
    scale = theta_final / (theta[-1] + 1e-12)
    alpha *= scale
    omega *= scale
    theta *= scale

    return t, theta, omega, alpha


def compute_trapezoidal_trajectory(theta_final, duration, dt=DEFAULT_SAMPLE_DT, accel_fraction=0.40):
    """
    Compute a symmetric trapezoidal-velocity trajectory.

    Inputs:
    - theta_final: target rotation angle [rad].
    - duration: total maneuver duration [s].
    - dt: sample time [s].
    - accel_fraction: fraction of duration used for acceleration (and deceleration).
      For fixed duration, values near 0.5 minimize peak acceleration.

    Outputs:
    - (t, theta, omega, alpha) arrays.
    """
    t = np.arange(0, duration + dt, dt)
    n = len(t)

    # Acceleration phase duration, clamped so it never exceeds half the maneuver
    t_acc = float(np.clip(accel_fraction * duration, dt, 0.5 * duration))
    # Remaining time at constant (peak) velocity
    t_const = max(duration - 2.0 * t_acc, 0.0)
    # Peak acceleration derived from the kinematic constraint theta = integral(omega dt)
    denom = t_acc * (t_acc + t_const)
    alpha_max = theta_final / denom if denom > 0 else 0.0

    theta = np.zeros(n)
    omega = np.zeros(n)
    alpha = np.zeros(n)

    omega_plateau = alpha_max * t_acc
    theta_after_acc = 0.5 * alpha_max * t_acc**2
    theta_after_const = theta_after_acc + omega_plateau * t_const

    for i, ti in enumerate(t):
        if ti <= t_acc:
            alpha[i] = alpha_max
            omega[i] = alpha_max * ti
            theta[i] = 0.5 * alpha_max * ti**2
        elif ti <= t_acc + t_const:
            tau = ti - t_acc
            alpha[i] = 0.0
            omega[i] = omega_plateau
            theta[i] = theta_after_acc + omega_plateau * tau
        else:
            tau = ti - (t_acc + t_const)
            alpha[i] = -alpha_max
            omega[i] = omega_plateau - alpha_max * tau
            theta[i] = theta_after_const + omega_plateau * tau - 0.5 * alpha_max * tau**2

    if abs(theta[-1]) > 1e-12:
        scale = theta_final / theta[-1]
        alpha *= scale
        omega *= scale
        theta *= scale

    return t, theta, omega, alpha


def compute_s_curve_trajectory(theta_final, duration, dt=DEFAULT_SAMPLE_DT):
    """
    Compute a minimum-jerk S-curve trajectory.

    Inputs:
    - theta_final: target rotation angle [rad].
    - duration: total maneuver duration [s].
    - dt: sample time [s].

    Outputs:
    - (t, theta, omega, alpha) arrays.
    """
    t = np.arange(0, duration + dt, dt)
    if duration <= 0:
        return t, np.zeros_like(t), np.zeros_like(t), np.zeros_like(t)

    # Normalized time on [0, 1]
    s = np.clip(t / duration, 0.0, 1.0)

    # Fifth order (minimum jerk) polynomial: s(t) = 10s^3 - 15s^4 + 6s^5
    # Guarantees zero velocity, zero acceleration, and zero jerk at both endpoints.
    theta = theta_final * (10 * s**3 - 15 * s**4 + 6 * s**5)
    omega = (theta_final / duration) * (30 * s**2 - 60 * s**3 + 30 * s**4)
    alpha = (theta_final / duration**2) * (60 * s - 180 * s**2 + 120 * s**3)
    return t, theta, omega, alpha


def compute_step_command_torque(theta_final, axis, inertia, duration, dt=DEFAULT_SAMPLE_DT, 
                                 trajectory_type='bang-bang'):
    """
    Compute the body torque profile for an unshaped step command.

    Inputs:
    - theta_final: target rotation angle [rad].
    - axis: rotation axis (3,).
    - inertia: inertia matrix (3x3).
    - duration: maneuver duration [s].
    - dt: sample time [s].
    - trajectory_type: 'bang-bang', 'trapezoidal', 'smooth', or 's_curve'.

    Outputs:
    - (t, torque, trajectory_dict) where torque is Nx3.
    """
    if trajectory_type in {'bang-bang', 'unshaped'}:
        t, theta, omega, alpha = compute_bang_bang_trajectory(theta_final, duration, dt)
    elif trajectory_type == 'trapezoidal':
        t, theta, omega, alpha = compute_trapezoidal_trajectory(theta_final, duration, dt)
    elif trajectory_type in {'s_curve', 's-curve'}:
        t, theta, omega, alpha = compute_s_curve_trajectory(theta_final, duration, dt)
    elif trajectory_type == 'smooth':
        t, theta, omega, alpha = compute_smooth_trajectory(theta_final, duration, dt)
    else:
        raise ValueError(f"Unknown trajectory_type: {trajectory_type}")
    
    # For single axis rotation the scalar Euler equation gives:
    #   tau_scalar = I_axis * alpha
    # where I_axis is the projection of the inertia tensor onto the rotation axis.
    axis = np.array(axis) / np.linalg.norm(axis)

    # Scalar moment of inertia about rotation axis: I_axis = e^T J e
    I_axis = axis @ inertia @ axis

    # Scalar torque at each time step
    torque_mag = I_axis * alpha

    # Expand to 3 component body torque vector along the rotation axis
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

    # The shaped signal is longer than the original because the shaper
    # introduces a time delay equal to the last impulse time.
    shaper_duration = shaper_times[-1]
    t_shaped = np.arange(0, t[-1] + shaper_duration + dt, dt)
    n_shaped = len(t_shaped)

    torque_shaped = np.zeros((n_shaped, 3))

    # Discrete convolution: each shaper impulse produces a time shifted,
    # amplitude scaled copy of the original torque. The superposition of
    # all copies cancels vibration at the target modal frequencies.
    for amp, t_imp in zip(shaper_amplitudes, shaper_times):
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
    # Pseudo inverse maps the desired body torque to individual wheel commands.
    Gs_pinv = np.linalg.pinv(Gs_matrix)

    # Negative sign from Newtons third law: the wheels must spin up in the
    # opposite direction to produce the desired body torque.
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

    # Scalar inertia about the commanded rotation axis
    I_axis = axis @ inertia @ axis

    # Determine the maximum body torque the wheel array can deliver along
    # this axis.  Apply a unit body torque along the rotation axis, compute
    # the wheel torque allocation, then rescale so no wheel exceeds its limit.
    Gs_pinv = np.linalg.pinv(Gs_matrix)
    unit_body_torque = axis
    rw_for_unit = -unit_body_torque @ Gs_pinv.T
    max_rw_needed = np.max(np.abs(rw_for_unit))

    # Scale factor keeps the worst case wheel within the torque budget
    torque_scale = max_rw_torque / max_rw_needed if max_rw_needed > 0 else max_rw_torque

    # Maximum achievable body torque along the rotation axis
    tau_body_max = torque_scale

    # Minimum duration for a bang bang maneuver: T_min = sqrt(4 * I * theta / tau_max)
    T_min = np.sqrt(4 * I_axis * abs(theta_final) / tau_body_max)

    return T_min


def create_feedforward_torque_profile(theta_final, rotation_axis, inertia, 
                                       Gs_matrix, maneuver_duration,
                                       shaper_amplitudes=None, shaper_times=None,
                                       dt=DEFAULT_SAMPLE_DT, trajectory_type='bang-bang',
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
    - trajectory_type: 'bang-bang', 'trapezoidal', 'smooth', or 's_curve'.
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

    Pre computes a complete torque profile for a requested slew maneuver
    and stores spline interpolators so that individual wheel torque
    commands can be queried at arbitrary simulation times.

    Attributes:
        inertia: spacecraft inertia matrix (3x3) in [kg*m^2].
        Gs_matrix: reaction wheel spin axis matrix (3 x n_wheels).
        max_torque: per wheel torque limit in [Nm].
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

        # These are populated by design_maneuver or load_fourth_order_trajectory
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
        - trajectory_type: 'bang-bang', 'trapezoidal', 'smooth', or 's_curve'.
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
            dt=DEFAULT_SAMPLE_DT,
            trajectory_type=trajectory_type,
            max_torque=self.max_torque
        )
        
        self.t_profile = t
        self.rw_torque_profile = rw_torque
        self.trajectory = traj
        self.is_shaped = shaper_amplitudes is not None
        
        # Build per wheel linear interpolators so get_torque() can return
        # the torque command at any arbitrary time during the simulation.
        n_wheels = rw_torque.shape[1]
        self.torque_interp = []
        for i in range(n_wheels):
            interp = interpolate.interp1d(
                t, rw_torque[:, i],
                kind='linear',
                bounds_error=False,
                fill_value=0.0  # zero torque outside the profile window
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
            n_wheels = self.Gs_matrix.shape[1]
            return np.zeros(n_wheels)
        
        torque = np.array([interp(t) for interp in self.torque_interp])
        return torque
    
    def load_fourth_order_trajectory(self, trajectory_file, rotation_axis):
        """
        Load a precomputed fourth order trajectory from an NPZ file.

        The file must contain arrays named 'time', 'theta', 'omega', and
        'alpha'.  If the stored sample rate differs from DEFAULT_SAMPLE_DT
        the data is resampled via linear interpolation.

        Args:
            trajectory_file: path to the .npz file.
            rotation_axis: unit rotation axis for torque direction (3,).
        """
        # Load trajectory
        traj_data = np.load(trajectory_file, allow_pickle=True)
        
        t = np.array(traj_data['time'], dtype=float)
        theta = np.array(traj_data['theta'], dtype=float)
        omega = np.array(traj_data['omega'], dtype=float)
        alpha = np.array(traj_data['alpha'], dtype=float)

        if len(t) > 1:
            dt = float(np.median(np.diff(t)))
            if np.isfinite(dt) and abs(dt - DEFAULT_SAMPLE_DT) > 1e-9:
                t_new = np.arange(t[0], t[-1] + DEFAULT_SAMPLE_DT * 0.5, DEFAULT_SAMPLE_DT)
                theta = np.interp(t_new, t, theta)
                omega = np.interp(t_new, t, omega)
                alpha = np.interp(t_new, t, alpha)
                t = t_new
                print(f"Warning: trajectory dt={dt:.6f}s resampled to {DEFAULT_SAMPLE_DT:.2f}s")
        
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
    # Flexible modes are modeled separately; use hub inertia for rigid body slews.
    inertia = np.diag([900.0, 800.0, 600.0])  # kg*m^2 [I_xx, I_yy, I_zz]
    # Gs matrix: columns are wheel spin axes
    Gs_matrix = np.array([
        [np.sqrt(2)/2, -np.sqrt(2)/2, 0.0],
        [np.sqrt(2)/2,  np.sqrt(2)/2, 0.0],
        [0.0,           0.0,          1.0]
    ])
    
    # Test 180 degree YAW (Z axis) matches spacecraft_model.py configuration
    theta = np.radians(180.0)
    axis = np.array([0.0, 0.0, 1.0])  # Z axis (YAW) for strong flex coupling
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
