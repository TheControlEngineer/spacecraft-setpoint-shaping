"""
Vizard 3D Visualization Demo - Comet Photography Mission.

This version uses the FlexibleSpacecraft class directly, matching the setup
in test_gravity_effect.py and check_rotation.py which both achieve ~179.7 deg.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import numpy as np
from pathlib import Path
from typing import Optional

# Add src directory to path for basilisk_sim imports
_script_dir = Path(__file__).parent.resolve()
_src_dir = _script_dir.parent / "src"
if str(_src_dir) not in sys.path:
    sys.path.insert(0, str(_src_dir))

from Basilisk.utilities import SimulationBaseClass
from Basilisk.utilities import macros
from Basilisk.utilities import vizSupport
from Basilisk.utilities import orbitalMotion
from Basilisk.utilities import simIncludeGravBody
from Basilisk.architecture import messaging
from Basilisk.simulation import spacecraft
from Basilisk.architecture import swig_common_model
from Basilisk.simulation import vizInterface
from Basilisk.fswAlgorithms import mrpFeedback
from Basilisk.fswAlgorithms import inertial3D
from Basilisk.fswAlgorithms import attTrackingError

from basilisk_sim.spacecraft_model import FlexibleSpacecraft
from basilisk_sim.spacecraft_properties import compute_modal_gains
from basilisk_sim.feedforward_control import FeedforwardController, body_torque_to_rw_torque
from basilisk_sim.feedback_control import (
    FilteredDerivativeController,
    NotchFilterController,
    TrajectoryTrackingController,
    _mrp_shadow,
    _mrp_subtract
)


CONTROLLERS = {"standard_pd", "filtered_pd", "notch", "trajectory_tracking"}
RUN_MODES = {"combined", "fb_only", "ff_only"}


def _skew(vec: np.ndarray) -> np.ndarray:
    """Return the skew-symmetric matrix for a 3-vector."""
    x, y, z = vec
    return np.array([
        [0.0, -z, y],
        [z, 0.0, -x],
        [-y, x, 0.0],
    ])


def _mrp_to_dcm(sigma: np.ndarray) -> np.ndarray:
    """Convert MRP to DCM (C_BN maps inertial to body)."""
    sigma = np.array(sigma, dtype=float).reshape(3)
    s2 = float(np.dot(sigma, sigma))
    s_tilde = _skew(sigma)
    denom = (1.0 + s2) ** 2
    return np.eye(3) + (8.0 * (s_tilde @ s_tilde) - 4.0 * (1.0 - s2) * s_tilde) / denom


class CometPhotographyDemo:
    """Demonstrates comet photography with input shaping - uses FlexibleSpacecraft class."""

    def __init__(self, shaping_method='fourth', controller='standard_pd', run_mode='combined',
                 use_trajectory_tracking=True, jitter_lever_arm=4.0, pixel_scale_arcsec=2.0,
                 config_overrides: Optional[dict] = None, output_dir: Optional[str] = None):
        """
        Initialize the demo.

        Args:
            shaping_method: 'fourth' or 'unshaped'
            controller: 'standard_pd', 'filtered_pd', 'notch', or 'trajectory_tracking'
            run_mode: 'combined', 'fb_only', or 'ff_only'
            use_trajectory_tracking: If True, feedback tracks instantaneous FF trajectory
                                     (RECOMMENDED - prevents feedback from fighting FF)
            jitter_lever_arm: Lever arm from rotation axis to camera [m]
            pixel_scale_arcsec: Camera plate scale [arcsec/pixel]
            output_dir: Directory for output files (default: output/cache relative to script)
        """
        self.output_dir = output_dir
        if controller not in CONTROLLERS:
            raise ValueError(f"Unknown controller '{controller}'")
        if run_mode not in RUN_MODES:
            raise ValueError(f"Unknown run mode '{run_mode}'")
        self.method = shaping_method
        self.controller = controller
        self.run_mode = run_mode
        self.use_trajectory_tracking = use_trajectory_tracking or (controller == "trajectory_tracking")
        self.dt = 0.01
        self.slew_duration = 30.0
        self.settling_window = 60.0  # Increased from 30s for better convergence
        self.slew_angle_deg = 180.0
        self.slew_angle = np.radians(self.slew_angle_deg)
        self.camera_body = np.array([0.0, -1.0, 0.0])
        self.comet_direction = None
        self.jitter_lever_arm = float(jitter_lever_arm)
        self.pixel_scale_arcsec = float(pixel_scale_arcsec)

        # Create spacecraft model (SAME CLASS as check_rotation.py)
        self.sc = FlexibleSpacecraft()

        # Feedback controller tuning targets
        # Use bandwidth at first_mode/2.5 to preserve margin and avoid mode excitation
        first_mode_hz = self.sc.array_modes[0]['frequency']
        self.control_bandwidth_hz = first_mode_hz / 2.5
        self.control_damping_ratio = 0.90
        self.mrp_K = None
        self.mrp_P = None
        self.mrp_Ki = -1.0

        # Feedback controller options
        self.use_filtered_pd = controller == "filtered_pd"
        self.use_notch = controller == "notch"
        # Filter cutoff: high enough to preserve phase margin with low bandwidth
        self.filter_cutoff_hz = None

        # Optional injection parameters (for V&V / MC)
        self.sensor_noise_std_rad_s = 0.0
        self.sensor_noise_type = "white"
        self.sensor_noise_frequency_hz = 0.0
        self.sensor_noise_axis = np.array([0.0, 0.0, 1.0])
        self.disturbance_torque_nm = 0.0
        self.disturbance_type = "bias"
        self.disturbance_amplitude_nm = 0.0
        self.disturbance_frequency_hz = 0.0
        self.disturbance_axis = np.array([0.0, 0.0, 1.0])
        self.modal_gains_scale = 1.0

        # Apply optional overrides (MC/validation support)
        if config_overrides:
            self._apply_config_overrides(config_overrides)

        # Target attitude for a +Z yaw slew
        self.initial_sigma = [0.0, 0.0, 0.0]
        self.target_sigma = [0.0, 0.0, float(np.tan(self.slew_angle / 4.0))]

        # Data storage
        self.time_log = []
        self.sigma_log = []
        self.omega_log = []
        self.modal_log = {
            'mode1_port': [],
            'mode2_port': [],
            'mode1_stbd': [],
            'mode2_stbd': [],
            'mode1_port_acc': [],
            'mode2_port_acc': [],
            'mode1_stbd_acc': [],
            'mode2_stbd_acc': [],
            'time': [],
        }
        self.control_mode_log = []
        self.ff_torque_log = []
        self.fb_torque_log = []
        self.total_torque_log = []
        self.rw_torque_log = []
        self.camera_error_log = []
        self.rng = np.random.default_rng(42)
        self.hud_storage = []
        self.hud_sensor = None
        self.vib_trace = []

    def _apply_config_overrides(self, overrides: dict) -> None:
        """Apply configuration overrides for MC/validation runs."""
        if not isinstance(overrides, dict):
            return

        # Trajectory overrides
        if "slew_angle_deg" in overrides:
            self.slew_angle_deg = float(overrides["slew_angle_deg"])
            self.slew_angle = np.radians(self.slew_angle_deg)
        if "slew_duration_s" in overrides:
            self.slew_duration = float(overrides["slew_duration_s"])

        # Control overrides
        if "control_filter_cutoff_hz" in overrides:
            self.filter_cutoff_hz = float(overrides["control_filter_cutoff_hz"])

        # Disturbance/noise overrides
        if "sensor_noise_std_rad_s" in overrides:
            self.sensor_noise_std_rad_s = float(overrides["sensor_noise_std_rad_s"])
        if "sensor_noise_type" in overrides:
            self.sensor_noise_type = str(overrides["sensor_noise_type"]).lower()
        if "sensor_noise_frequency_hz" in overrides:
            try:
                self.sensor_noise_frequency_hz = float(overrides["sensor_noise_frequency_hz"])
            except (TypeError, ValueError):
                self.sensor_noise_frequency_hz = 0.0
        if "sensor_noise_axis" in overrides:
            axis = overrides["sensor_noise_axis"]
            if isinstance(axis, (list, tuple, np.ndarray)) and len(axis) == 3:
                axis_vec = np.array(axis, dtype=float)
                norm = np.linalg.norm(axis_vec)
                if norm > 0:
                    self.sensor_noise_axis = axis_vec / norm
        if "disturbance_torque_nm" in overrides:
            dist = overrides["disturbance_torque_nm"]
            if isinstance(dist, (list, tuple, np.ndarray)) and len(dist) == 3:
                self.disturbance_torque_nm = np.array(dist, dtype=float)
            else:
                self.disturbance_torque_nm = float(dist)
        if "disturbance_type" in overrides:
            self.disturbance_type = str(overrides["disturbance_type"]).lower()
        if "disturbance_amplitude_nm" in overrides:
            try:
                self.disturbance_amplitude_nm = float(overrides["disturbance_amplitude_nm"])
            except (TypeError, ValueError):
                self.disturbance_amplitude_nm = 0.0
        if "disturbance_frequency_hz" in overrides:
            try:
                self.disturbance_frequency_hz = float(overrides["disturbance_frequency_hz"])
            except (TypeError, ValueError):
                self.disturbance_frequency_hz = 0.0
        if "disturbance_axis" in overrides:
            axis = overrides["disturbance_axis"]
            if isinstance(axis, (list, tuple, np.ndarray)) and len(axis) == 3:
                axis_vec = np.array(axis, dtype=float)
                norm = np.linalg.norm(axis_vec)
                if norm > 0:
                    self.disturbance_axis = axis_vec / norm

        # Spacecraft / flexible mode overrides
        inertia_scale = overrides.get("inertia_scale")
        if inertia_scale is not None:
            try:
                scale = float(inertia_scale)
                self.sc.hub_inertia = (np.array(self.sc.hub_inertia, dtype=float) * scale).tolist()
            except (TypeError, ValueError):
                pass

        if "rw_max_torque_nm" in overrides:
            try:
                self.sc.rw_max_torque = float(overrides["rw_max_torque_nm"])
            except (TypeError, ValueError):
                pass

        if "modal_mass_kg" in overrides:
            try:
                self.sc.modal_mass = float(overrides["modal_mass_kg"])
            except (TypeError, ValueError):
                pass

        if "modal_freqs_hz" in overrides or "modal_damping" in overrides:
            freqs = overrides.get("modal_freqs_hz")
            damping = overrides.get("modal_damping")
            if freqs is not None:
                try:
                    freqs = [float(f) for f in freqs]
                except Exception:
                    freqs = None
            if damping is not None:
                try:
                    damping = [float(d) for d in damping]
                except Exception:
                    damping = None
            if freqs is not None and damping is not None and len(freqs) == len(damping) == len(self.sc.array_modes):
                for i, mode in enumerate(self.sc.array_modes):
                    mode["frequency"] = freqs[i]
                    mode["damping"] = damping[i]
                if freqs:
                    self.control_bandwidth_hz = min(freqs) / 2.5

        if "modal_gains_scale" in overrides:
            try:
                self.modal_gains_scale = float(overrides["modal_gains_scale"])
            except (TypeError, ValueError):
                self.modal_gains_scale = 1.0

        # Update target sigma after overrides
        self.target_sigma = [0.0, 0.0, float(np.tan(self.slew_angle / 4.0))]

    def build_simulation(self):
        """Build simulation using FlexibleSpacecraft class."""
        self.scSim = SimulationBaseClass.SimBaseClass()

        simulationTimeStep = macros.sec2nano(self.dt)
        dynProcess = self.scSim.CreateNewProcess("DynamicsProcess")
        dynTask = self.scSim.CreateNewTask("simTask", simulationTimeStep)
        dynProcess.addTask(dynTask)

        fswProcess = self.scSim.CreateNewProcess("FSWProcess")
        fswTask = self.scSim.CreateNewTask("fswTask", simulationTimeStep)
        fswProcess.addTask(fswTask)

        # =====================================================
        # USE FlexibleSpacecraft CLASS (same as check_rotation.py)
        # =====================================================
        print("\nCreating spacecraft using FlexibleSpacecraft class...")
        scObject = self.sc.create_rigid_spacecraft()
        rwStateEffector = self.sc.add_reaction_wheels()
        flexModes = self.sc.add_flexible_solar_arrays()
        navObject = self.sc.add_simple_nav()

        # Add gravity
        gravFactory = simIncludeGravBody.gravBodyFactory()
        mars = gravFactory.createMars()
        mars.isCentralBody = True
        mu = mars.mu
        gravFactory.addBodiesTo(scObject)
        print("  Gravity: ENABLED (Mars)")

        # Orbit (Mars radius ~3396 km)
        MARS_RADIUS = 3396200.0  # meters
        oe = orbitalMotion.ClassicElements()
        oe.a = MARS_RADIUS + 500000.0
        oe.e = 0.0001
        oe.i = 45.0 * macros.D2R
        oe.Omega = 0.0
        oe.omega = 0.0
        oe.f = 0.0
        rN, vN = orbitalMotion.elem2rv(mu, oe)
        rN = np.array(rN)
        vN = np.array(vN)

        scObject.hub.r_CN_NInit = rN
        scObject.hub.v_CN_NInit = vN
        print(f"  Orbit altitude: {(np.linalg.norm(rN) - MARS_RADIUS)/1000:.1f} km")

        # Initial attitude: ZERO
        scObject.hub.sigma_BNInit = [[0.0], [0.0], [0.0]]
        scObject.hub.omega_BN_BInit = [[0.0], [0.0], [0.0]]

        print(f"  Initial sigma: {self.initial_sigma}")
        print(f"  Target sigma: {self.target_sigma} (ideal 180 deg yaw)")

        # =====================================================
        # COMET - placed where camera will point after feedforward slew
        # =====================================================
        # After the commanded yaw, the camera (-Y body) points to this inertial direction.
        cometObject = spacecraft.Spacecraft()
        cometObject.ModelTag = "Comet67P"
        cometObject.hub.mHub = 1.0
        cometObject.hub.IHubPntBc_B = [[0.1, 0, 0], [0, 0.1, 0], [0, 0, 0.1]]

        comet_offset = 50000.0
        camera_body = self.camera_body
        c_yaw = float(np.cos(self.slew_angle))
        s_yaw = float(np.sin(self.slew_angle))
        rot_z = np.array([[c_yaw, -s_yaw, 0.0],
                          [s_yaw,  c_yaw, 0.0],
                          [0.0,    0.0,   1.0]])
        comet_direction = rot_z @ camera_body
        comet_direction = comet_direction / np.linalg.norm(comet_direction)
        self.comet_direction = comet_direction.copy()
        
        cometObject.hub.r_CN_NInit = rN + comet_offset * comet_direction
        cometObject.hub.v_CN_NInit = vN.copy()
        cometObject.hub.sigma_BNInit = [[0.0], [0.0], [0.0]]
        gravFactory.addBodiesTo(cometObject)
        self.cometObject = cometObject
        print(f"  Comet: {comet_offset/1000:.1f} km in camera target direction")

        # =====================================================
        # FEEDFORWARD CONTROLLER (same as check_rotation.py)
        # =====================================================
        hub_inertia = np.array(self.sc.hub_inertia)
        ff_inertia = self.sc.compute_effective_inertia(include_flex=True)
        self.inertia_for_control = ff_inertia
        self.inertia_feedforward = ff_inertia

        if self.mrp_K is None or self.mrp_P is None:
            I_control = self.inertia_for_control[2, 2]
            omega_n = 2 * np.pi * self.control_bandwidth_hz
            sigma_scale = 4.0
            self.mrp_K = sigma_scale * omega_n**2 * I_control
            self.mrp_P = 2 * self.control_damping_ratio * I_control * omega_n

        if self.use_filtered_pd:
            # Retune filtered PD damping to offset filter attenuation near modes.
            self.mrp_P *= 1.5

        if self.filter_cutoff_hz is None:
            base_cutoff = max(2.0, 5.0 * self.control_bandwidth_hz)
            if self.use_filtered_pd:
                self.filter_cutoff_hz = 8.0
            else:
                self.filter_cutoff_hz = base_cutoff

        print(f"  Control gains: K={self.mrp_K:.1f}, P={self.mrp_P:.1f}")
        print(f"  Target bandwidth: {self.control_bandwidth_hz:.2f} Hz, filter cutoff: {self.filter_cutoff_hz:.2f} Hz")

        self.Gs_matrix = np.array([
            [np.sqrt(2)/2, -np.sqrt(2)/2, 0.0],
            [np.sqrt(2)/2,  np.sqrt(2)/2, 0.0],
            [0.0,           0.0,          1.0]
        ])

        rotation_axis = np.array([0.0, 0.0, 1.0])
        self.modal_gains = compute_modal_gains(self.inertia_for_control, rotation_axis)
        if not self.modal_gains:
            self.modal_gains = [0.0] * len(self.sc.array_modes)
        if self.modal_gains_scale != 1.0:
            self.modal_gains = [float(g) * self.modal_gains_scale for g in self.modal_gains]

        self.ff_controller = FeedforwardController(
            ff_inertia, self.Gs_matrix, max_torque=self.sc.rw_max_torque
        )

        if self.method == 'fourth':
            traj_candidates = [
                os.path.join(os.path.dirname(__file__), "..", "data", "trajectories", "spacecraft_trajectory_4th_180deg_30s.npz"),
                os.path.join(os.path.dirname(__file__), "spacecraft_trajectory_4th_180deg_30s.npz"),
                os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "spacecraft_trajectory_4th_180deg_30s.npz")),
                "spacecraft_trajectory_4th_180deg_30s.npz",
            ]
            traj_path = next((path for path in traj_candidates if os.path.isfile(path)), None)
            if traj_path is None:
                raise FileNotFoundError(
                    "Fourth-order trajectory file not found. Expected "
                    "`spacecraft_trajectory_4th_180deg_30s.npz` in data/trajectories/."
                )
            self.ff_controller.load_fourth_order_trajectory(
                traj_path,
                rotation_axis
            )
        else:
            self.ff_controller.design_maneuver(
                theta_final=self.slew_angle,
                rotation_axis=rotation_axis,
                duration=self.slew_duration,
                trajectory_type='bang-bang'
            )

        self.actual_duration = self.ff_controller.t_profile[-1]
        print(f"  Feedforward: {self.method}, duration {self.actual_duration:.1f}s")

        # =====================================================
        # FEEDBACK CONTROLLER (for post-slew fine pointing)
        # =====================================================
        inertial3DObj = inertial3D.inertial3D()
        inertial3DObj.ModelTag = "inertial3D"
        inertial3DObj.sigma_R0N = self.target_sigma

        attErrorObj = attTrackingError.attTrackingError()
        attErrorObj.ModelTag = "attError"
        attErrorObj.attNavInMsg.subscribeTo(navObject.attOutMsg)
        attErrorObj.attRefInMsg.subscribeTo(inertial3DObj.attRefOutMsg)

        mrpControlObj = mrpFeedback.mrpFeedback()
        mrpControlObj.ModelTag = "mrpFeedback"
        mrpControlObj.guidInMsg.subscribeTo(attErrorObj.attGuidOutMsg)
        mrpControlObj.K = self.mrp_K
        mrpControlObj.P = self.mrp_P
        mrpControlObj.Ki = self.mrp_Ki

        vehicleConfigOut = messaging.VehicleConfigMsgPayload()
        vehicleConfigOut.ISCPntB_B = [
            self.inertia_for_control[0, 0], 0, 0,
            0, self.inertia_for_control[1, 1], 0,
            0, 0, self.inertia_for_control[2, 2]
        ]
        vcMsg = messaging.VehicleConfigMsg().write(vehicleConfigOut)
        mrpControlObj.vehConfigInMsg.subscribeTo(vcMsg)

        self.inertial3DObj = inertial3DObj
        self.attErrorObj = attErrorObj
        self.mrpControlObj = mrpControlObj

        # =====================================================
        # FEEDBACK CONTROLLER SETUP
        # =====================================================
        modal_freqs = [mode['frequency'] for mode in self.sc.array_modes]
        modal_damping = [mode['damping'] for mode in self.sc.array_modes]
        rotation_axis = np.array([0.0, 0.0, 1.0])

        # Determine which controller type to use
        if self.use_trajectory_tracking:
            # RECOMMENDED: Use trajectory tracking controller
            # This tracks the instantaneous FF reference, not the final target
            # Prevents feedback from fighting the feedforward trajectory
            if self.use_filtered_pd:
                ctrl_type = 'filtered_pd'
            elif self.use_notch:
                ctrl_type = 'notch'
            else:
                ctrl_type = 'standard_pd'

            self.trajectory_controller = TrajectoryTrackingController(
                inertia=self.inertia_for_control,
                K=self.mrp_K,
                P=self.mrp_P,
                controller_type=ctrl_type,
                filter_freq_hz=self.filter_cutoff_hz,
                notch_freqs_hz=modal_freqs,
            )
            # Set the feedforward trajectory for tracking
            self.trajectory_controller.set_feedforward_trajectory(
                self.ff_controller, rotation_axis
            )
            self.trajectory_controller.set_final_target(np.array(self.target_sigma))
            print(f"  Trajectory Tracking Controller ({ctrl_type})")
            print(f"    Tracks instantaneous FF reference during slew")
            print(f"    Filter cutoff: {self.filter_cutoff_hz:.2f} Hz")

        elif self.use_filtered_pd:
            self.filtered_pd = FilteredDerivativeController(
                inertia=self.inertia_for_control,
                K=self.mrp_K,
                P=self.mrp_P,
                filter_freq_hz=self.filter_cutoff_hz
            )
            self.filtered_pd.set_target(np.array(self.target_sigma))
            print(f"  Filtered PD: cutoff={self.filter_cutoff_hz:.2f} Hz (tuned for phase margin)")

        elif self.use_notch:
            self.notch_controller = NotchFilterController(
                inertia=self.inertia_for_control,
                K=self.mrp_K,
                P=self.mrp_P,
                notch_freqs_hz=modal_freqs,
                notch_depth_db=20.0,
                notch_width=0.3
            )
            self.notch_controller.set_target(np.array(self.target_sigma))
            print(f"  Notch Filter: frequencies={modal_freqs} Hz")

        feedback_label = "Trajectory Tracking" if self.use_trajectory_tracking else (
            "Filtered PD" if self.use_filtered_pd else (
                "Notch" if self.use_notch else "MRP Feedback"
            )
        )
        print(f"  Control: FF+{feedback_label} combined continuously (0-{self.actual_duration:.0f}s), then {feedback_label} only")

        # =====================================================
        # Direct RW torque command (bypass rwMotorTorque mapping)
        # =====================================================
        self.rwMotorCmdMsg = messaging.ArrayMotorTorqueMsg()
        rwStateEffector.rwMotorCmdInMsg.subscribeTo(self.rwMotorCmdMsg)

        # =====================================================
        # ADD TO SIMULATION (same order as check_rotation.py)
        # =====================================================
        self.scSim.AddModelToTask("simTask", scObject)
        self.scSim.AddModelToTask("simTask", rwStateEffector, ModelPriority=987)
        for mode_name, mode_obj in flexModes.items():
            self.scSim.AddModelToTask("simTask", mode_obj)
        self.scSim.AddModelToTask("simTask", navObject)
        self.scSim.AddModelToTask("simTask", cometObject)

        self.scSim.AddModelToTask("fswTask", inertial3DObj)
        self.scSim.AddModelToTask("fswTask", attErrorObj)
        self.scSim.AddModelToTask("fswTask", mrpControlObj)
        # No rwMotorTorque module needed when commanding wheel torques directly.

        navObject.scStateInMsg.subscribeTo(scObject.scStateOutMsg)

        self.scObject = scObject
        self.rwStateEffector = rwStateEffector
        self.flexModes = flexModes
        self.navObject = navObject

        self._setup_vizard()
        self.scSim.InitializeSimulation()

        # Get state objects for modal logging
        self.mode1_port_rho = scObject.dynManager.getStateObject(flexModes['mode1_port'].nameOfRhoState)
        self.mode2_port_rho = scObject.dynManager.getStateObject(flexModes['mode2_port'].nameOfRhoState)
        self.mode1_stbd_rho = scObject.dynManager.getStateObject(flexModes['mode1_stbd'].nameOfRhoState)
        self.mode2_stbd_rho = scObject.dynManager.getStateObject(flexModes['mode2_stbd'].nameOfRhoState)
        self.mode1_port_rho_dot = scObject.dynManager.getStateObject(flexModes['mode1_port'].nameOfRhoDotState)
        self.mode2_port_rho_dot = scObject.dynManager.getStateObject(flexModes['mode2_port'].nameOfRhoDotState)
        self.mode1_stbd_rho_dot = scObject.dynManager.getStateObject(flexModes['mode1_stbd'].nameOfRhoDotState)
        self.mode2_stbd_rho_dot = scObject.dynManager.getStateObject(flexModes['mode2_stbd'].nameOfRhoDotState)

        return self.scSim

    def _setup_vizard(self):
        """Setup Vizard visualization."""
        def _make_storage(label: str, units: str, color, max_value: float):
            storage = vizInterface.GenericStorage()
            storage.label = label
            storage.units = units
            color_vec = swig_common_model.IntVector()
            if isinstance(color, str):
                rgba = vizSupport.toRGBA255(color, alpha=1.0)
            else:
                rgba = color
            for val in rgba:
                color_vec.append(int(val))
            storage.color = color_vec
            storage.maxValue = float(max_value)
            storage.currentValue = 0.0
            storage.type = "generic"
            return storage

        self.hud_storage = [
            _make_storage("Vibration", "", [255, 180, 0, 255], 50.0),
            _make_storage("Acceleration", "", [255, 80, 80, 255], 500.0),
            _make_storage("Pointing Err", "", [80, 200, 255, 255], 180.0),
            _make_storage("Rate", "", [120, 255, 120, 255], 10.0),
        ]

        viz = vizSupport.enableUnityVisualization(
            self.scSim,
            "simTask",
            [self.scObject, self.cometObject],
            saveFile=f"viz_demo_{self.method}",
            genericStorageList=[self.hud_storage, None],
        )

        viz.settings.spacecraftSizeMultiplier = 2.0
        viz.settings.showSpacecraftLabels = 1
        viz.settings.orbitLinesOn = 1
        viz.settings.spacecraftCSon = 1
        viz.settings.showCSLabels = 1
        viz.settings.planetCSon = 1
        viz.settings.showHillFrame = 1
        viz.settings.linesAndFramesLineWidth = 3.0
        viz.settings.viewCameraBoresightHUD = 1
        vizSupport.setInstrumentGuiSetting(
            viz,
            spacecraftName=self.scObject.ModelTag,
            showGenericStoragePanel=1,
        )

        # Comet with tail - single elongated ellipsoid shape
        # Stretched heavily along Y axis to create tail effect
        vizSupport.createCustomModel(viz,
            modelPath="SPHERE",
            simBodiesToModify=[self.cometObject.ModelTag],
            color=[200, 230, 255, 200],  # Bright cyan-white
            scale=[50.0, 250.0, 50.0],  # Stretched along Y for tail-like shape
            shader=1
        )

        # Camera FOV cone (soft, elegant style)
        vizSupport.createConeInOut(viz,
            toBodyName=self.cometObject.ModelTag,
            fromBodyName=self.scObject.ModelTag,
            normalVector_B=[0, -1, 0],
            incidenceAngle=np.radians(5),
            isKeepIn=True,
            coneColor=[100, 200, 255, 40],
            coneHeight=80.0,
            coneName="Camera FOV"
        )

        # Cameras
        vizSupport.createStandardCamera(viz,
            spacecraftName=self.scObject.ModelTag,
            setMode=1,
            pointingVector_B=[0, -1, 0],
            position_B=[0, 50, 15],
            displayName="Chase Cam"
        )

        vizSupport.createStandardCamera(viz,
            spacecraftName=self.scObject.ModelTag,
            setMode=1,
            pointingVector_B=[0, -1, 0],
            position_B=[0, -2, 0],
            fieldOfView=np.radians(0.5),
            displayName="Telephoto"
        )

        vizSupport.createStandardCamera(viz,
            spacecraftName=self.scObject.ModelTag,
            setMode=1,
            pointingVector_B=[0, -1, 0],
            position_B=[0, -2, 0],
            fieldOfView=np.radians(10),
            displayName="Wide"
        )

        self.viz = viz
        print("  Vizard: configured")

    def run(self):
        """Execute simulation."""
        print(f"\n{'='*60}")
        print(f"Running {self.method.upper()}")
        print(f"{'='*60}")

        total_sim_time = self.actual_duration + self.settling_window
        n_steps = int(total_sim_time / self.dt)

        rw_cmd_payload = messaging.ArrayMotorTorqueMsgPayload()
        switched_to_feedback = False

        def compute_feedback_torque(sim_time: float):
            sigma_current = np.array(self.scObject.scStateOutMsg.read().sigma_BN)
            omega_current = np.array(self.scObject.scStateOutMsg.read().omega_BN_B)
            if self.sensor_noise_type == "sine" and self.sensor_noise_std_rad_s > 0:
                omega_current = omega_current + (
                    self.sensor_noise_axis
                    * self.sensor_noise_std_rad_s
                    * np.sin(2.0 * np.pi * self.sensor_noise_frequency_hz * sim_time)
                )
            elif self.sensor_noise_std_rad_s > 0:
                omega_current = omega_current + self.rng.normal(
                    0.0, self.sensor_noise_std_rad_s, size=omega_current.shape
                )

            # Get modal displacements for logging
            mode1 = 0.5 * (self.mode1_port_rho.getState()[0][0] - self.mode1_stbd_rho.getState()[0][0])
            mode2 = 0.5 * (self.mode2_port_rho.getState()[0][0] - self.mode2_stbd_rho.getState()[0][0])

            # RECOMMENDED: Use trajectory tracking controller
            # This tracks the instantaneous FF reference, preventing feedback from fighting FF
            if self.use_trajectory_tracking:
                torque_cmd = self.trajectory_controller.compute_torque(
                    sigma_current, omega_current, sim_time
                )
                return torque_cmd, "TRK"

            # Legacy controllers (track final target, not trajectory)
            if self.use_filtered_pd:
                torque_cmd = self.filtered_pd.compute_torque(sigma_current, omega_current, sim_time)
                return torque_cmd, "FB(FILT)"

            if self.use_notch:
                torque_cmd = self.notch_controller.compute_torque(sigma_current, omega_current, sim_time)
                return torque_cmd, "FB(NOTCH)"

            fb_torque = self.mrpControlObj.cmdTorqueOutMsg.read().torqueRequestBody
            return np.array(fb_torque), "FB(MRP)"

        for step in range(n_steps):
            current_time = step * self.dt

            # CORRECTED CONTROL ARCHITECTURE:
            # Feedforward and feedback work TOGETHER continuously
            # FF provides the trajectory, FB tracks it and rejects disturbances
            
            fb_torque = np.zeros(3)
            fb_label = "FB(OFF)"
            if self.run_mode != "ff_only":
                fb_torque, fb_label = compute_feedback_torque(current_time)

            ff_torque = np.zeros(3)
            control_mode = fb_label
            if self.run_mode != "fb_only":
                if current_time <= self.actual_duration:
                    rw_torques = self.ff_controller.get_torque(current_time)
                    ff_torque = self.Gs_matrix @ (-rw_torques)
                    ff_magnitude = float(np.linalg.norm(ff_torque))
                    if self.run_mode == "combined":
                        if ff_magnitude > 0.01:
                            control_mode = f"FF+{fb_label}"
                        else:
                            control_mode = f"FF(0)+{fb_label}"
                    else:
                        control_mode = "FF" if ff_magnitude > 0.01 else "FF(0)"
                else:
                    ff_torque = np.zeros(3)
                    if self.run_mode == "combined":
                        control_mode = fb_label
                    else:
                        control_mode = "FF(0)"

            # COMBINED control: FF + FB (+ disturbance)
            body_torque = ff_torque + fb_torque
            if self.disturbance_type == "sine" and self.disturbance_amplitude_nm > 0:
                disturbance_vec = (
                    self.disturbance_axis
                    * self.disturbance_amplitude_nm
                    * np.sin(2.0 * np.pi * self.disturbance_frequency_hz * current_time)
                )
                body_torque = body_torque + disturbance_vec
            elif np.any(self.disturbance_torque_nm):
                if isinstance(self.disturbance_torque_nm, np.ndarray):
                    disturbance_vec = self.disturbance_torque_nm
                else:
                    disturbance_vec = np.array([0.0, 0.0, float(self.disturbance_torque_nm)])
                body_torque = body_torque + disturbance_vec
            rw_torque_cmd = body_torque_to_rw_torque(body_torque.reshape(1, 3), self.Gs_matrix)[0]

            if self.run_mode == "combined" and not switched_to_feedback and current_time > self.actual_duration:
                print(f"  t={current_time:.1f}s: Feedforward trajectory complete (180 deg slew finished)")
                print(f"           Continuing with {fb_label} for attitude maintenance")
                switched_to_feedback = True

            rw_torque_cmd = np.clip(rw_torque_cmd, -self.sc.rw_max_torque, self.sc.rw_max_torque)
            motor_torque = rw_cmd_payload.motorTorque
            motor_torque[0] = float(rw_torque_cmd[0])
            motor_torque[1] = float(rw_torque_cmd[1])
            motor_torque[2] = float(rw_torque_cmd[2])
            rw_cmd_payload.motorTorque = motor_torque
            self.rwMotorCmdMsg.write(rw_cmd_payload)

            self.scSim.ConfigureStopTime(macros.sec2nano((step + 1) * self.dt))
            self.scSim.ExecuteSimulation()

            # Logging
            self.time_log.append(current_time)
            self.sigma_log.append(self.scObject.scStateOutMsg.read().sigma_BN)
            self.omega_log.append(self.scObject.scStateOutMsg.read().omega_BN_B)
            self.control_mode_log.append(control_mode)
            self.ff_torque_log.append(ff_torque.copy())
            self.fb_torque_log.append(fb_torque.copy())
            self.total_torque_log.append(body_torque.copy())
            self.rw_torque_log.append(rw_torque_cmd.copy())

            sigma_current = self.sigma_log[-1]
            if self.comet_direction is not None:
                c_bn = _mrp_to_dcm(sigma_current)
                boresight_n = c_bn.T @ self.camera_body
                dot = float(np.clip(np.dot(boresight_n, self.comet_direction), -1.0, 1.0))
                pointing_error = float(np.degrees(np.arccos(dot)))
            else:
                pointing_error = 0.0
            self.camera_error_log.append(pointing_error)

            mode1_port = self.mode1_port_rho.getState()[0][0]
            mode2_port = self.mode2_port_rho.getState()[0][0]
            mode1_stbd = self.mode1_stbd_rho.getState()[0][0]
            mode2_stbd = self.mode2_stbd_rho.getState()[0][0]
            mode1_port_acc = self.mode1_port_rho_dot.getStateDeriv()[0][0]
            mode2_port_acc = self.mode2_port_rho_dot.getStateDeriv()[0][0]
            mode1_stbd_acc = self.mode1_stbd_rho_dot.getStateDeriv()[0][0]
            mode2_stbd_acc = self.mode2_stbd_rho_dot.getStateDeriv()[0][0]

            self.modal_log['time'].append(current_time)
            self.modal_log['mode1_port'].append(mode1_port)
            self.modal_log['mode2_port'].append(mode2_port)
            self.modal_log['mode1_stbd'].append(mode1_stbd)
            self.modal_log['mode2_stbd'].append(mode2_stbd)
            self.modal_log['mode1_port_acc'].append(mode1_port_acc)
            self.modal_log['mode2_port_acc'].append(mode2_port_acc)
            self.modal_log['mode1_stbd_acc'].append(mode1_stbd_acc)
            self.modal_log['mode2_stbd_acc'].append(mode2_stbd_acc)

            if self.hud_storage:
                mode1_signed = 0.5 * (mode1_port - mode1_stbd)
                mode2_signed = 0.5 * (mode2_port - mode2_stbd)
                mode1_acc_signed = 0.5 * (mode1_port_acc - mode1_stbd_acc)
                mode2_acc_signed = 0.5 * (mode2_port_acc - mode2_stbd_acc)
                vib_mm = np.sqrt(mode1_signed**2 + mode2_signed**2) * 1000.0
                acc_mm = np.sqrt(mode1_acc_signed**2 + mode2_acc_signed**2) * 1000.0
                omega_deg_s = float(np.linalg.norm(self.omega_log[-1]) * 180.0 / np.pi)
                # Protect against NaN values in HUD
                self.hud_storage[0].currentValue = float(vib_mm) if np.isfinite(vib_mm) else 0.0
                self.hud_storage[1].currentValue = float(acc_mm) if np.isfinite(acc_mm) else 0.0
                self.hud_storage[2].currentValue = float(pointing_error) if np.isfinite(pointing_error) else 0.0
                self.hud_storage[3].currentValue = float(omega_deg_s) if np.isfinite(omega_deg_s) else 0.0

                if self.hud_storage:
                    max_trace = 5.0
                    trace_len = 48
                    # Handle NaN values gracefully
                    vib_val = float(vib_mm) if np.isfinite(vib_mm) else 0.0
                    self.vib_trace.append(vib_val)
                    if len(self.vib_trace) > trace_len:
                        self.vib_trace = self.vib_trace[-trace_len:]
                    levels = " .:-=+*#%@"
                    trace_chars = []
                    for val in self.vib_trace:
                        if not np.isfinite(val):
                            val = 0.0
                        norm = min(max(val / max_trace, 0.0), 1.0)
                        idx = int(round(norm * (len(levels) - 1)))
                        trace_chars.append(levels[idx])
                    trace = "".join(trace_chars)

                    vib_display = vib_mm if np.isfinite(vib_mm) else 0.0
                    acc_display = acc_mm if np.isfinite(acc_mm) else 0.0
                    self.hud_storage[0].label = f"Vibration: {vib_display:6.2f} mm"
                    self.hud_storage[1].label = f"Acceleration: {acc_display:6.1f} mm/s²"
                    self.hud_storage[2].label = f"Pointing Error: {pointing_error:6.3f}°"
                    self.hud_storage[3].label = f"Angular Rate: {omega_deg_s:6.3f}°/s"

            if step % 500 == 0:
                print(f"  {current_time:.0f}s / {total_sim_time:.0f}s [{control_mode}]")

        print("Done!")
        self._analyze_results()
        self._save_results()

    def _analyze_results(self):
        """Analyze results."""
        sigma = np.array(self.sigma_log)
        time = np.array(self.time_log)

        sigma_final = sigma[-1]
        sigma_final_short = _mrp_shadow(sigma_final)
        sigma_mag = np.linalg.norm(sigma_final_short)
        rotation_achieved = np.degrees(4 * np.arctan(sigma_mag))

        sigma_target = np.array(self.target_sigma)
        if self.camera_error_log:
            pointing_error = float(self.camera_error_log[-1])
        else:
            sigma_error = _mrp_subtract(sigma_final_short, sigma_target)
            pointing_error = np.degrees(4 * np.arctan(np.linalg.norm(sigma_error)))

        print(f"\n  RESULTS:")
        print(f"    Rotation: {rotation_achieved:.1f} deg (target: 180 deg)")
        print(f"    Pointing error: {pointing_error:.2f} deg")
        print(f"    Final sigma: [{sigma_final[0]:.4f}, {sigma_final[1]:.4f}, {sigma_final[2]:.4f}]")
        print(f"    Target sigma: [{sigma_target[0]:.4f}, {sigma_target[1]:.4f}, {sigma_target[2]:.4f}]")

        # Modal vibration
        post_slew_idx = time >= self.actual_duration
        mode1 = np.sqrt(np.array(self.modal_log['mode1_port'])**2 + np.array(self.modal_log['mode1_stbd'])**2)
        mode2 = np.sqrt(np.array(self.modal_log['mode2_port'])**2 + np.array(self.modal_log['mode2_stbd'])**2)

        if np.any(post_slew_idx):
            mode1_rms = np.sqrt(np.mean(mode1[post_slew_idx]**2)) * 1000
            mode2_rms = np.sqrt(np.mean(mode2[post_slew_idx]**2)) * 1000
            total_rms = np.sqrt(mode1_rms**2 + mode2_rms**2)
            
            jitter_rad = (total_rms / 1000.0) / self.jitter_lever_arm
            jitter_arcsec = jitter_rad * (180 / np.pi) * 3600
            blur_px = jitter_arcsec / self.pixel_scale_arcsec

            print(f"\n  VIBRATION (imaging impact):")
            print(f"    Modal RMS: {total_rms:.2f} mm")
            print(f"    Jitter: {jitter_arcsec:.1f} arcsec (lever arm: {self.jitter_lever_arm:.1f} m)")
            print(f"    Blur: {blur_px:.1f} px (scale: {self.pixel_scale_arcsec:.1f} arcsec/px)")

    def _save_results(self):
        """Save results."""
        # Save to output directory (use provided dir or default to output/cache)
        if self.output_dir:
            cache_dir = self.output_dir
        else:
            cache_dir = os.path.join(os.path.dirname(__file__), "..", "output", "cache")
        os.makedirs(cache_dir, exist_ok=True)
        if self.run_mode == "combined":
            output = os.path.join(cache_dir, f'vizard_demo_{self.method}_{self.controller}.npz')
        elif self.run_mode == "fb_only":
            output = os.path.join(cache_dir, f'vizard_demo_{self.method}_{self.controller}_fb_only.npz')
        else:
            output = os.path.join(cache_dir, f'vizard_demo_{self.method}_ff_only.npz')
        mode1_port = np.array(self.modal_log['mode1_port'])
        mode1_stbd = np.array(self.modal_log['mode1_stbd'])
        mode2_port = np.array(self.modal_log['mode2_port'])
        mode2_stbd = np.array(self.modal_log['mode2_stbd'])
        mode1_port_acc = np.array(self.modal_log['mode1_port_acc'])
        mode1_stbd_acc = np.array(self.modal_log['mode1_stbd_acc'])
        mode2_port_acc = np.array(self.modal_log['mode2_port_acc'])
        mode2_stbd_acc = np.array(self.modal_log['mode2_stbd_acc'])

        mode1_signed = 0.5 * (mode1_port - mode1_stbd)
        mode2_signed = 0.5 * (mode2_port - mode2_stbd)
        mode1_acc_signed = 0.5 * (mode1_port_acc - mode1_stbd_acc)
        mode2_acc_signed = 0.5 * (mode2_port_acc - mode2_stbd_acc)

        mode1_rss = np.sqrt(mode1_port**2 + mode1_stbd**2)
        mode2_rss = np.sqrt(mode2_port**2 + mode2_stbd**2)
        mode1_acc_rss = np.sqrt(mode1_port_acc**2 + mode1_stbd_acc**2)
        mode2_acc_rss = np.sqrt(mode2_port_acc**2 + mode2_stbd_acc**2)

        np.savez(output,
                 time=np.array(self.time_log),
                 sigma=np.array(self.sigma_log),
                 omega=np.array(self.omega_log),
                 mode1=mode1_signed, mode2=mode2_signed,
                 mode1_signed=mode1_signed, mode2_signed=mode2_signed,
                 mode1_rss=mode1_rss, mode2_rss=mode2_rss,
                 mode1_acc=mode1_acc_signed, mode2_acc=mode2_acc_signed,
                 mode1_acc_signed=mode1_acc_signed, mode2_acc_signed=mode2_acc_signed,
                 mode1_acc_rss=mode1_acc_rss, mode2_acc_rss=mode2_acc_rss,
                 control_mode=self.control_mode_log,
                 method=self.method,
                 controller=self.controller,
                 run_mode=self.run_mode,
                 ff_torque=np.array(self.ff_torque_log),
                 fb_torque=np.array(self.fb_torque_log),
                 total_torque=np.array(self.total_torque_log),
                 rw_torque=np.array(self.rw_torque_log),
                 slew_angle_deg=self.slew_angle_deg,
                 slew_duration_s=self.slew_duration,
                 dt=self.dt,
                 initial_sigma=self.initial_sigma,
                 target_sigma=self.target_sigma,
                 camera_body=self.camera_body,
                 comet_direction=self.comet_direction,
                 camera_error_deg=np.array(self.camera_error_log),
                 modal_freqs_hz=[mode['frequency'] for mode in self.sc.array_modes],
                 modal_damping=[mode['damping'] for mode in self.sc.array_modes],
                 modal_gains=self.modal_gains,
                 control_bandwidth_hz=self.control_bandwidth_hz,
                 control_filter_cutoff_hz=self.filter_cutoff_hz,
                 control_damping_ratio=self.control_damping_ratio,
                 control_gains=np.array([self.mrp_K, self.mrp_P, self.mrp_Ki], dtype=float),
                 sensor_noise_std_rad_s=self.sensor_noise_std_rad_s,
                 sensor_noise_type=self.sensor_noise_type,
                 sensor_noise_frequency_hz=self.sensor_noise_frequency_hz,
                 sensor_noise_axis=self.sensor_noise_axis,
                 disturbance_torque_nm=self.disturbance_torque_nm,
                 disturbance_type=self.disturbance_type,
                 disturbance_amplitude_nm=self.disturbance_amplitude_nm,
                 disturbance_frequency_hz=self.disturbance_frequency_hz,
                 disturbance_axis=self.disturbance_axis,
                 inertia_control=self.inertia_for_control,
                 inertia_feedforward=self.inertia_feedforward,
                 use_trajectory_tracking=self.use_trajectory_tracking,
                 rotation_axis=np.array([0.0, 0.0, 1.0]))
        print(f"\nSaved {output}")


if __name__ == "__main__":
    print("=" * 60)
    print("Comet Photography Demo")
    print("Using FlexibleSpacecraft class (matches check_rotation.py)")
    print("=" * 60)

    parser = argparse.ArgumentParser(description="Run the comet photography demo.")
    parser.add_argument("method", nargs="?", default="fourth", choices=["unshaped", "fourth"])
    parser.add_argument("--controller", default="standard_pd", choices=sorted(CONTROLLERS))
    parser.add_argument("--mode", default="combined", choices=sorted(RUN_MODES))
    parser.add_argument("--lever-arm", type=float, default=4.0,
                        help="Camera lever arm in meters (default: 4.0)")
    parser.add_argument("--pixel-scale", type=float, default=2.0,
                        help="Camera plate scale in arcsec/pixel (default: 2.0)")
    parser.add_argument("--config", default=None,
                        help="Optional JSON config override file")
    parser.add_argument("--output-dir", default=None,
                        help="Output directory for NPZ files (default: output/cache)")
    args = parser.parse_args()

    print(f"\nMethod: {args.method}")
    print(f"Controller: {args.controller}")
    print(f"Run mode: {args.mode}")

    config_overrides = None
    if args.config:
        try:
            with open(args.config, "r", encoding="utf-8") as f:
                config_overrides = json.load(f)
        except (OSError, json.JSONDecodeError):
            config_overrides = None

    demo = CometPhotographyDemo(
        args.method,
        controller=args.controller,
        run_mode=args.mode,
        jitter_lever_arm=args.lever_arm,
        pixel_scale_arcsec=args.pixel_scale,
        config_overrides=config_overrides,
        output_dir=args.output_dir,
    )
    demo.build_simulation()
    demo.run()
