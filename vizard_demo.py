"""
Vizard 3D Visualization Demo - Comet Photography Mission.

This version uses the FlexibleSpacecraft class directly, matching the setup
in test_gravity_effect.py and check_rotation.py which both achieve ~179.7 deg.
"""

from __future__ import annotations

import argparse
import numpy as np

from Basilisk.utilities import SimulationBaseClass
from Basilisk.utilities import macros
from Basilisk.utilities import vizSupport
from Basilisk.utilities import orbitalMotion
from Basilisk.utilities import simIncludeGravBody
from Basilisk.architecture import messaging
from Basilisk.simulation import spacecraft
from Basilisk.fswAlgorithms import mrpFeedback
from Basilisk.fswAlgorithms import inertial3D
from Basilisk.fswAlgorithms import attTrackingError

# Use the SAME FlexibleSpacecraft class as check_rotation.py
from spacecraft_model import FlexibleSpacecraft
from feedforward_control import FeedforwardController, body_torque_to_rw_torque
from feedback_control import ActiveVibrationController, FilteredDerivativeController, _mrp_shadow, _mrp_subtract


CONTROLLERS = {"standard_pd", "filtered_pd", "avc"}
RUN_MODES = {"combined", "fb_only", "ff_only"}


class CometPhotographyDemo:
    """Demonstrates comet photography with input shaping - uses FlexibleSpacecraft class."""

    def __init__(self, shaping_method='fourth', controller='standard_pd', run_mode='combined'):
        if controller not in CONTROLLERS:
            raise ValueError(f"Unknown controller '{controller}'")
        if run_mode not in RUN_MODES:
            raise ValueError(f"Unknown run mode '{run_mode}'")
        self.method = shaping_method
        self.controller = controller
        self.run_mode = run_mode
        self.dt = 0.01
        self.slew_duration = 30.0
        self.settling_window = 60.0  # Increased from 30s for better convergence
        self.slew_angle_deg = 180.0
        self.slew_angle = np.radians(self.slew_angle_deg)

        # Create spacecraft model (SAME CLASS as check_rotation.py)
        self.sc = FlexibleSpacecraft()

        # Feedback controller tuning targets (for post-slew fine pointing)
        self.control_bandwidth_hz = 0.10
        self.control_damping_ratio = 0.70
        self.mrp_K = None
        self.mrp_P = None
        self.mrp_Ki = -1.0

        # Feedback controller options
        self.use_avc = controller == "avc"
        self.use_filtered_pd = controller == "filtered_pd"
        self.filter_cutoff_hz = 0.5  # Higher cutoff = less phase lag at modes
        self.ppf_damping = 0.5
        self.ppf_gains = [0.1, 0.1]  # PPF enabled for active modal damping

        # Target attitude for a +Z yaw slew
        self.initial_sigma = [0.0, 0.0, 0.0]
        self.target_sigma = [0.0, 0.0, float(np.tan(self.slew_angle / 4.0))]

        # Data storage
        self.time_log = []
        self.sigma_log = []
        self.omega_log = []
        self.modal_log = {'mode1_port': [], 'mode2_port': [], 'mode1_stbd': [], 'mode2_stbd': [], 'time': []}
        self.control_mode_log = []
        self.ff_torque_log = []
        self.fb_torque_log = []
        self.total_torque_log = []
        self.rw_torque_log = []

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
        earth = gravFactory.createEarth()
        earth.isCentralBody = True
        mu = earth.mu
        gravFactory.addBodiesTo(scObject)
        print("  Gravity: ENABLED")

        # Orbit
        oe = orbitalMotion.ClassicElements()
        oe.a = 6378137.0 + 500000.0
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
        print(f"  Orbit altitude: {(np.linalg.norm(rN) - 6378137)/1000:.1f} km")

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
        camera_body = np.array([0.0, -1.0, 0.0])
        c_yaw = float(np.cos(self.slew_angle))
        s_yaw = float(np.sin(self.slew_angle))
        rot_z = np.array([[c_yaw, -s_yaw, 0.0],
                          [s_yaw,  c_yaw, 0.0],
                          [0.0,    0.0,   1.0]])
        comet_direction = rot_z @ camera_body
        comet_direction = comet_direction / np.linalg.norm(comet_direction)
        
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
        self.inertia_for_control = hub_inertia
        ff_inertia = self.sc.compute_effective_inertia(include_flex=True)

        if self.mrp_K is None or self.mrp_P is None:
            I_control = hub_inertia[2, 2]
            omega_n = 2 * np.pi * self.control_bandwidth_hz
            self.mrp_K = omega_n**2 * I_control
            self.mrp_P = 2 * self.control_damping_ratio * np.sqrt(self.mrp_K * I_control)

        if self.filter_cutoff_hz is None:
            first_mode_hz = self.sc.array_modes[0]['frequency']
            self.filter_cutoff_hz = min(0.5 * first_mode_hz, 2.0 * self.control_bandwidth_hz)

        print(f"  Control gains: K={self.mrp_K:.1f}, P={self.mrp_P:.1f}")
        print(f"  Target bandwidth: {self.control_bandwidth_hz:.2f} Hz, filter cutoff: {self.filter_cutoff_hz:.2f} Hz")

        self.Gs_matrix = np.array([
            [np.sqrt(2)/2, -np.sqrt(2)/2, 0.0],
            [np.sqrt(2)/2,  np.sqrt(2)/2, 0.0],
            [0.0,           0.0,          1.0]
        ])

        self.ff_controller = FeedforwardController(
            ff_inertia, self.Gs_matrix, max_torque=self.sc.rw_max_torque
        )

        rotation_axis = np.array([0.0, 0.0, 1.0])

        if self.method == 'zvd':
            shaper_data = np.load('spacecraft_shaper_zvd_180deg_30s.npz', allow_pickle=True)
            shaper_info = shaper_data['shaper_info'].item()
            self.ff_controller.design_maneuver(
                theta_final=self.slew_angle,
                rotation_axis=rotation_axis,
                duration=shaper_info['base_duration'],
                shaper_amplitudes=shaper_data['amplitudes'],
                shaper_times=shaper_data['times'],
                trajectory_type='bang-bang'
            )
        elif self.method == 'fourth':
            self.ff_controller.load_fourth_order_trajectory(
                'spacecraft_trajectory_4th_180deg_30s.npz',
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
            hub_inertia[0, 0], 0, 0,
            0, hub_inertia[1, 1], 0,
            0, 0, hub_inertia[2, 2]
        ]
        vcMsg = messaging.VehicleConfigMsg().write(vehicleConfigOut)
        mrpControlObj.vehConfigInMsg.subscribeTo(vcMsg)

        self.inertial3DObj = inertial3DObj
        self.attErrorObj = attErrorObj
        self.mrpControlObj = mrpControlObj

        if self.use_avc:
            modal_freqs = [mode['frequency'] for mode in self.sc.array_modes]
            modal_damping = [mode['damping'] for mode in self.sc.array_modes]
            self.avc = ActiveVibrationController(
                inertia=hub_inertia,
                K=self.mrp_K,
                P=self.mrp_P,
                filter_freq_hz=self.filter_cutoff_hz,
                modal_freqs_hz=modal_freqs,
                modal_damping=modal_damping,
                ppf_damping=self.ppf_damping,
                ppf_gains=self.ppf_gains
            )
            self.avc.set_target(np.array(self.target_sigma))
            print(f"  AVC enabled: cutoff={self.filter_cutoff_hz:.2f} Hz, gains={self.ppf_gains}")
        elif self.use_filtered_pd:
            self.filtered_pd = FilteredDerivativeController(
                inertia=hub_inertia,
                K=self.mrp_K,
                P=self.mrp_P,
                filter_freq_hz=self.filter_cutoff_hz
            )
            self.filtered_pd.set_target(np.array(self.target_sigma))
            print(f"  Filtered PD: cutoff={self.filter_cutoff_hz:.2f} Hz (below modes at 0.4, 1.3 Hz)")

        feedback_label = "AVC" if self.use_avc else ("Filtered PD" if self.use_filtered_pd else "MRP Feedback")
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

        return self.scSim

    def _setup_vizard(self):
        """Setup Vizard visualization."""
        viz = vizSupport.enableUnityVisualization(
            self.scSim, "simTask", [self.scObject, self.cometObject],
            saveFile=f"viz_demo_{self.method}"
        )

        viz.settings.spacecraftSizeMultiplier = 2.0
        viz.settings.showSpacecraftLabels = 1
        viz.settings.orbitLinesOn = 1

        # Comet as bright sphere
        vizSupport.createCustomModel(viz,
            modelPath="SPHERE",
            simBodiesToModify=[self.cometObject.ModelTag],
            color=vizSupport.toRGBA255("cyan"),
            scale=[50.0, 50.0, 50.0],
            shader=1
        )

        # Camera FOV cone
        vizSupport.createConeInOut(viz,
            toBodyName=self.cometObject.ModelTag,
            fromBodyName=self.scObject.ModelTag,
            normalVector_B=[0, -1, 0],
            incidenceAngle=np.radians(10),
            isKeepIn=True,
            coneColor='cyan',
            coneHeight=200.0,
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

            if self.use_avc:
                mode1 = 0.5 * (self.mode1_port_rho.getState()[0][0] + self.mode1_stbd_rho.getState()[0][0])
                mode2 = 0.5 * (self.mode2_port_rho.getState()[0][0] + self.mode2_stbd_rho.getState()[0][0])
                torque_cmd = self.avc.compute_torque(
                    sigma_current, omega_current, sim_time, modal_displacements=[mode1, mode2]
                )
                return torque_cmd, "AVC"

            if self.use_filtered_pd:
                torque_cmd = self.filtered_pd.compute_torque(sigma_current, omega_current, sim_time)
                return torque_cmd, "FB(FILT)"

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

            # COMBINED control: FF + FB
            body_torque = ff_torque + fb_torque
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

            self.modal_log['time'].append(current_time)
            self.modal_log['mode1_port'].append(self.mode1_port_rho.getState()[0][0])
            self.modal_log['mode2_port'].append(self.mode2_port_rho.getState()[0][0])
            self.modal_log['mode1_stbd'].append(self.mode1_stbd_rho.getState()[0][0])
            self.modal_log['mode2_stbd'].append(self.mode2_stbd_rho.getState()[0][0])

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
            
            jitter_arcsec = (total_rms / 1000 / 4.0) * (180/np.pi) * 3600
            blur_px = jitter_arcsec / 2.0
            
            print(f"\n  VIBRATION (imaging impact):")
            print(f"    Modal RMS: {total_rms:.2f} mm")
            print(f"    Jitter: {jitter_arcsec:.1f} arcsec")
            print(f"    Blur: {blur_px:.1f} px")

    def _save_results(self):
        """Save results."""
        if self.run_mode == "combined":
            output = f'vizard_demo_{self.method}_{self.controller}.npz'
        elif self.run_mode == "fb_only":
            output = f'vizard_demo_{self.method}_{self.controller}_fb_only.npz'
        else:
            output = f'vizard_demo_{self.method}_ff_only.npz'
        mode1 = np.sqrt(np.array(self.modal_log['mode1_port'])**2 + np.array(self.modal_log['mode1_stbd'])**2)
        mode2 = np.sqrt(np.array(self.modal_log['mode2_port'])**2 + np.array(self.modal_log['mode2_stbd'])**2)

        np.savez(output,
                 time=np.array(self.time_log),
                 sigma=np.array(self.sigma_log),
                 omega=np.array(self.omega_log),
                 mode1=mode1, mode2=mode2,
                 control_mode=self.control_mode_log,
                 method=self.method,
                 controller=self.controller,
                 run_mode=self.run_mode,
                 ff_torque=np.array(self.ff_torque_log),
                 fb_torque=np.array(self.fb_torque_log),
                 total_torque=np.array(self.total_torque_log),
                 rw_torque=np.array(self.rw_torque_log),
                 slew_angle_deg=self.slew_angle_deg,
                 initial_sigma=self.initial_sigma,
                 target_sigma=self.target_sigma)
        print(f"\nSaved {output}")


if __name__ == "__main__":
    print("=" * 60)
    print("Comet Photography Demo")
    print("Using FlexibleSpacecraft class (matches check_rotation.py)")
    print("=" * 60)

    parser = argparse.ArgumentParser(description="Run the comet photography demo.")
    parser.add_argument("method", nargs="?", default="fourth", choices=["unshaped", "zvd", "fourth"])
    parser.add_argument("--controller", default="standard_pd", choices=sorted(CONTROLLERS))
    parser.add_argument("--mode", default="combined", choices=sorted(RUN_MODES))
    args = parser.parse_args()

    print(f"\nMethod: {args.method}")
    print(f"Controller: {args.controller}")
    print(f"Run mode: {args.mode}")

    demo = CometPhotographyDemo(args.method, controller=args.controller, run_mode=args.mode)
    demo.build_simulation()
    demo.run()
