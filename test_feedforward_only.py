"""
Diagnostic Test: Feedforward Only (No Feedback)

Tests if the feedforward trajectory alone can achieve 180-degree slew.
If successful: feedforward is sufficient, feedback is for fine pointing
If unsuccessful: feedforward is insufficient, feedback must complete the maneuver
"""

import sys
import numpy as np
from Basilisk.utilities import SimulationBaseClass, macros
from spacecraft_model import FlexibleSpacecraft
from feedforward_control import FeedforwardController, body_torque_to_rw_torque

def test_feedforward_only(method='unshaped', duration=30.0, angle_deg=180.0):
    """Test feedforward trajectory without any feedback."""
    
    print(f"\n{'='*70}")
    print(f"DIAGNOSTIC TEST: Feedforward Only ({method.upper()})")
    print(f"Testing if feedforward can complete {angle_deg:.1f} deg slew in {duration}s")
    print(f"{'='*70}\n")
    
    # Create simulation
    scSim = SimulationBaseClass.SimBaseClass()
    dt = 0.01
    simulationTimeStep = macros.sec2nano(dt)
    dynProcess = scSim.CreateNewProcess("DynamicsProcess")
    dynTask = scSim.CreateNewTask("simTask", simulationTimeStep)
    dynProcess.addTask(dynTask)
    
    # Create spacecraft
    sc = FlexibleSpacecraft()
    scObject = sc.create_rigid_spacecraft()
    rwStateEffector = sc.add_reaction_wheels()
    flexModes = sc.add_flexible_solar_arrays()
    
    scObject.hub.sigma_BNInit = [[0.0], [0.0], [0.0]]
    scObject.hub.omega_BN_BInit = [[0.0], [0.0], [0.0]]
    
    scSim.AddModelToTask("simTask", scObject)
    scSim.AddModelToTask("simTask", rwStateEffector)
    
    # Create feedforward controller
    inertia = sc.compute_effective_inertia(include_flex=True)
    gs_matrix = np.array([
        [np.sqrt(2)/2, -np.sqrt(2)/2, 0.0],
        [np.sqrt(2)/2, np.sqrt(2)/2, 0.0],
        [0.0, 0.0, 1.0]
    ])
    
    ff_controller = FeedforwardController(inertia, gs_matrix, max_torque=70.0)
    rotation_axis = np.array([0.0, 0.0, 1.0])
    target_angle_deg = float(angle_deg)
    slew_angle = np.radians(target_angle_deg)
    
    # Design maneuver
    if method == 'fourth':
        ff_controller.load_fourth_order_trajectory(
            'spacecraft_trajectory_4th_180deg_30s.npz',
            rotation_axis
        )
    elif method == 'zvd':
        shaper_data = np.load('spacecraft_shaper_zvd_180deg_30s.npz', allow_pickle=True)
        shaper_info = shaper_data['shaper_info'].item()
        ff_controller.design_maneuver(
            theta_final=slew_angle,
            rotation_axis=rotation_axis,
            duration=shaper_info['base_duration'],
            shaper_amplitudes=shaper_data['amplitudes'],
            shaper_times=shaper_data['times'],
            trajectory_type='bang-bang'
        )
    else:  # unshaped
        ff_controller.design_maneuver(
            theta_final=slew_angle,
            rotation_axis=rotation_axis,
            duration=duration,
            trajectory_type='bang-bang'
        )
    
    actual_duration = ff_controller.t_profile[-1]
    print(f"Feedforward trajectory duration: {actual_duration:.2f}s")
    print(f"Peak torque: {np.max(np.abs(ff_controller.rw_torque_profile)):.2f} Nm")
    print(f"RW torque limit: {sc.rw_max_torque:.2f} Nm\n")
    
    # Initialize RW command message
    from Basilisk.architecture import messaging
    rwMotorCmdMsg = messaging.ArrayMotorTorqueMsg()
    rwStateEffector.rwMotorCmdInMsg.subscribeTo(rwMotorCmdMsg)
    
    scSim.InitializeSimulation()
    
    # Run simulation with FEEDFORWARD ONLY (no feedback)
    total_time = actual_duration + 10.0  # Run 10s past feedforward end
    n_steps = int(total_time / dt)
    
    time_log = []
    sigma_log = []
    omega_log = []
    torque_log = []
    
    print("Running simulation (feedforward only, no feedback)...")
    for step in range(n_steps):
        current_time = step * dt
        
        # ONLY FEEDFORWARD - NO FEEDBACK
        if current_time <= actual_duration:
            rw_torques = ff_controller.get_torque(current_time)
            rw_torque_cmd = rw_torques
        else:
            rw_torque_cmd = np.zeros(3)
        
        # Clip to RW limits
        rw_torque_cmd = np.clip(rw_torque_cmd, -sc.rw_max_torque, sc.rw_max_torque)
        
        # Command RWs
        rw_cmd_payload = messaging.ArrayMotorTorqueMsgPayload()
        motor_torque = rw_cmd_payload.motorTorque
        motor_torque[0] = float(rw_torque_cmd[0])
        motor_torque[1] = float(rw_torque_cmd[1])
        motor_torque[2] = float(rw_torque_cmd[2])
        rw_cmd_payload.motorTorque = motor_torque
        rwMotorCmdMsg.write(rw_cmd_payload)
        
        scSim.ConfigureStopTime(macros.sec2nano((step + 1) * dt))
        scSim.ExecuteSimulation()
        
        # Log data
        sigma = scObject.scStateOutMsg.read().sigma_BN
        omega = scObject.scStateOutMsg.read().omega_BN_B
        
        time_log.append(current_time)
        sigma_log.append(sigma)
        omega_log.append(omega)
        torque_log.append(rw_torque_cmd.copy())
        
        if step % 1000 == 0:
            # Compute current rotation angle
            sigma_vec = np.array(sigma)
            current_angle_rad = 4 * np.arctan(np.linalg.norm(sigma_vec))
            current_angle_deg = np.degrees(current_angle_rad)
            print(f"  t={current_time:.1f}s: Rotation = {current_angle_deg:.2f} deg")
    
    print("\nDone!\n")
    
    # Analyze results
    print(f"{'='*70}")
    print("RESULTS:")
    print(f"{'='*70}")
    
    # Check rotation at key times
    key_times = [actual_duration - 0.01, actual_duration, actual_duration + 5.0, actual_duration + 10.0]
    
    for t_check in key_times:
        idx = int(t_check / dt)
        if idx >= len(sigma_log):
            continue
        
        sigma = np.array(sigma_log[idx])
        omega = np.array(omega_log[idx])
        
        angle_rad = 4 * np.arctan(np.linalg.norm(sigma))
        achieved_angle_deg = np.degrees(angle_rad)
        omega_deg_s = np.degrees(omega)
        
        print(f"\nAt t={time_log[idx]:.2f}s:")
        print(f"  Rotation achieved: {achieved_angle_deg:.4f} deg (target: {target_angle_deg:.4f} deg)")
        print(f"  Error: {achieved_angle_deg - target_angle_deg:.4f} deg")
        print(f"  Angular velocity: {np.linalg.norm(omega_deg_s):.6f} deg/s")
        print(f"  Sigma: [{sigma[0]:.6f}, {sigma[1]:.6f}, {sigma[2]:.6f}]")
    
    # Final verdict
    final_sigma = np.array(sigma_log[-1])
    final_angle_deg = np.degrees(4 * np.arctan(np.linalg.norm(final_sigma)))
    error = abs(final_angle_deg - target_angle_deg)
    
    print(f"\n{'='*70}")
    if error < 1.0:  # Within 1 degree
        print(f"SUCCESS: Feedforward ALONE achieved {final_angle_deg:.2f} deg")
        print(f"  The feedforward trajectory is SUFFICIENT for the maneuver")
        print(f"  Feedback is used for fine pointing and disturbance rejection")
    else:
        print(f"FAILURE: Feedforward only reached {final_angle_deg:.2f} deg")
        print(f"  Missing: {target_angle_deg - final_angle_deg:.2f} deg")
        print(f"  The feedforward trajectory is INSUFFICIENT")
        print(f"  Feedback must complete the remaining rotation")
    print(f"{'='*70}\n")
    
    return error < 1.0

if __name__ == '__main__':
    method = sys.argv[1] if len(sys.argv) > 1 else 'unshaped'
    success = test_feedforward_only(method)
    sys.exit(0 if success else 1)
