"""Quick diagnostic to check vibration levels."""
import numpy as np
from mission_simulation import (
    default_config, _compute_torque_profile, _simulate_modal_response,
    _extract_vibration_signals, _compute_feedforward_metrics,
    _collect_feedforward_data
)

config = default_config()
print(f"Modal gains: {config.modal_gains}")
print(f"Modal freqs: {config.modal_freqs_hz}")
print(f"Modal damping: {config.modal_damping}")

# Test the full data collection path that run_feedforward_comparison uses
print("\n=== Testing _collect_feedforward_data (prefer_npz=False) ===")
torque_data, vibration_data = _collect_feedforward_data(config, prefer_npz=False)

for method in ["unshaped", "zvd", "fourth"]:
    vd = vibration_data[method]
    td = torque_data[method]
    disp = vd["displacement"]
    maneuver_end = vd.get("maneuver_end", config.slew_duration_s)
    t = vd["time"]
    settle_idx = int(np.searchsorted(t, maneuver_end))
    
    if len(disp) > settle_idx:
        residual = disp[settle_idx:]
        rms_residual = np.sqrt(np.mean(residual**2)) * 1000
        peak_residual = np.max(np.abs(residual)) * 1000
    else:
        rms_residual = 0
        peak_residual = 0
    
    metrics = _compute_feedforward_metrics(td, vd, config)
    print(f"{method}: manual RMS={rms_residual:.2f}mm, peak={peak_residual:.2f}mm | "
          f"metrics: RMS={metrics['rms_vibration_mm']:.2f}mm, peak={metrics['peak_vibration_mm']:.2f}mm")

# Also check with prefer_npz=True
print("\n=== Testing _collect_feedforward_data (prefer_npz=True) ===")
torque_data2, vibration_data2 = _collect_feedforward_data(config, prefer_npz=True)

for method in ["unshaped", "zvd", "fourth"]:
    vd = vibration_data2[method]
    td = torque_data2[method]
    disp = vd["displacement"]
    
    metrics = _compute_feedforward_metrics(td, vd, config)
    print(f"{method}: max_disp={np.max(np.abs(disp))*1000:.2f}mm | "
          f"metrics: RMS={metrics['rms_vibration_mm']:.2f}mm")
