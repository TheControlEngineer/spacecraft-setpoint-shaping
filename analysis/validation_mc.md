# Verification, Validation, and Monte Carlo Plan

This plan is tailored to the current Basilisk-based feedforward + feedback attitude control workflow in this repo. It is designed to be **comprehensive and actionable**, with each item stating **what** we do, **why** we do it, **how it verifies/validates**, and **how to execute** in practice.

---

## 0) Scope and Objectives

**Objective:** Prove (a) correctness of implementation (verification), (b) physics and performance validity (validation), and (c) robustness under uncertainty (Monte Carlo). We focus on:
- Feedforward (FF) shaping (unshaped vs fourth-order)
- Feedback control (standard PD vs filtered PD)
- Closed-loop tracking
- Disturbance rejection
- Noise rejection
- Model uncertainty robustness

**Primary outputs:**
- Tracking metrics (error time series, RMS, peak)
- Vibration metrics (modal displacement/accel, PSD, RMS)
- Torque metrics (peak, RMS, PSD, saturation percent)
- Sensitivity/Complementary sensitivity (S/T), noise and disturbance TFs
- Coherence (FF vs FB), PSD and time domain comparisons
- Monte Carlo distributions with pass/fail rates

---

## 1) Verification Plan (Implementation Correctness)

### 1.1 Feedforward Trajectory Consistency
**What:** Verify the designed FF trajectories (bang-bang for unshaped, 4th-order file for shaped) are correctly loaded, resampled, and used.
**Why:** If the reference trajectory is wrong or misaligned, all downstream tracking and vibration results are invalid.
**How it verifies:** Checks that the trajectory is internally consistent (theta, omega, alpha) and final angle is correct.
**How to do it:**
- Compare trajectory end angle to target (`slew_angle_deg`) for both methods.
- Confirm torque = I_axis * alpha used consistently.
- Ensure time/trajectory arrays align in length and sampling.
- Check that the resampling warnings are not excessive.

### 1.2 Feedback Controller Implementation
**What:** Verify standard PD and filtered PD controllers use correct gains and filter cutoff.
**Why:** Incorrect gains/cutoff can lead to misleading performance or instability.
**How it verifies:** Confirms the control law implemented matches the intended equations and config.
**How to do it:**
- Inspect controller setup in `feedback_control.py` and `_compute_control_analysis`.
- Log actual K/P values printed by runtime and compare to analytic design values.
- Verify filtered PD cutoff (`control_filter_cutoff_hz`) is passed to controller.

### 1.3 Logging and Signal Integrity
**What:** Verify logged signals are consistent and correctly aligned (time, torque, modes, sigma, omega).
**Why:** Misalignment or use of wrong signals will corrupt PSDs, RMS metrics, and plots.
**How it verifies:** Ensures raw data integrity and correct axis projection.
**How to do it:**
- Confirm array lengths are consistent after alignment.
- Verify `fb_torque`, `total_torque`, and `rw_torque` are logged and non-empty.
- Verify acceleration is taken from model states (`mode*_acc`) and not numerically differentiated.

### 1.4 Frequency-Domain Computations
**What:** Verify PSD and S/T computations are correct in units and scaling.
**Why:** Incorrect PSD scaling gives wrong quantitative conclusions.
**How it verifies:** Confirms PSD uses 10log10 and proper Welch settings.
**How to do it:**
- Check Welch parameters: `fs`, `nperseg`, `noverlap`, `window`, `detrend`.
- Confirm PSD conversions use 10log10 (not 20log10).
- Confirm PSD units (N·m)^2/Hz for torque.

---

## 2) Validation Plan (Physics and Performance Validity)

### 2.1 Closed-Loop Tracking Validation
**What:** Compare reference trajectory vs actual angle and error.
**Why:** Ensures system tracks desired trajectory and ends at correct final target.
**How it validates:** Confirms actual system behavior matches expected attitude response.
**How to do it:**
- Use `mission_tracking_response.png` (unwrapped angle and wrapped error).
- Quantify post-slew steady-state error and compare to tolerance.
- Confirm no drift in unshaped case after switching to designed reference trajectory.

### 2.2 Disturbance Rejection Validation
**What:** Validate that disturbances do not significantly degrade pointing.
**Why:** Real systems face disturbances; rejection must meet requirements.
**How it validates:** Confirms modeled disturbances are suppressed as expected.
**How to do it:**
- Use `mission_disturbance_tf.png` and `mission_disturbance_to_torque.png`.
- Inject known disturbance torque profiles (constant bias, sinusoid near modes, broadband).
- Compare pointing error RMS with and without disturbances.

### 2.3 Noise Rejection Validation
**What:** Validate measurement noise does not cause excessive torque or pointing noise.
**Why:** Noise amplification can destroy pointing quality and actuators.
**How it validates:** Confirms noise transfer is within acceptable bounds.
**How to do it:**
- Use noise TF (`mission_noise_to_torque.png`) to check |C/(1+PC)|.
- Inject white/colored noise at sensor and confirm error PSD and torque PSD behavior.

### 2.4 Flexible Mode Suppression Validation
**What:** Validate vibration response at modal frequencies.
**Why:** Feedforward shaping and PD must suppress vibration.
**How it validates:** Confirms modal PSD peaks are reduced at target modes.
**How to do it:**
- Inspect vibration PSD plots and modal displacement metrics.
- Compare unshaped vs fourth-order performance at 0.4 Hz and 1.3 Hz.
- Use band-limited RMS metrics for modes.

### 2.5 Torque/Actuator Capability Validation
**What:** Validate commanded torque is within actuator limits and not too aggressive.
**Why:** Hardware torque/momentum limits are hard constraints.
**How it validates:** Ensures all commanded torque is feasible.
**How to do it:**
- Use `torque_command_metrics.csv` to report peak/RMS and saturation percent.
- Ensure peak wheel torque < rw_max_torque.
- Optionally compute margin to saturation in %.

---

## 3) Monte Carlo Plan (Robustness Under Uncertainty)

### 3.1 Uncertain Parameters
**What:** Sample uncertainties for a range of physical and control parameters.
**Why:** Real spacecraft parameters are uncertain; must show robustness.
**How it verifies/validates:** Ensures performance is stable under parameter variations.
**How to do it:**
- Use 500–5000 runs depending on resources.
- Sample distributions:
  - Inertia matrix elements ±10–20%
  - Modal frequencies ±10%
  - Modal damping ±50%
  - Modal gains ±20%
  - Sensor noise std dev ±50%
  - Disturbance torque bias ±50%
  - Control filter cutoff ±20%

### 3.2 Monte Carlo Metrics
**What:** Track distributions for each run.
**Why:** Need statistical confidence, not a single run.
**How it verifies/validates:** Measures robustness and pass rate.
**How to do it:**
- Track metrics:
  - Peak pointing error
  - RMS pointing error
  - RMS vibration (post-slew)
  - Peak torque
  - RMS torque
  - Torque saturation percent
- Produce histograms and percentile tables (P50, P95, P99).

### 3.3 Monte Carlo Pass/Fail Criteria
**What:** Define objective thresholds for success.
**Why:** Allows go/no-go decisions.
**How it verifies/validates:** Confirms design meets mission requirements.
**How to do it:**
- Example thresholds:
  - RMS pointing error < 0.005 deg (P95)
  - Peak torque < 70 Nm (P99)
  - RMS vibration < 0.1 mm (P95)
  - Closed-loop stable for all runs
- Compute pass rate and confidence.

---

## 4) Step-by-Step Execution Workflow

### 4.1 Verification (One-time per code change)
1. Run baseline simulation (unshaped/4th with both controllers).
2. Check trajectory load, sampling, and final angles.
3. Confirm log signals exist and are aligned.
4. Verify PSD parameters and scaling.

### 4.2 Validation (Nominal case)
1. Run `mission_simulation.py` and generate plots/CSVs.
2. Inspect tracking response and error plots.
3. Inspect disturbance/noise transfer functions.
4. Check vibration PSD peaks at modes.
5. Check torque metrics vs actuator limits.

### 4.3 Monte Carlo (Robustness)
1. Generate random parameter sets (JSON/YAML or CSV input).
2. For each run:
   - Update config (inertia, modal frequencies, damping, etc.).
   - Run feedforward + feedback simulation.
   - Log key metrics to MC summary file.
3. Post-process to produce distributions and pass/fail report.

---

## 5) Deliverables

- **Plots**
  - mission_tracking_response.png
  - mission_tracking_tf.png
  - mission_disturbance_tf.png
  - mission_noise_to_torque.png
  - mission_disturbance_to_torque.png
  - mission_torque_command.png
  - mission_torque_command_psd.png
  - mission_torque_psd_split.png
  - mission_torque_psd_coherence.png

- **CSV Reports**
  - torque_command_metrics.csv
  - torque_psd_rms.csv
  - mission_summary.csv
  - psd_mission.csv

- **Monte Carlo Summary**
  - distribution plots, percentile tables, pass/fail counts

---

## 6) Notes / Assumptions

- The feedforward trajectory for “fourth” is loaded from `spacecraft_trajectory_4th_180deg_30s.npz`.
- Noise and disturbance models should be explicitly defined in the MC harness.
- Reaction wheel saturation is based on `rw_max_torque` in `spacecraft_model.py` (default 70 Nm).

---

## 7) Next Actions (optional)

1. Implement a Monte Carlo runner (Python script) that perturbs config values and aggregates metrics.
2. Add explicit disturbance/noise injection hooks in `vizard_demo.py` to drive controlled experiments.
3. Define mission-specific pass/fail thresholds.

---

End of plan.
