# Setpoint Shaping and Pointing Control for Satellite Repositioning

This project designs, implements, and validates a control architecture for a flexible spacecraft that must execute a fast 180 deg yaw slew in 30 seconds while preserving post-maneuver pointing stability for imaging.

Core idea: combine a shaped feedforward trajectory (to avoid exciting flexible modes) with MRP-based feedback control (to track and reject disturbances).

## Mission Context

- Spacecraft mass: 750 kg
- Slew maneuver: 180 deg about body Z axis in 30 s
- Structural challenge: lightly damped flexible solar-array modes at 0.4 Hz and 1.3 Hz
- Pointing requirement: post-slew RMS pointing error below 5 arcsec
- Software baseline: Python 3.10+, NumPy/SciPy/Matplotlib, Basilisk

## Why This Matters

Large slews inject energy into flexible appendages. Residual vibration then degrades pointing and image quality, forcing mission downtime while the structure settles. This project targets that bottleneck by reducing modal excitation at the command-design level instead of relying only on feedback damping.

## Technical Approach

- **Nonlinear plant modeling:** rigid hub + 3-wheel actuation + flexible appendage dynamics in Basilisk.
- **Feedback design:** MRP-based PD control with gain tuning around a 70 to 75 deg phase-margin target.
- **Feedforward design:** compare S-curve baseline against fourth-order setpoint shaping with spectral notches near structural modes.
- **Validation campaign:** mission run and one-factor sensitivity sweeps in the nonlinear simulation loop.

## Engineering Skills Demonstrated

- Spacecraft attitude dynamics and MRP kinematics
- Flexible-mode modeling and vibration suppression
- Frequency-domain control analysis (sensitivity and disturbance/noise pathways)
- Simulation-based verification and robustness sweeps
- Python tooling for reproducible analysis outputs (plots + metrics exports)

## Simulation Architecture

![Mission simulation architecture](Docs/plots/image-15.png)

## Key Results

Mission-run comparison from `Docs/design_report.md`:

| Metric | S-curve + PD | Fourth-order + PD | Outcome |
|---|---:|---:|---|
| Post-slew RMS modal displacement | 0.2534 mm | 0.099 mm | 60.9% reduction |
| Post-slew RMS modal acceleration | 1.561 mm/s^2 | 0.445 mm/s^2 | 71.5% reduction |
| Post-slew RMS pointing error | Baseline higher | 4.65 arcsec | Meets 5 arcsec requirement (64.1% lower than baseline) |
| Estimated imaging blur | 10 px | 3.3 px | About 70% blur reduction |
| Relative mode-2 residual content | Baseline | 4.83x lower | Less energy injected into higher mode |

### Residual Vibration Comparison

![Mission vibration comparison](Docs/plots/mission_vibration.png)

### Pointing Error Comparison

![Mission pointing error comparison](Docs/plots/mission_pointing_error.png)

### Imaging Blur Impact

![Comet blur comparison](Docs/plots/comet_blur_comparison_psd_check.png)

### Robustness Sweep Example (Modal Frequency Uncertainty)

![Modal frequency sweep](Docs/plots/mc_sweep_modal_frequency.png)

## Vizard Demo (Video Placeholder)

Replace the link below with your simulation video:

[![Vizard demo placeholder](Docs/plots/mission_simulation_architecture.svg)](PASTE_YOUR_VIZARD_VIDEO_LINK_HERE)

## Repository Structure

```text
basilisk_simulation/
|- src/basilisk_sim/          # Dynamics, feedforward, feedback, models
|- scripts/
|  |- run_vizard_demo.py      # Nonlinear mission simulation + Vizard scenario
|  |- run_mission.py          # Unified mission analysis, plots, CSV exports
|  |- run_survey_scenario.py  # Step-and-stare mission framing
|- analysis/
|  |- monte_carlo_factor_sweeps.py
|  |- validation_mc_runner.py
|- Docs/
|  |- design_report.md
|  |- plots/
|- data/trajectories/         # Reference trajectories + simulation NPZ files
|- output/                    # Generated plots/metrics
```

## Installation

Prerequisites:

- Python 3.10 or newer
- Basilisk Python package
- Optional for interactive 3D viewing: Vizard runtime

### 1. Create and activate a virtual environment

```bash
python -m venv .venv
```

Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

macOS/Linux:

```bash
source .venv/bin/activate
```

### 2. Install package dependencies

```bash
python -m pip install --upgrade pip
pip install -e ".[basilisk]"
```

If you are running from the full monorepo and want the shared input-shaping package too:

```bash
pip install -e ..
```

## How To Run

### A. Generate fresh mission trajectories (recommended)

From `basilisk_simulation/`:

```bash
python scripts/run_vizard_demo.py s_curve --controller standard_pd --mode combined --output-dir data/trajectories
python scripts/run_vizard_demo.py fourth --controller standard_pd --mode combined --output-dir data/trajectories
```

### B. Run mission analysis and create plots/metrics

```bash
python scripts/run_mission.py --data-dir data/trajectories
```

Generated artifacts:

- Plots: `output/plots/`
- Metrics CSVs: `output/metrics/`

### C. Run sensitivity sweeps

```bash
python analysis/monte_carlo_factor_sweeps.py --samples 500 --out-dir output/mc_sweep_500
```

### D. Run verification/validation checks

```bash
python analysis/validation_mc_runner.py --verification --validation --output-dir analysis
```

## Design Report

Full derivations, modeling assumptions, controller rationale, and result interpretation are documented in:

- `Docs/design_report.md`

## Notes

- Monte Carlo automation and report JSON outputs are included under `analysis/`.
- This project emphasizes dynamics/control behavior and vibration suppression effectiveness for fast repositioning and imaging readiness.
