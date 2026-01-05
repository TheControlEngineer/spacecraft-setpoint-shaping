# Spacecraft Setpoint Shaping for Vibration Control 

A comprehensive Python library implementing input shaping techniques for controlling residual vibrations in flexible spacecraft structures, with application to  comet tracking and imaging.

## 🎯 Project Overview

This project demonstrates the design and application of input shaping controllers for spacecraft with flexible appendages (solar arrays), culminating in a high-fidelity mission scenario: tracking and photographing a comet while minimizing image blur from structural vibrations.

**Inspired by:** Mars Reconnaissance Orbiter's HiRISE camera observations of Comet 3I ATLAS

## 🚀 Features

### Implemented Shapers
- **ZV (Zero Vibration):** 2-impulse shaper, fastest but least robust
- **ZVD (Zero Vibration Derivative):** 3-impulse shaper with improved robustness
- **ZVDD (Zero Vibration Double Derivative):** 4-impulse shaper, maximum robustness
- **EI (Extra-Insensitive):** Optimized for specified tolerance across frequency uncertainty

### Analysis Capabilities
- Frequency response analysis
- Robustness quantification (Monte Carlo with ±20% frequency uncertainty)
- Performance vs. duration trade-off studies
- Multi-mode shaping (cascaded and simultaneous optimization) - *Coming soon*

### Mission Application
- High-fidelity flexible spacecraft dynamics in Basilisk
- Comet tracking scenario with camera blur modeling
- Comparison with LQR feedback control - *Coming soon*
- 3D visualization with Vizard - *Coming soon*

## 📦 Installation

### Requirements
- Python 3.10+
- NumPy, SciPy, Matplotlib
- Basilisk

### Install Package
```bash
git clone https://github.com/YOUR_USERNAME/spacecraft-input-shaping.git
cd spacecraft-input-shaping
pip install -e .
```

## 🔧 Quick Start
```python
from input_shaping import ZV, ZVD, ZVDD, design_shaper
import numpy as np

# Define flexible mode parameters
omega_n = 2 * np.pi * 0.5  # 0.5 Hz natural frequency
zeta = 0.02                 # 2% damping ratio

# Design shapers
A_zv, t_zv = ZV(omega_n, zeta)
A_zvd, t_zvd = ZVD(omega_n, zeta)

# Or use convenience function
A, t = design_shaper(omega_n, zeta, method='ZVD')

print(f"ZVD Shaper: {len(A)} impulses")
print(f"Amplitudes: {A}")
print(f"Times: {t}")
print(f"Duration: {t[-1]:.2f} seconds")
```

## 🧪 Testing

Run the test suite:
```bash
pytest tests/ -v
```

With coverage:
```bash
pytest tests/ --cov=src/input_shaping --cov-report=html
```

## 📚 Theory

Input shaping is a feedforward control technique that shapes commanded inputs to avoid exciting structural vibrations. By convolving a reference command with a sequence of impulses designed based on the system's natural frequency and damping, residual vibrations can be eliminated or significantly reduced.

**Key Trade-off:** More impulses → better robustness to frequency uncertainty → longer maneuver duration

## 🗺️ Roadmap

- [x] Core shaper library (ZV/ZVD/ZVDD/EI)
- [x] Unit tests and validation
- [x] Robustness analysis
- [x] Multi-mode shaping (cascaded + simultaneous)
- [ ] Basilisk spacecraft model with flexible solar arrays
- [ ] Comet tracking mission scenario
- [ ] LQR comparison
- [ ] Camera blur modeling
- [ ] Vizard visualization
- [ ] Technical write-up

## 👤 Author

**Jomin Joseph Karukakalam**
- Background: M.Sc Systems & Control
- LinkedIn: https://www.linkedin.com/in/jomin-joseph-karukakalam-601955225
- Email: j.j.karukakalam@outlook.com



## 📝 License

MIT License - See LICENSE file for details

---

**Status:** 🟢 Active Development | Week 1 of 12-week intensive project
