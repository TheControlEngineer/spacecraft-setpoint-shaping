# Input Shaping Library

A comprehensive Python library implementing various input shaping techniques for suppressing residual vibrations in flexible structures, with applications to spacecraft attitude control, precision positioning systems, and robotics.

## Features

- **Multiple Shaper Types**:
  - **ZV (Zero Vibration)**: 2-impulse shaper for basic vibration cancellation
  - **ZVD (Zero Vibration Derivative)**: 3-impulse shaper with moderate robustness
  - **ZVDD (Zero Vibration Double Derivative)**: 4-impulse shaper with enhanced robustness
  - **EI (Extra-Insensitive)**: Optimized shaper for specific frequency uncertainty bands

- **Easy-to-use API**: Simple function calls with clear parameter definitions
- **Well-documented**: Comprehensive docstrings and examples
- **Type hints**: Full type annotation support for better IDE integration

## Installation

```bash
pip install -e .
```

For development with testing tools:

```bash
pip install -e ".[dev]"
```

## Quick Start

```python
from input_shaping import design_shaper
import numpy as np

# System parameters
omega_n = 1.0  # Natural frequency (rad/s)
zeta = 0.05    # Damping ratio

# Design a ZVD shaper
amplitudes, times = design_shaper(omega_n, zeta, method='ZVD')

print(f"Impulse amplitudes: {amplitudes}")
print(f"Impulse times: {times}")
```

## Usage Examples

### Using specific shapers directly

```python
from input_shaping import ZV, ZVD, ZVDD, EI

# Zero Vibration shaper
A_zv, t_zv, K = ZV(omega_n=1.0, zeta=0.05)

# Zero Vibration Derivative shaper
A_zvd, t_zvd, K = ZVD(omega_n=1.0, zeta=0.05)

# Zero Vibration Double Derivative shaper
A_zvdd, t_zvdd = ZVDD(omega_n=1.0, zeta=0.05)

# Extra-Insensitive shaper
A_ei, t_ei = EI(omega_n=1.0, zeta=0.05, Vtol=0.05, tol_band=0.20)
```

### Using the convenience function

```python
from input_shaping import design_shaper

# Design different types of shapers
A, t = design_shaper(omega_n=1.0, zeta=0.05, method='ZV')
A, t = design_shaper(omega_n=1.0, zeta=0.05, method='ZVD')
A, t = design_shaper(omega_n=1.0, zeta=0.05, method='ZVDD')
A, t = design_shaper(omega_n=1.0, zeta=0.05, method='EI', Vtol=0.05, tol_band=0.20)
```

## Shaper Comparison

| Shaper | Impulses | Robustness | Duration | Use Case |
|--------|----------|------------|----------|----------|
| ZV     | 2        | Low        | Shortest | Fast response, minimal uncertainty |
| ZVD    | 3        | Moderate   | Medium   | Balanced performance |
| ZVDD   | 4        | High       | Longest  | High uncertainty environments |
| EI     | 3        | Tunable    | Optimized| Custom frequency uncertainty bands |

## Requirements

- Python >= 3.10
- numpy >= 1.20.0
- scipy >= 1.7.0
- matplotlib >= 3.3.0

## License

[Add your license here]

## References

Singer, N. C., & Seering, W. P. (1990). Preshaping command inputs to reduce system vibration. *Journal of Dynamic Systems, Measurement, and Control*, 112(1), 76-82.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
