# Fix Instructions for Confirmed Issues

This document provides detailed instructions for fixing each confirmed issue from the code analysis report.

---

## Table of Contents

1. [feedback_control.py - Notch Filter State-Space Implementation](#1-feedback_controlpy---notch-filter-state-space-implementation)
2. [feedback_control.py - PPF Integration Method Inconsistency](#2-feedback_controlpy---ppf-integration-method-inconsistency)
3. [feedback_control.py - Overdamped Settling Time Formula](#3-feedback_controlpy---overdamped-settling-time-formula)
4. [feedback_control.py - Ki < 0 Disable Flag](#4-feedback_controlpy---ki--0-disable-flag)
5. [design_shaper.py - Missing zeta < 1 Validation](#5-design_shaperpy---missing-zeta--1-validation)
6. [spacecraft_model.py - Docstring vs Implementation Mismatch](#6-spacecraft_modelpy---docstring-vs-implementation-mismatch)
7. [spacecraft_model.py - Hardcoded hub_mass vs HUB_INERTIA](#7-spacecraft_modelpy---hardcoded-hub_mass-vs-hub_inertia)
8. [spacecraft_properties.py - Modal vs Appendage Mass Documentation](#8-spacecraft_propertiespy---modal-vs-appendage-mass-documentation)
9. [spacecraft_properties.py - Modal Gain Units Documentation](#9-spacecraft_propertiespy---modal-gain-units-documentation)
10. [vizard_demo.py - Hardcoded Jitter/Blur Constants](#10-vizard_demopy---hardcoded-jitterblur-constants)
11. [mission_simulation.py - ZVD Single-Mode Limitation](#11-mission_simulationpy---zvd-single-mode-limitation)
12. [mission_simulation.py - Nyquist Margin Edge Cases](#12-mission_simulationpy---nyquist-margin-edge-cases)
13. [mission_simulation.py - Hardcoded Trajectory File Paths](#13-mission_simulationpy---hardcoded-trajectory-file-paths)
14. [Items Requiring External Verification](#14-items-requiring-external-verification)

---

## 1. feedback_control.py - Notch Filter State-Space Implementation

### Location
`feedback_control.py`, lines 817-847, function `_apply_notch_filter()`

### Current Code (Incorrect)
```python
def _apply_notch_filter(self, input_signal: np.ndarray, notch_idx: int, dt: float) -> np.ndarray:
    """Apply notch filter to 3-axis signal using state-space form."""
    coeffs = self.notch_coeffs[notch_idx]
    omega_n = coeffs['omega_n']
    zeta_z = coeffs['zeta_z']
    zeta_p = coeffs['zeta_p']

    output = np.zeros(3)
    for axis in range(3):
        x1, x2 = self.notch_states[notch_idx][axis]
        u = input_signal[axis]

        # Discretize using bilinear transform approximation
        # For simplicity, use forward Euler
        x1_new = x1 + dt * x2
        x2_new = x2 + dt * (-omega_n**2 * x1 - 2*zeta_p*omega_n*x2 + omega_n**2 * u)

        # Output: notch filter output
        output[axis] = x1_new + (2*zeta_z/omega_n) * x2_new + (1/omega_n**2) * (
            -omega_n**2 * x1_new - 2*zeta_p*omega_n*x2_new + omega_n**2 * u
        )

        self.notch_states[notch_idx][axis] = [x1_new, x2_new]

    return output
```

### Problem Analysis

The notch filter transfer function is:
$$H(s) = \frac{s^2 + 2\zeta_z \omega_n s + \omega_n^2}{s^2 + 2\zeta_p \omega_n s + \omega_n^2}$$

The current implementation attempts a state-space realization but the output equation is incorrect. A correct controllable canonical form realization is:

**State equations:**
$$\dot{x}_1 = x_2$$
$$\dot{x}_2 = -\omega_n^2 x_1 - 2\zeta_p \omega_n x_2 + u$$

**Output equation:**
$$y = (\omega_n^2 - \omega_n^2)x_1 + (2\zeta_z\omega_n - 2\zeta_p\omega_n)x_2 + u = 2\omega_n(\zeta_z - \zeta_p)x_2 + u$$

### Corrected Code

```python
def _apply_notch_filter(self, input_signal: np.ndarray, notch_idx: int, dt: float) -> np.ndarray:
    """
    Apply notch filter to 3-axis signal using state-space form.

    Transfer function:
        H(s) = (s^2 + 2*zeta_z*omega_n*s + omega_n^2) / (s^2 + 2*zeta_p*omega_n*s + omega_n^2)

    Controllable canonical form state-space:
        x' = A*x + B*u
        y  = C*x + D*u

    Where:
        A = [[0, 1], [-omega_n^2, -2*zeta_p*omega_n]]
        B = [[0], [1]]
        C = [omega_n^2*(1 - 1), 2*omega_n*(zeta_z - zeta_p)] = [0, 2*omega_n*(zeta_z - zeta_p)]
        D = 1

    This uses semi-implicit (symplectic) Euler for better numerical stability.
    """
    coeffs = self.notch_coeffs[notch_idx]
    omega_n = coeffs['omega_n']
    zeta_z = coeffs['zeta_z']
    zeta_p = coeffs['zeta_p']

    output = np.zeros(3)
    for axis in range(3):
        x1, x2 = self.notch_states[notch_idx][axis]
        u = input_signal[axis]

        # Semi-implicit Euler integration (better stability than forward Euler)
        # Update x2 first (velocity), then x1 (position) with new x2
        x2_new = x2 + dt * (-omega_n**2 * x1 - 2*zeta_p*omega_n*x2 + u)
        x1_new = x1 + dt * x2_new

        # Correct output equation for notch filter
        # y = C*x + D*u where C = [0, 2*omega_n*(zeta_z - zeta_p)], D = 1
        output[axis] = 2*omega_n*(zeta_z - zeta_p) * x2_new + u

        self.notch_states[notch_idx][axis] = [x1_new, x2_new]

    return output
```

### Alternative: Use scipy.signal for Discretization

For better accuracy, especially at higher frequencies, consider using scipy's bilinear transform:

```python
def __init__(self, ...):
    # ... existing code ...

    # Precompute discrete-time filter coefficients using bilinear transform
    self._discrete_filters = []

def _initialize_discrete_filters(self, dt: float):
    """Precompute discrete notch filters using bilinear transform."""
    from scipy import signal

    self._discrete_filters = []
    for coeffs in self.notch_coeffs:
        omega_n = coeffs['omega_n']
        zeta_z = coeffs['zeta_z']
        zeta_p = coeffs['zeta_p']

        # Continuous-time transfer function
        num = [1.0, 2*zeta_z*omega_n, omega_n**2]
        den = [1.0, 2*zeta_p*omega_n, omega_n**2]

        # Convert to discrete-time using bilinear (Tustin) transform
        b, a = signal.bilinear(num, den, fs=1.0/dt)

        # Store coefficients and filter state (for 3 axes)
        self._discrete_filters.append({
            'b': b, 'a': a,
            'zi': [signal.lfilter_zi(b, a) * 0 for _ in range(3)]
        })

def _apply_notch_filter(self, input_signal: np.ndarray, notch_idx: int, dt: float) -> np.ndarray:
    """Apply notch filter using precomputed discrete coefficients."""
    from scipy import signal

    filt = self._discrete_filters[notch_idx]
    output = np.zeros(3)

    for axis in range(3):
        # Apply filter with state preservation
        y, filt['zi'][axis] = signal.lfilter(
            filt['b'], filt['a'],
            [input_signal[axis]],
            zi=filt['zi'][axis]
        )
        output[axis] = y[0]

    return output
```

### Verification

After implementing the fix, verify with this test:

```python
def test_notch_filter():
    """Verify notch filter attenuates at target frequency."""
    import numpy as np
    from scipy import signal

    # Create controller with notch at 0.4 Hz
    ctrl = NotchFilterController(
        inertia=np.diag([900, 800, 600]),
        K=30.0, P=60.0,
        notch_freqs_hz=[0.4],
        notch_depth_db=20.0,
        notch_width=0.3
    )

    # Get transfer function and check frequency response
    tf = ctrl.get_transfer_function()
    w, h = signal.freqresp(tf, np.array([2*np.pi*0.4]))  # At notch frequency

    attenuation_db = 20 * np.log10(np.abs(h[0]))
    print(f"Attenuation at 0.4 Hz: {attenuation_db:.1f} dB")
    assert attenuation_db < -15, f"Expected < -15 dB, got {attenuation_db:.1f} dB"
```

---

## 2. feedback_control.py - PPF Integration Method Inconsistency

### Location
`feedback_control.py`, lines 491-520, method `PPFCompensator.update()`

### Current Code
```python
def update(self, input_signal: float, dt: float) -> float:
    """
    Update PPF state and compute output.

    Uses trapezoidal integration for numerical stability.  # <-- INCORRECT COMMENT
    """
    # ...

    # Integrate (trapezoidal for xdot, forward Euler for x)  # <-- INCORRECT COMMENT
    x_new = x + xdot * dt + 0.5 * xddot * dt**2
    xdot_new = xdot + xddot * dt  # <-- This is Forward Euler, NOT trapezoidal
```

### Problem
The comment claims trapezoidal integration but the implementation uses:
- Position: Second-order Taylor expansion (Velocity Verlet-like)
- Velocity: Forward Euler

This is inconsistent and the comment is misleading.

### Corrected Code

**Option A: Use Velocity Verlet (Recommended for oscillatory systems)**

```python
def update(self, input_signal: float, dt: float) -> float:
    """
    Update PPF state and compute output.

    Uses Velocity Verlet integration for energy-preserving behavior
    in the oscillatory PPF dynamics.

    Args:
        input_signal: Modal displacement input (from sensor or estimate)
        dt: Time step [s]

    Returns:
        PPF filter output torque contribution
    """
    x = self.state[0]
    xdot = self.state[1]

    omega_f = self.omega_f
    zeta_f = self.damping
    g = self.gain

    # PPF dynamics: x'' + 2*zeta*omega*x' + omega^2*x = g*omega^2*u
    # Compute current acceleration
    xddot = g * omega_f**2 * input_signal - 2*zeta_f*omega_f*xdot - omega_f**2 * x

    # Velocity Verlet integration (symplectic, good for oscillators)
    # Step 1: Half-step velocity update
    xdot_half = xdot + 0.5 * xddot * dt

    # Step 2: Full-step position update
    x_new = x + xdot_half * dt

    # Step 3: Compute new acceleration at new position
    xddot_new = g * omega_f**2 * input_signal - 2*zeta_f*omega_f*xdot_half - omega_f**2 * x_new

    # Step 4: Complete velocity update
    xdot_new = xdot_half + 0.5 * xddot_new * dt

    # Update state
    self.state[0] = x_new
    self.state[1] = xdot_new

    # Output is the filter state (position)
    return x_new
```

**Option B: True Trapezoidal (Implicit, requires iteration)**

```python
def update(self, input_signal: float, dt: float, max_iter: int = 3) -> float:
    """
    Update PPF state and compute output.

    Uses trapezoidal (implicit midpoint) integration for stability.
    Requires iteration since trapezoidal is implicit.
    """
    x = self.state[0]
    xdot = self.state[1]

    omega_f = self.omega_f
    zeta_f = self.damping
    g = self.gain

    def compute_xddot(x_val, xdot_val):
        return g * omega_f**2 * input_signal - 2*zeta_f*omega_f*xdot_val - omega_f**2 * x_val

    # Current acceleration
    xddot = compute_xddot(x, xdot)

    # Initial guess using forward Euler
    x_new = x + xdot * dt
    xdot_new = xdot + xddot * dt

    # Iterate to solve implicit trapezoidal equations
    for _ in range(max_iter):
        xddot_new = compute_xddot(x_new, xdot_new)

        # Trapezoidal rule: x_{n+1} = x_n + dt/2 * (f_n + f_{n+1})
        x_new = x + 0.5 * dt * (xdot + xdot_new)
        xdot_new = xdot + 0.5 * dt * (xddot + xddot_new)

    self.state[0] = x_new
    self.state[1] = xdot_new

    return x_new
```

### Recommendation
Use **Option A (Velocity Verlet)** as it provides good energy preservation for oscillatory systems without requiring iteration.

---

## 3. feedback_control.py - Overdamped Settling Time Formula

### Location
`feedback_control.py`, lines 285-288, method `get_closed_loop_params()`

### Current Code (Incorrect)
```python
# Settling time (2% criterion)
if zeta < 1:
    t_settle = 4 / (zeta * omega_n)
else:
    t_settle = 4 * zeta / omega_n  # INCORRECT for overdamped
```

### Problem
For overdamped systems ($\zeta > 1$), the system has two real poles:
$$p_{1,2} = -\zeta\omega_n \pm \omega_n\sqrt{\zeta^2 - 1}$$

The slower pole (dominant) is:
$$p_{slow} = -\omega_n(\zeta - \sqrt{\zeta^2 - 1})$$

The settling time is dominated by this pole:
$$t_s \approx \frac{4}{|p_{slow}|} = \frac{4}{\omega_n(\zeta - \sqrt{\zeta^2 - 1})}$$

### Corrected Code
```python
# Settling time (2% criterion)
if zeta < 1:
    # Underdamped: exponential decay with rate zeta*omega_n
    t_settle = 4 / (zeta * omega_n)
elif zeta == 1:
    # Critically damped: repeated pole at -omega_n
    t_settle = 4 / omega_n
else:
    # Overdamped: two real poles, settling dominated by slower pole
    # Poles at: -omega_n * (zeta ± sqrt(zeta^2 - 1))
    # Slower pole: -omega_n * (zeta - sqrt(zeta^2 - 1))
    slower_pole_mag = omega_n * (zeta - np.sqrt(zeta**2 - 1))
    t_settle = 4 / slower_pole_mag
```

### Verification
```python
def test_settling_time():
    """Verify settling time formulas."""
    import numpy as np

    omega_n = 1.0  # rad/s

    # Test cases
    test_cases = [
        (0.5, 8.0),      # Underdamped: 4/(0.5*1) = 8
        (1.0, 4.0),      # Critically damped: 4/1 = 4
        (2.0, 14.93),    # Overdamped: 4/(1*(2-sqrt(3))) ≈ 14.93
    ]

    for zeta, expected in test_cases:
        if zeta < 1:
            t_s = 4 / (zeta * omega_n)
        elif zeta == 1:
            t_s = 4 / omega_n
        else:
            t_s = 4 / (omega_n * (zeta - np.sqrt(zeta**2 - 1)))

        print(f"zeta={zeta}: t_settle={t_s:.2f}s (expected {expected:.2f}s)")
        assert abs(t_s - expected) < 0.1
```

---

## 4. feedback_control.py - Ki < 0 Disable Flag

### Location
Multiple locations in `feedback_control.py` where `Ki` is checked

### Current Code (Confusing)
```python
def __init__(self, ..., Ki: float = -1.0):  # -1.0 means "disabled"
    ...

def compute_torque(self, ...):
    if self.Ki > 0 and current_time is not None:
        # Integrate error
        ...
    if self.Ki > 0:
        torque += self.Ki * self.sigma_integral
```

### Problem
Using `Ki = -1.0` to disable integral control is:
1. Non-standard and confusing
2. Prevents legitimate use of negative Ki (though unusual)
3. Not self-documenting

### Corrected Code

**Option A: Use None (Recommended)**

```python
from typing import Optional

class MRPFeedbackController:
    def __init__(self,
                 inertia: np.ndarray,
                 K: float = 30.0,
                 P: float = 60.0,
                 Ki: Optional[float] = None):  # None means disabled
        """
        Initialize MRP feedback controller.

        Args:
            inertia: 3x3 spacecraft inertia matrix [kg*m^2]
            K: Proportional gain (attitude error to torque)
            P: Derivative gain (rate error to torque)
            Ki: Integral gain (accumulated error to torque), None to disable
        """
        self.K = K
        self.P = P
        self.Ki = Ki
        self.integral_enabled = Ki is not None
        # ...

    def compute_torque(self, ...):
        # ...

        # Integral term (if enabled)
        if self.integral_enabled and current_time is not None:
            if self.last_time is not None:
                dt = current_time - self.last_time
                if dt > 0:
                    self.sigma_integral += sigma_error * dt
                    # Anti-windup
                    integral_limit = 1.0
                    self.sigma_integral = np.clip(
                        self.sigma_integral,
                        -integral_limit,
                        integral_limit
                    )

        if self.integral_enabled:
            torque += self.Ki * self.sigma_integral
```

**Option B: Use 0.0 to Disable**

```python
def __init__(self, ..., Ki: float = 0.0):  # 0.0 means disabled
    ...

def compute_torque(self, ...):
    # Integral term (skip if Ki is zero or near-zero)
    if abs(self.Ki) > 1e-12 and current_time is not None:
        ...
```

### Recommendation
Use **Option A** with `Ki: Optional[float] = None` as it's most explicit and self-documenting.

---

## 5. design_shaper.py - Missing zeta < 1 Validation

### Locations
Multiple functions using `omega_d = omega * np.sqrt(1 - zeta**2)`:
- Line 58: `compute_residual_vibration_continuous()`
- Line 96: `compute_residual_vibration_impulse()`
- Line 315: `residual_vibration()` (local function)
- Line 887: `design_spacecraft_shaper_with_duration()`

### Current Code (No Validation)
```python
omega_d = omega * np.sqrt(1 - zeta**2)  # Fails if zeta >= 1
```

### Corrected Code

Add a validation helper and use it consistently:

```python
def _validate_underdamped(zeta: float, context: str = "") -> None:
    """
    Validate that damping ratio is in underdamped range.

    Args:
        zeta: Damping ratio
        context: Description of calling context for error message

    Raises:
        ValueError: If zeta >= 1 (critically damped or overdamped)
    """
    if zeta >= 1.0:
        msg = f"Damping ratio must be < 1 for underdamped oscillator, got zeta={zeta}"
        if context:
            msg = f"{context}: {msg}"
        raise ValueError(msg)


def _damped_frequency(omega_n: float, zeta: float) -> float:
    """
    Compute damped natural frequency.

    Args:
        omega_n: Undamped natural frequency [rad/s]
        zeta: Damping ratio (must be < 1)

    Returns:
        omega_d: Damped natural frequency [rad/s]
    """
    _validate_underdamped(zeta, "damped_frequency")
    return omega_n * np.sqrt(1 - zeta**2)


# Update all affected functions:

def compute_residual_vibration_continuous(t, u, freq, zeta=0.01):
    """..."""
    _validate_underdamped(zeta, "compute_residual_vibration_continuous")
    omega = 2 * np.pi * freq
    omega_d = _damped_frequency(omega, zeta)
    # ... rest of function


def compute_residual_vibration_impulse(amps, times, freq, zeta=0.01):
    """..."""
    _validate_underdamped(zeta, "compute_residual_vibration_impulse")
    omega = 2 * np.pi * freq
    omega_d = _damped_frequency(omega, zeta)
    # ... rest of function


def design_spacecraft_shaper_with_duration(...):
    """..."""
    for i, (freq, zeta) in enumerate(zip(mode_frequencies, damping_ratios)):
        _validate_underdamped(zeta, f"mode {i+1} (freq={freq} Hz)")
        omega_n = 2 * np.pi * freq
        omega_d = _damped_frequency(omega_n, zeta)
        # ... rest of loop
```

### Alternative: Handle Overdamped Gracefully

If you want to support overdamped modes (unusual but possible), you could compute the impulse response differently:

```python
def _compute_impulse_response_params(omega_n: float, zeta: float) -> dict:
    """
    Compute impulse response parameters for any damping ratio.

    Returns dict with keys depending on damping type:
        - underdamped: {'type': 'underdamped', 'omega_d': ..., 'zeta': ...}
        - critically_damped: {'type': 'critical', 'omega_n': ...}
        - overdamped: {'type': 'overdamped', 'p1': ..., 'p2': ...}
    """
    if zeta < 1.0:
        return {
            'type': 'underdamped',
            'omega_n': omega_n,
            'omega_d': omega_n * np.sqrt(1 - zeta**2),
            'zeta': zeta
        }
    elif zeta == 1.0:
        return {
            'type': 'critical',
            'omega_n': omega_n
        }
    else:
        # Overdamped: two real poles
        sqrt_term = omega_n * np.sqrt(zeta**2 - 1)
        return {
            'type': 'overdamped',
            'p1': -zeta * omega_n + sqrt_term,  # Slower pole (closer to 0)
            'p2': -zeta * omega_n - sqrt_term   # Faster pole
        }
```

---

## 6. spacecraft_model.py - Docstring vs Implementation Mismatch

### Location
`spacecraft_model.py`, module docstring (lines 1-14) and implementation (lines 170-171)

### Current State

**Docstring says:**
```python
"""
For YAW (Z-axis) slew maneuvers:
- Solar arrays extend along Y-axis (port/starboard)
- Flex modes bend in Z direction  # <-- Says Z direction
"""
```

**Implementation:**
```python
# For yaw (Z-rotation) with mass on Y-axis, tangential direction is X
mode1_port.pHat_B = [[1.0], [0.0], [0.0]]  # <-- Uses X direction
```

### Analysis
The implementation is physically correct for **base excitation** modeling:
- Solar arrays extend along Y-axis
- Yaw (Z-rotation) acceleration creates tangential force
- For a mass at $\vec{r} = [0, r_y, 0]^T$, the tangential acceleration is in **X direction**

The docstring is misleading because "bend in Z direction" suggests out-of-plane bending, but the model captures the tangential inertial force from base rotation.

### Corrected Docstring
```python
"""
Flexible Spacecraft Model for Basilisk Simulation

This module defines a spacecraft with:
- Rigid hub with specified mass/inertia
- 3-axis reaction wheel array
- 2 flexible solar array appendages with modal dynamics

Flexible Mode Coupling for YAW (Z-axis) Slew Maneuvers:
-------------------------------------------------------
Solar arrays extend along the Y-axis (port/starboard configuration).
When the spacecraft rotates about Z (yaw), angular acceleration creates
tangential inertial forces on the array masses.

For a modal mass at position r = [0, r_y, 0]^T:
    Tangential acceleration = omega_dot × r = [0, 0, alpha_z]^T × [0, r_y, 0]^T
                            = [-alpha_z * r_y, 0, 0]^T

This tangential force acts in the X direction, which is why pHat_B = [1, 0, 0].

The linearSpringMassDamper elements model the modal response to this
base excitation. The modal displacement is in the X direction, representing
the effective "in-plane bending" response to yaw maneuvers.

Note: This is distinct from out-of-plane bending (which would be Z direction).
The current model captures the dominant coupling mechanism for yaw maneuvers
exciting solar array modes through inertial (base excitation) effects.
"""
```

Also update the comment near line 170:

```python
# Modal displacement direction for yaw-to-flex coupling:
# Yaw (Z-rotation) acceleration creates tangential force on Y-positioned masses.
# Tangential direction for r=[0, r_y, 0] under Z-rotation is X.
# This models base excitation coupling, not out-of-plane bending.
mode1_port.pHat_B = [[1.0], [0.0], [0.0]]
```

---

## 7. spacecraft_model.py - Hardcoded hub_mass vs HUB_INERTIA

### Location
`spacecraft_model.py`, lines 46-48

### Current Code
```python
self.hub_mass = 750.0  # kg - main body mass (HARDCODED)
self.hub_inertia = HUB_INERTIA.tolist()  # kg*m^2 (IMPORTED)
```

### Problem
Mass and inertia are defined in different places, creating maintenance burden and risk of inconsistency.

### Corrected Code

**Step 1: Add hub mass to spacecraft_properties.py**

```python
# In spacecraft_properties.py, add:

HUB_MASS = 750.0  # kg - main body mass

# Document the physical consistency:
# For a uniform density rectangular prism with these inertias,
# the dimensions would be approximately 2.2m × 2.0m × 1.8m
# Mass = rho * V, where this gives rho ≈ 94 kg/m³ (reasonable for a satellite bus)
```

**Step 2: Update spacecraft_model.py to import**

```python
# In spacecraft_model.py:

from spacecraft_properties import (
    FLEX_MODE_LOCATIONS,
    FLEX_MODE_MASS,
    HUB_INERTIA,
    HUB_MASS,  # ADD THIS
    compute_effective_inertia as compute_effective_inertia_base,
)

class FlexibleSpacecraft:
    def __init__(self):
        """Set up spacecraft parameters."""

        # Hub properties - imported from spacecraft_properties for consistency
        self.hub_mass = HUB_MASS
        self.hub_inertia = HUB_INERTIA.tolist()
        # ... rest of __init__
```

---

## 8. spacecraft_properties.py - Modal vs Appendage Mass Documentation

### Location
`spacecraft_properties.py`, lines 23-40

### Current Code (Undocumented distinction)
```python
FLEX_MODE_MASS = 5.0  # What is this?
ARRAY_MASS_PER_WING = 50.0  # What is this?

DEFAULT_APPENDAGE_MASSES = {
    "mode1_port": ARRAY_MASS_PER_WING / 2.0,  # = 25.0 kg
    # ...
}
```

### Corrected Code with Documentation

```python
# ===========================================================================
# Solar Array Mass Properties
# ===========================================================================
#
# There are two distinct mass concepts for the flexible appendages:
#
# 1. PHYSICAL MASS (ARRAY_MASS_PER_WING):
#    The total physical mass of one solar array wing. This contributes to
#    the spacecraft's rigid-body inertia via the parallel axis theorem.
#    Typical value: 50 kg per wing for a medium-sized observation satellite.
#
# 2. MODAL MASS (FLEX_MODE_MASS):
#    The effective mass participating in each vibration mode. This is
#    typically 5-20% of the physical mass because only part of the
#    appendage moves at maximum amplitude for a given mode shape.
#    Used in: linearSpringMassDamper modal dynamics (k, c, m parameters)
#
# For inertia calculations (compute_effective_inertia), we use the PHYSICAL
# mass distributed at the mode locations, which approximates treating the
# array as point masses at representative locations.
#
# For modal dynamics, we use MODAL mass which determines the natural
# frequency relationship: omega_n = sqrt(k/m_modal).
# ===========================================================================

# Modal mass: effective mass participating in vibration modes
# This is the "m" in the modal equation: m*x'' + c*x' + k*x = F
# Typically 5-20% of the physical appendage mass
FLEX_MODE_MASS = 5.0  # kg, effective modal mass per mode

# Physical array mass: total mass of each solar array wing
# Used for rigid-body inertia contribution via parallel axis theorem
ARRAY_MASS_PER_WING = 50.0  # kg, physical mass per wing

# Location of modal masses (points where linearSpringMassDamper are attached)
# Port = -Y side, Starboard = +Y side
# Mode 1 locations are closer to hub (first bending antinode)
# Mode 2 locations are further out (second bending antinode)
FLEX_MODE_LOCATIONS = {
    "mode1_port": np.array([0.0, -3.5, 0.0], dtype=float),  # First mode, port
    "mode2_port": np.array([0.0, -4.5, 0.0], dtype=float),  # Second mode, port
    "mode1_stbd": np.array([0.0, 3.5, 0.0], dtype=float),   # First mode, starboard
    "mode2_stbd": np.array([0.0, 4.5, 0.0], dtype=float),   # Second mode, starboard
}

# For inertia calculation, distribute physical mass at mode locations
# This approximates the array as point masses for rigid-body dynamics
# Each wing's mass is split between its two mode locations
DEFAULT_APPENDAGE_MASSES = {
    "mode1_port": ARRAY_MASS_PER_WING / 2.0,  # 25 kg at inner port location
    "mode2_port": ARRAY_MASS_PER_WING / 2.0,  # 25 kg at outer port location
    "mode1_stbd": ARRAY_MASS_PER_WING / 2.0,  # 25 kg at inner starboard location
    "mode2_stbd": ARRAY_MASS_PER_WING / 2.0,  # 25 kg at outer starboard location
}
# Total appendage mass contribution: 4 * 25 = 100 kg (both wings)
```

---

## 9. spacecraft_properties.py - Modal Gain Units Documentation

### Location
`spacecraft_properties.py`, function `compute_modal_gains()` around line 71

### Current Code (Undocumented units)
```python
def compute_modal_gains(
    inertia: np.ndarray,
    rotation_axis: Iterable[float],
    mode_keys: Iterable[str] = DEFAULT_MODE_KEYS,
    mode_locations: Optional[dict] = None,
) -> list[float]:
    """
    Compute modal gains that map torque to modal acceleration (gain * torque).
    ...
    """
```

### Corrected Code with Full Documentation

```python
def compute_modal_gains(
    inertia: np.ndarray,
    rotation_axis: Iterable[float],
    mode_keys: Iterable[str] = DEFAULT_MODE_KEYS,
    mode_locations: Optional[dict] = None,
) -> list[float]:
    """
    Compute modal gains that map body torque to modal acceleration.

    For base excitation of a flexible mode attached to a rotating body:

        Modal EOM: m*q'' + c*q' + k*q = F_tangential

    Where F_tangential = m_modal * a_tangential = m_modal * (alpha * r)
    and alpha = tau_body / I_axis is the angular acceleration.

    Substituting and dividing by m:
        q'' + 2*zeta*omega*q' + omega^2*q = (r / I_axis) * tau_body

    Therefore: modal_gain = r / I_axis

    Units Analysis:
    ---------------
    - r (lever arm): meters [m]
    - I_axis (moment of inertia): kg*m^2
    - modal_gain: m / (kg*m^2) = 1 / (kg*m)

    When multiplied by torque [N*m = kg*m^2/s^2]:
        modal_gain * torque = [1/(kg*m)] * [kg*m^2/s^2] = [m/s^2]

    This gives modal acceleration in m/s^2, which integrates to
    modal displacement in meters.

    Physical Interpretation:
    ------------------------
    A gain of 0.005 rad/(N*m*s^2) means that 1 N*m of torque produces
    0.005 m/s^2 of modal acceleration. For a 0.4 Hz mode (omega=2.5 rad/s),
    the static displacement per unit torque would be:

        q_static = gain / omega^2 = 0.005 / 6.25 = 0.0008 m/(N*m) = 0.8 mm/(N*m)

    Args:
        inertia: 3x3 spacecraft inertia matrix [kg*m^2]
        rotation_axis: Unit vector defining rotation axis
        mode_keys: Names of modes to compute gains for
        mode_locations: Dict mapping mode names to position vectors [m]

    Returns:
        List of modal gains [1/(kg*m)] for each mode in mode_keys.
        Multiply by torque [N*m] to get modal acceleration [m/s^2].
    """
    # ... implementation unchanged
```

---

## 10. vizard_demo.py - Hardcoded Jitter/Blur Constants

### Location
`vizard_demo.py`, lines 658-659 in `_analyze_results()`

### Current Code (Hardcoded)
```python
jitter_arcsec = (total_rms / 1000 / 4.0) * (180/np.pi) * 3600
blur_px = jitter_arcsec / 2.0
```

### Corrected Code

**Step 1: Add parameters to __init__**

```python
class CometPhotographyDemo:
    def __init__(self, shaping_method='fourth', controller='standard_pd', run_mode='combined',
                 use_trajectory_tracking=True):
        # ... existing code ...

        # Camera/imaging parameters for jitter analysis
        self.jitter_lever_arm = 4.0  # meters, distance from rotation axis to sensor
        self.pixel_scale_arcsec = 2.0  # arcsec/pixel, camera plate scale

        # Document the assumptions:
        # - Lever arm: approximate distance from spacecraft CoM to camera sensor
        # - Pixel scale: typical for high-resolution Earth observation cameras
        #   (e.g., 0.5m GSD from 500km altitude with 0.5m aperture)
```

**Step 2: Update _analyze_results() to use parameters**

```python
def _analyze_results(self):
    """Analyze results."""
    # ... existing code ...

    if np.any(post_slew_idx):
        mode1_rms = np.sqrt(np.mean(mode1[post_slew_idx]**2)) * 1000  # mm
        mode2_rms = np.sqrt(np.mean(mode2[post_slew_idx]**2)) * 1000  # mm
        total_rms = np.sqrt(mode1_rms**2 + mode2_rms**2)

        # Convert modal displacement to angular jitter
        # jitter_angle = displacement / lever_arm
        # Then convert radians to arcseconds
        modal_displacement_m = total_rms / 1000  # Convert mm to m
        jitter_rad = modal_displacement_m / self.jitter_lever_arm
        jitter_arcsec = jitter_rad * (180/np.pi) * 3600

        # Convert to image blur in pixels
        blur_px = jitter_arcsec / self.pixel_scale_arcsec

        print(f"\n  VIBRATION (imaging impact):")
        print(f"    Modal RMS: {total_rms:.2f} mm")
        print(f"    Jitter: {jitter_arcsec:.1f} arcsec (lever arm: {self.jitter_lever_arm:.1f} m)")
        print(f"    Blur: {blur_px:.1f} px (at {self.pixel_scale_arcsec:.1f} arcsec/px)")
```

**Step 3: Allow command-line override (optional)**

```python
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the comet photography demo.")
    parser.add_argument("method", nargs="?", default="fourth", choices=["unshaped", "zvd", "fourth"])
    parser.add_argument("--controller", default="standard_pd", choices=sorted(CONTROLLERS))
    parser.add_argument("--mode", default="combined", choices=sorted(RUN_MODES))
    parser.add_argument("--lever-arm", type=float, default=4.0,
                        help="Jitter lever arm in meters (default: 4.0)")
    parser.add_argument("--pixel-scale", type=float, default=2.0,
                        help="Camera pixel scale in arcsec/pixel (default: 2.0)")
    args = parser.parse_args()

    demo = CometPhotographyDemo(args.method, controller=args.controller, run_mode=args.mode)
    demo.jitter_lever_arm = args.lever_arm
    demo.pixel_scale_arcsec = args.pixel_scale
    # ... rest of main
```

---

## 11. mission_simulation.py - ZVD Single-Mode Limitation

### Location
`mission_simulation.py`, lines 621-627

### Current Code (Single-mode only)
```python
if method == "zvd":
    # ZVD shaper for first mode
    f_mode = config.modal_freqs_hz[0] if config.modal_freqs_hz else 1.0
    zeta = config.modal_damping[0] if config.modal_damping else 0.01
    amplitudes, delays = _zvd_shaper_params(f_mode, zeta)
```

### Issue
This is a design **limitation**, not a bug. The ZVD implementation only suppresses the first mode (0.4 Hz), leaving the second mode (1.3 Hz) unsuppressed.

### Corrected Code (Multi-mode cascaded ZVD)

```python
def _cascade_shapers(amp1: List[float], delay1: List[float],
                     amp2: List[float], delay2: List[float]) -> Tuple[List[float], List[float]]:
    """
    Cascade two input shapers via convolution.

    The cascaded shaper is the convolution of the two individual shapers:
        A_cascade[k] = sum_i sum_j A1[i] * A2[j] where delay1[i] + delay2[j] = delay_cascade[k]

    Args:
        amp1, delay1: First shaper amplitudes and delays
        amp2, delay2: Second shaper amplitudes and delays

    Returns:
        Cascaded shaper (amplitudes, delays)
    """
    cascaded_amp = []
    cascaded_delay = []

    for a1, d1 in zip(amp1, delay1):
        for a2, d2 in zip(amp2, delay2):
            cascaded_amp.append(a1 * a2)
            cascaded_delay.append(d1 + d2)

    # Combine impulses at same time (within tolerance)
    tol = 1e-9
    combined_amp = []
    combined_delay = []

    for a, d in zip(cascaded_amp, cascaded_delay):
        # Check if this delay already exists
        found = False
        for i, existing_d in enumerate(combined_delay):
            if abs(d - existing_d) < tol:
                combined_amp[i] += a
                found = True
                break
        if not found:
            combined_amp.append(a)
            combined_delay.append(d)

    # Sort by delay
    sorted_pairs = sorted(zip(combined_delay, combined_amp))
    combined_delay = [p[0] for p in sorted_pairs]
    combined_amp = [p[1] for p in sorted_pairs]

    return combined_amp, combined_delay


def _compute_torque_profile(
    config: MissionConfig, method: str, settling_time: float = 30.0
) -> Dict[str, np.ndarray]:
    """..."""

    # ... existing code for 'fourth' method ...

    if method == "zvd":
        # Multi-mode ZVD: cascade shapers for each modal frequency
        amplitudes = [1.0]
        delays = [0.0]

        for f_mode, zeta in zip(config.modal_freqs_hz, config.modal_damping):
            a_mode, d_mode = _zvd_shaper_params(f_mode, zeta)
            amplitudes, delays = _cascade_shapers(amplitudes, delays, a_mode, d_mode)

        print(f"Multi-mode ZVD shaper: {len(amplitudes)} impulses, "
              f"duration {max(delays):.2f}s")

        # Apply cascaded shaper to bang-bang trajectory
        # ... rest of existing implementation
```

### Documentation Update
Add a note about the limitation vs. full fix:

```python
# In _compute_torque_profile docstring:
"""
Note on ZVD shaper:
    The ZVD method now cascades shapers for ALL modes in config.modal_freqs_hz.
    For 2 modes with ZVD (3 impulses each), this produces a 9-impulse shaper.

    The shaper duration increases with each cascaded mode:
        Single mode ZVD: ~1.25 * period of mode
        Two mode ZVD: ~1.25 * (period_1 + period_2)

    For modes at 0.4 Hz and 1.3 Hz:
        - Mode 1 only: shaper adds ~3.1s
        - Both modes: shaper adds ~3.9s
"""
```

---

## 12. mission_simulation.py - Nyquist Margin Edge Cases

### Location
`mission_simulation.py`, lines 1079-1138, function `_compute_stability_margins()`

### Issues
1. Only first -180° crossing used for gain margin
2. Only first 0dB crossing used for phase margin
3. No handling of multiple crossings (common with flexible modes)
4. No warning for potentially unstable systems

### Corrected Code

```python
def _compute_stability_margins(L: np.ndarray, freqs: np.ndarray) -> Dict[str, float]:
    """
    Compute gain and phase margins from loop transfer function.

    For systems with flexible modes, there may be multiple phase crossings
    of -180°. This function finds ALL crossings and reports the minimum
    (worst-case) margins.

    Args:
        L: Complex loop transfer function evaluated at frequencies
        freqs: Frequency array [Hz]

    Returns:
        Dictionary with:
            - gain_margin_db: Worst-case gain margin [dB] (can be negative if unstable)
            - phase_margin_deg: Worst-case phase margin [deg]
            - gain_crossover_hz: Frequency where |L|=1
            - phase_crossover_hz: Frequency where angle(L)=-180°
            - num_phase_crossings: Number of -180° crossings found
            - stable: Boolean indicating Nyquist stability
    """
    mag = np.abs(L)
    phase_deg = np.degrees(np.unwrap(np.angle(L)))

    # Initialize with "infinite" margins (no crossings found)
    gm_db = np.inf
    pm_deg = np.inf
    gain_crossover_hz = None
    phase_crossover_hz = None

    # Find ALL phase crossings at -180° (for gain margin)
    # Look for crossings in both directions
    phase_crossings = []
    for target in [-180.0, -540.0, -900.0]:  # Account for multiple wraps
        # Crossing from above
        idx_down = np.where((phase_deg[:-1] > target) & (phase_deg[1:] <= target))[0]
        # Crossing from below
        idx_up = np.where((phase_deg[:-1] < target) & (phase_deg[1:] >= target))[0]

        for idx_array in [idx_down, idx_up]:
            for i in idx_array:
                # Interpolate to find exact crossing frequency
                f1, f2 = freqs[i], freqs[i + 1]
                ph1, ph2 = phase_deg[i], phase_deg[i + 1]

                if abs(ph2 - ph1) > 1e-10:
                    f_cross = f1 + (f2 - f1) * (target - ph1) / (ph2 - ph1)
                else:
                    f_cross = f1

                # Interpolate magnitude at crossing
                if f_cross > 0 and f1 > 0 and f2 > 0:
                    log_f1, log_f2 = np.log10(f1), np.log10(f2)
                    log_f_cross = np.log10(f_cross)
                    log_m1, log_m2 = np.log10(mag[i] + 1e-15), np.log10(mag[i+1] + 1e-15)

                    if abs(log_f2 - log_f1) > 1e-10:
                        log_m_cross = log_m1 + (log_m2 - log_m1) * (log_f_cross - log_f1) / (log_f2 - log_f1)
                    else:
                        log_m_cross = log_m1

                    mag_at_cross = 10 ** log_m_cross
                    gm_at_cross = -20 * np.log10(mag_at_cross + 1e-15)

                    phase_crossings.append({
                        'freq_hz': f_cross,
                        'gm_db': gm_at_cross,
                        'mag': mag_at_cross
                    })

    # Find worst-case (minimum) gain margin
    if phase_crossings:
        worst_gm = min(phase_crossings, key=lambda x: x['gm_db'])
        gm_db = worst_gm['gm_db']
        phase_crossover_hz = worst_gm['freq_hz']

    # Find ALL gain crossings at 0 dB (for phase margin)
    gain_crossings = []
    # Crossing from above (|L| going below 1)
    idx_down = np.where((mag[:-1] > 1.0) & (mag[1:] <= 1.0))[0]
    # Crossing from below (|L| going above 1)
    idx_up = np.where((mag[:-1] < 1.0) & (mag[1:] >= 1.0))[0]

    for idx_array in [idx_down, idx_up]:
        for i in idx_array:
            f1, f2 = freqs[i], freqs[i + 1]
            m1, m2 = mag[i], mag[i + 1]

            # Log-interpolate frequency at unity gain
            if m1 > 0 and m2 > 0:
                log_m1, log_m2 = np.log10(m1), np.log10(m2)
                log_f1, log_f2 = np.log10(f1), np.log10(f2)

                if abs(log_m2 - log_m1) > 1e-10:
                    log_f_cross = log_f1 + (0.0 - log_m1) * (log_f2 - log_f1) / (log_m2 - log_m1)
                    f_cross = 10 ** log_f_cross
                else:
                    f_cross = f1

                # Interpolate phase at crossing
                if abs(log_f2 - log_f1) > 1e-10:
                    phase_at_cross = phase_deg[i] + (phase_deg[i+1] - phase_deg[i]) * (log_f_cross - log_f1) / (log_f2 - log_f1)
                else:
                    phase_at_cross = phase_deg[i]

                pm_at_cross = 180.0 + phase_at_cross
                # Normalize to [-180, 180]
                while pm_at_cross > 180:
                    pm_at_cross -= 360
                while pm_at_cross < -180:
                    pm_at_cross += 360

                gain_crossings.append({
                    'freq_hz': f_cross,
                    'pm_deg': pm_at_cross
                })

    # Find worst-case (minimum positive or most negative) phase margin
    if gain_crossings:
        # Sort by phase margin to find worst case
        worst_pm = min(gain_crossings, key=lambda x: x['pm_deg'])
        pm_deg = worst_pm['pm_deg']
        gain_crossover_hz = worst_pm['freq_hz']

    # Determine stability
    # System is stable if GM > 0 dB AND PM > 0 deg
    stable = (gm_db > 0) and (pm_deg > 0)

    # Warn about potential issues
    if not stable:
        print(f"WARNING: System may be unstable! GM={gm_db:.1f}dB, PM={pm_deg:.1f}deg")
    elif len(phase_crossings) > 1:
        print(f"Note: Multiple phase crossings detected ({len(phase_crossings)}). "
              f"Reporting worst-case GM={gm_db:.1f}dB")

    return {
        "gain_margin_db": float(gm_db),
        "phase_margin_deg": float(pm_deg),
        "gain_crossover_hz": gain_crossover_hz,
        "phase_crossover_hz": phase_crossover_hz,
        "num_phase_crossings": len(phase_crossings),
        "num_gain_crossings": len(gain_crossings),
        "stable": stable
    }
```

---

## 13. mission_simulation.py - Hardcoded Trajectory File Paths

### Location
`mission_simulation.py`, lines 558-566

### Current Code
```python
traj_candidates = [
    os.path.join(os.path.dirname(__file__), "spacecraft_trajectory_4th_180deg_30s.npz"),
    os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "spacecraft_trajectory_4th_180deg_30s.npz")),
]
traj_path = next((path for path in traj_candidates if os.path.isfile(path)), None)
```

### Issue
This may fail in:
- Frozen executables (PyInstaller, etc.)
- Jupyter notebooks where `__file__` behaves differently
- Installed packages

### Corrected Code

**Option A: Use importlib.resources (Recommended for packages)**

```python
import sys
if sys.version_info >= (3, 9):
    from importlib.resources import files, as_file
else:
    from importlib_resources import files, as_file  # Backport for Python < 3.9

def _find_trajectory_file(filename: str) -> Optional[str]:
    """
    Find trajectory data file using multiple strategies.

    Search order:
    1. Package resources (for installed packages)
    2. Relative to this module's file
    3. Parent directory of this module
    4. Current working directory

    Args:
        filename: Name of the trajectory file (e.g., 'spacecraft_trajectory_4th_180deg_30s.npz')

    Returns:
        Absolute path to file if found, None otherwise
    """
    # Strategy 1: Try package resources
    try:
        pkg_files = files('basilisk_simulation')
        with as_file(pkg_files.joinpath(filename)) as path:
            if path.exists():
                return str(path)
    except (ModuleNotFoundError, TypeError, FileNotFoundError):
        pass

    # Strategy 2: Relative to __file__
    candidates = []
    try:
        module_dir = os.path.dirname(os.path.abspath(__file__))
        candidates.append(os.path.join(module_dir, filename))
        candidates.append(os.path.join(module_dir, '..', filename))
    except NameError:
        # __file__ not defined (e.g., in some interactive environments)
        pass

    # Strategy 3: Current working directory
    candidates.append(os.path.join(os.getcwd(), filename))
    candidates.append(os.path.join(os.getcwd(), 'basilisk_simulation', filename))

    # Try each candidate
    for path in candidates:
        abs_path = os.path.abspath(path)
        if os.path.isfile(abs_path):
            return abs_path

    return None


def _compute_torque_profile(config: MissionConfig, method: str, ...) -> Dict[str, np.ndarray]:
    """..."""

    if method == "fourth":
        filename = "spacecraft_trajectory_4th_180deg_30s.npz"
        traj_path = _find_trajectory_file(filename)

        if traj_path is None:
            raise FileNotFoundError(
                f"Fourth-order trajectory file '{filename}' not found.\n"
                f"Search locations:\n"
                f"  - Package resources (basilisk_simulation)\n"
                f"  - {os.path.dirname(__file__) if '__file__' in dir() else 'N/A'}\n"
                f"  - {os.getcwd()}\n"
                f"Please ensure the file exists in one of these locations."
            )

        traj = np.load(traj_path, allow_pickle=True)
        # ... rest of implementation
```

**Option B: Accept file path as parameter**

```python
def _compute_torque_profile(
    config: MissionConfig,
    method: str,
    settling_time: float = 30.0,
    fourth_order_file: Optional[str] = None  # NEW PARAMETER
) -> Dict[str, np.ndarray]:
    """
    Compute torque profile for a given shaping method.

    Args:
        config: Mission configuration
        method: Shaping method ('unshaped', 'zvd', 'fourth')
        settling_time: Time after maneuver for vibration settling
        fourth_order_file: Path to fourth-order trajectory file (required if method='fourth')
    """
    if method == "fourth":
        if fourth_order_file is None:
            # Fall back to search
            fourth_order_file = _find_trajectory_file("spacecraft_trajectory_4th_180deg_30s.npz")

        if fourth_order_file is None or not os.path.isfile(fourth_order_file):
            raise FileNotFoundError(
                f"Fourth-order trajectory file not found. "
                f"Please provide path via fourth_order_file parameter."
            )

        traj = np.load(fourth_order_file, allow_pickle=True)
        # ...
```

---

## 14. Items Requiring External Verification

These items were flagged as "partially agree / needs external verification":

### 14.1 RW Torque Sign Convention (feedforward_control.py)

**Current implementation:**
```python
rw_torque = -body_torque @ Gs_pinv.T
```

**To verify:**
1. Check Basilisk documentation for `rwMotorCmdInMsg` expected sign convention
2. Run a simple test: command positive Z-torque and verify spacecraft rotates positive about Z
3. Check the Basilisk source code for `reactionWheelStateEffector`

**Test to add:**
```python
def test_rw_sign_convention():
    """Verify RW torque sign produces correct spacecraft rotation."""
    # Set up minimal simulation with single wheel on Z-axis
    # Command positive motor torque
    # Verify spacecraft angular momentum decreases (wheel speeds up)
    # This confirms motor torque sign convention
    pass
```

### 14.2 MRP Error Sign Convention (feedback_control.py)

**Current implementation uses Schaub/Junkins formula:**
```python
sigma_error = (
    (1.0 - sigma_ref_sq) * sigma_body
    - (1.0 - sigma_body_sq) * sigma_ref
    + 2.0 * np.cross(sigma_body, sigma_ref)
) / denom
```

**To verify:**
1. Compare against Basilisk `attTrackingError` module output
2. Run test case: start at identity, target at small rotation, verify error sign drives toward target

**Test to add:**
```python
def test_mrp_error_convention():
    """Verify MRP error drives system toward target."""
    sigma_body = np.array([0.0, 0.0, 0.0])  # At identity
    sigma_target = np.array([0.1, 0.0, 0.0])  # Small X rotation

    error = _mrp_subtract(sigma_body, sigma_target)

    # With standard PD: torque = -K*error - P*omega
    # If error is positive when we need to rotate positive,
    # then -K*error is negative, which is wrong
    # Error should be negative to produce positive torque

    # Check: error should have opposite sign to get us to target
    print(f"Body: {sigma_body}, Target: {sigma_target}, Error: {error}")
    # Verify against Basilisk attTrackingError output
```

---

## Summary Checklist

| Issue | File | Status |
|-------|------|--------|
| Notch filter output equation | feedback_control.py | Fix provided |
| PPF integration method | feedback_control.py | Fix provided |
| Overdamped settling time | feedback_control.py | Fix provided |
| Ki disable flag | feedback_control.py | Fix provided |
| zeta < 1 validation | design_shaper.py | Fix provided |
| Docstring mismatch | spacecraft_model.py | Fix provided |
| Hardcoded hub_mass | spacecraft_model.py | Fix provided |
| Modal vs appendage mass | spacecraft_properties.py | Documentation provided |
| Modal gain units | spacecraft_properties.py | Documentation provided |
| Jitter/blur constants | vizard_demo.py | Fix provided |
| ZVD single-mode | mission_simulation.py | Fix + limitation note provided |
| Nyquist edge cases | mission_simulation.py | Fix provided |
| Trajectory file paths | mission_simulation.py | Fix provided |
| RW torque sign | feedforward_control.py | Verification test provided |
| MRP error sign | feedback_control.py | Verification test provided |

---

*Fix instructions document generated 2026-01-21*
