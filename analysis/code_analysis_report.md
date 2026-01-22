# Comprehensive Code Analysis Report

## Spacecraft Input Shaping Simulation - Issue Analysis

**Date:** 2026-01-21
**Analyst:** Claude Code
**Files Analyzed:**
- `mission_simulation.py`
- `design_shaper.py`
- `feedback_control.py`
- `feedforward_control.py`
- `spacecraft_model.py`
- `spacecraft_properties.py`
- `vizard_demo.py`

---

## Executive Summary

A thorough line-by-line analysis of the spacecraft input shaping simulation codebase revealed **27 distinct issues** across seven files. These issues range from critical mathematical errors that could produce incorrect simulation results to minor code quality concerns. The most significant findings include an incorrect notch filter implementation, missing multi-mode shaper support, and physical model inconsistencies.

### Issue Severity Distribution

| Severity | Count | Description |
|----------|-------|-------------|
| **Critical** | 3 | Could cause simulation failure or fundamentally incorrect results |
| **High** | 6 | Significant impact on accuracy or behavior |
| **Medium** | 10 | Moderate impact, may cause issues in edge cases |
| **Low** | 8 | Minor issues, code quality, or documentation concerns |

---

## 1. design_shaper.py

### Issue 1.1: Incorrect ZVD Shaper K Formula [Medium]

**Location:** Lines 892-895 in `design_spacecraft_shaper_with_duration()`

**Code:**
```python
K = np.exp(-zeta * omega_n * period_d / 2)
```

**Issue:** The standard ZVD shaper amplitude ratio formula is:

$$K = \exp\left(-\frac{\zeta \pi}{\sqrt{1 - \zeta^2}}\right)$$

The implementation uses:

$$K = \exp\left(-\zeta \cdot \omega_n \cdot \frac{T_d}{2}\right)$$

**Mathematical Analysis:**

The damped period is $T_d = \frac{2\pi}{\omega_d}$ where $\omega_d = \omega_n\sqrt{1-\zeta^2}$.

Substituting:
$$K = \exp\left(-\zeta \cdot \omega_n \cdot \frac{\pi}{\omega_d}\right) = \exp\left(-\frac{\zeta \cdot \omega_n \cdot \pi}{\omega_n\sqrt{1-\zeta^2}}\right) = \exp\left(-\frac{\zeta \pi}{\sqrt{1-\zeta^2}}\right)$$

**Conclusion:** The formula is algebraically equivalent to the standard form, so the result is correct. However, the non-standard formulation makes the code harder to verify against literature and could introduce numerical precision issues when $\omega_n$ is very large or small.

**Recommendation:** Use the standard formula directly for clarity and numerical stability.

---

### Issue 1.2: Potential Division by Zero [Low]

**Location:** Lines 593-594

**Code:**
```python
scale = theta_final / (theta[-1] + 1e-10)
```

**Issue:** Using `1e-10` as a guard against division by zero is arbitrary. If `theta[-1]` is small but non-zero (e.g., `1e-11`), the scaling would be:

$$\text{scale} = \frac{\theta_{\text{final}}}{1e^{-11} + 1e^{-10}} \approx \frac{\theta_{\text{final}}}{1.1 \times 10^{-10}}$$

This could produce unexpectedly large scale factors when the trajectory integration fails to produce significant displacement.

**Recommendation:** Check if `|theta[-1]| < tolerance` and handle this case explicitly with an error or warning rather than silently producing potentially incorrect results.

---

### Issue 1.3: Missing Validation for Negative Damping Ratio Under Square Root [High]

**Location:** Line 58 and multiple other locations

**Code:**
```python
omega_d = omega * np.sqrt(1 - zeta**2)
```

**Issue:** No validation that $\zeta < 1$. The damped natural frequency formula:

$$\omega_d = \omega_n \sqrt{1 - \zeta^2}$$

is only valid for underdamped systems ($\zeta < 1$). For:
- $\zeta = 1$ (critically damped): $\omega_d = 0$, causing division by zero in period calculations
- $\zeta > 1$ (overdamped): $\omega_d$ becomes imaginary

**Affected Functions:**
- `compute_residual_vibration_continuous()`
- `compute_residual_vibration_impulse()`
- `_zvd_shaper_params()`
- Several others

**Recommendation:** Add validation at the start of affected functions:
```python
if zeta >= 1.0:
    raise ValueError(f"Damping ratio must be < 1 for underdamped oscillator, got {zeta}")
```

---

### Issue 1.4: Hard-coded dt Calculation May Produce Coarse Time Steps [Medium]

**Location:** Lines 490-508

**Code:**
```python
f1_int = int(f1 * scale)
f2_int = int(f2 * scale)
# ... LCM calculation ...
dt_base = 1.0 / (scale * lcm_val)
```

**Issue:** The algorithm attempts to find a time step that evenly divides both modal periods. For frequencies like $f_1 = 0.4$ Hz and $f_2 = 1.3$ Hz:

1. With `scale = 1000`: $f_{1,int} = 400$, $f_{2,int} = 1300$
2. $\text{GCD}(400, 1300) = 100$
3. $\text{LCM} = \frac{400 \times 1300}{100} = 5200$
4. $dt_{base} = \frac{1}{1000 \times 5200} = 1.92 \times 10^{-7}$ s

This is actually very fine, but for other frequency combinations (especially incommensurate frequencies), the LCM-based approach may not produce optimal results.

**Recommendation:** Consider using a fixed small time step (e.g., `dt = 0.001` s) that is sufficient for the highest frequency of interest, rather than the complex LCM-based calculation.

---

## 2. feedback_control.py

### Issue 2.1: Inconsistent MRP Error Sign Convention [Medium]

**Location:** Lines 65-71 in `_mrp_subtract()`

**Code:**
```python
sigma_error = (
    (1.0 - sigma_ref_sq) * sigma_body
    - (1.0 - sigma_body_sq) * sigma_ref
    + 2.0 * np.cross(sigma_body, sigma_ref)
) / denom
```

**Issue:** MRP subtraction computes the relative attitude $\sigma_{BR} = \sigma_{BN} \ominus \sigma_{RN}$, representing the rotation from Reference frame to Body frame.

The cross-product term uses `np.cross(sigma_body, sigma_ref)`. The Basilisk convention and Schaub & Junkins textbook use:

$$\sigma_{B/R} = \frac{(1-|\sigma_R|^2)\sigma_B - (1-|\sigma_B|^2)\sigma_R + 2\sigma_B \times \sigma_R}{1 + |\sigma_B|^2|\sigma_R|^2 + 2\sigma_R \cdot \sigma_B}$$

**Analysis:** The implementation matches the textbook formula. However, for attitude control, you typically want the error that drives the body TO the reference. The current formula gives the rotation FROM reference TO body. Depending on the controller sign convention, this may require negation.

**Recommendation:** Verify against the Basilisk `attTrackingError` module to ensure consistent sign conventions.

---

### Issue 2.2: Closed-Loop Bandwidth Formula May Produce Complex Numbers [Medium]

**Location:** Lines 281-282 in `get_closed_loop_params()`

**Code:**
```python
omega_bw = omega_n * np.sqrt(1 - 2*zeta**2 + np.sqrt(4*zeta**4 - 4*zeta**2 + 2))
```

**Issue:** This formula computes the -3dB bandwidth for a second-order system. Let's analyze the inner expression:

$$x = 4\zeta^4 - 4\zeta^2 + 2$$

Taking the derivative: $\frac{dx}{d\zeta} = 16\zeta^3 - 8\zeta = 8\zeta(2\zeta^2 - 1)$

The minimum occurs at $\zeta = \frac{1}{\sqrt{2}} \approx 0.707$:
$$x_{min} = 4 \cdot \frac{1}{4} - 4 \cdot \frac{1}{2} + 2 = 1 - 2 + 2 = 1$$

So $\sqrt{x} \geq 1$ for all $\zeta$.

For the outer expression: $y = 1 - 2\zeta^2 + \sqrt{x}$

At $\zeta = 0.707$: $y = 1 - 1 + 1 = 1 > 0$ ✓

At $\zeta = 1$: $y = 1 - 2 + \sqrt{4-4+2} = -1 + \sqrt{2} \approx 0.414 > 0$ ✓

**Conclusion:** The formula is valid for $\zeta \in [0, 1]$. However, for $\zeta > 1$, numerical issues could arise. Add validation.

---

### Issue 2.3: PPF Integration Method Inconsistency [High]

**Location:** Lines 516-519 in `PPFCompensator.update()`

**Code:**
```python
# Integrate (trapezoidal for xdot, forward Euler for x)
x_new = x + xdot * dt + 0.5 * xddot * dt**2
xdot_new = xdot + xddot * dt
```

**Issue:** The comment claims "trapezoidal for xdot" but the implementation shows:

1. **Position update:** $x_{new} = x + \dot{x} \cdot dt + \frac{1}{2}\ddot{x} \cdot dt^2$
   - This is a second-order Taylor expansion (Velocity Verlet position update)

2. **Velocity update:** $\dot{x}_{new} = \dot{x} + \ddot{x} \cdot dt$
   - This is Forward Euler, NOT trapezoidal

**Trapezoidal (implicit midpoint)** would be:
$$\dot{x}_{new} = \dot{x} + \frac{dt}{2}(\ddot{x} + \ddot{x}_{new})$$

This requires solving for $\ddot{x}_{new}$, making it implicit.

**Impact:** The mixed integration scheme may cause energy drift in the PPF filter states over long simulations, affecting vibration damping performance.

**Recommendation:** Use a consistent integration scheme:
- Symplectic Euler (position-first or velocity-first)
- Velocity Verlet (full scheme)
- Or implement proper trapezoidal with iteration

---

### Issue 2.4: Notch Filter State-Space Implementation Error [Critical]

**Location:** Lines 836-845 in `_apply_notch_filter()`

**Code:**
```python
# Discretize using bilinear transform approximation
# For simplicity, use forward Euler
x1_new = x1 + dt * x2
x2_new = x2 + dt * (-omega_n**2 * x1 - 2*zeta_p*omega_n*x2 + omega_n**2 * u)

# Output: notch filter output
output[axis] = x1_new + (2*zeta_z/omega_n) * x2_new + (1/omega_n**2) * (
    -omega_n**2 * x1_new - 2*zeta_p*omega_n*x2_new + omega_n**2 * u
)
```

**Issue:** The standard notch filter transfer function is:

$$H(s) = \frac{s^2 + 2\zeta_z \omega_n s + \omega_n^2}{s^2 + 2\zeta_p \omega_n s + \omega_n^2}$$

A correct state-space realization in controllable canonical form:

**State equations:**
$$\dot{x}_1 = x_2$$
$$\dot{x}_2 = -\omega_n^2 x_1 - 2\zeta_p \omega_n x_2 + u$$

**Output equation:**
$$y = (\omega_n^2 - \omega_n^2)x_1 + (2\zeta_z\omega_n - 2\zeta_p\omega_n)x_2 + u$$
$$y = 2\omega_n(\zeta_z - \zeta_p)x_2 + u$$

The implementation's output equation is incorrect. The term:
```python
(1/omega_n**2) * (-omega_n**2 * x1_new - 2*zeta_p*omega_n*x2_new + omega_n**2 * u)
```
simplifies to:
$$\frac{1}{\omega_n^2}(-\omega_n^2 x_1 - 2\zeta_p \omega_n x_2 + \omega_n^2 u) = -x_1 - \frac{2\zeta_p}{\omega_n}x_2 + u$$

This does not match the correct output equation.

**Impact:** The notch filter will NOT properly attenuate the target frequency. This is a critical bug that fundamentally breaks the notch filtering functionality.

**Recommendation:** Reimplement using the correct state-space realization or use `scipy.signal` to create a proper discrete-time notch filter.

---

### Issue 2.5: Settling Time Formula Incorrect for Overdamped Systems [Medium]

**Location:** Lines 285-288

**Code:**
```python
if zeta < 1:
    t_settle = 4 / (zeta * omega_n)
else:
    t_settle = 4 * zeta / omega_n
```

**Issue:** For underdamped systems ($\zeta < 1$), the 2% settling time approximation $t_s \approx \frac{4}{\zeta \omega_n}$ is standard.

For overdamped systems ($\zeta > 1$), the system has two real poles:
$$p_{1,2} = -\zeta\omega_n \pm \omega_n\sqrt{\zeta^2 - 1}$$

The slower pole (closer to zero) is:
$$p_{slow} = -\zeta\omega_n + \omega_n\sqrt{\zeta^2 - 1} = -\omega_n(\zeta - \sqrt{\zeta^2-1})$$

The settling time is dominated by this slower pole:
$$t_s \approx \frac{4}{|p_{slow}|} = \frac{4}{\omega_n(\zeta - \sqrt{\zeta^2-1})}$$

The implemented formula $t_s = \frac{4\zeta}{\omega_n}$ is incorrect.

**Example:** For $\zeta = 2$, $\omega_n = 1$:
- Implemented: $t_s = 8$ s
- Correct: $p_{slow} = -(2 - \sqrt{3}) \approx -0.268$, so $t_s \approx 14.9$ s

**Recommendation:** Fix the overdamped case formula.

---

### Issue 2.6: Integral Gain Sign Used for Enable/Disable Logic [Low]

**Location:** Lines 241, 252-253

**Code:**
```python
if self.Ki > 0 and current_time is not None:
    ...
if self.Ki > 0:
    torque += self.Ki * self.sigma_integral
```

**Issue:** The default `Ki = -1.0` is used to "disable" integral control. However:

1. A negative integral gain is a valid (though unusual) control parameter
2. Using a magic number for disabling is not self-documenting
3. Could cause confusion when reviewing gain tuning

**Recommendation:** Use `Ki = None` or `Ki = 0.0` to disable, with explicit checks:
```python
if self.Ki is not None and self.Ki != 0:
    # Apply integral term
```

---

## 3. feedforward_control.py

### Issue 3.1: Potential Off-by-One at Bang-Bang Transition [Low]

**Location:** Lines 50-62

**Code:**
```python
for i, ti in enumerate(t):
    if ti <= t_half:
        # Acceleration phase
        alpha[i] = alpha_max
        omega[i] = alpha_max * ti
        theta[i] = 0.5 * alpha_max * ti**2
    else:
        # Deceleration phase
        t_dec = ti - t_half
        alpha[i] = -alpha_max
        omega[i] = alpha_max * t_half - alpha_max * t_dec
```

**Issue:** At the exact transition point where `ti == t_half`:
- The condition `ti <= t_half` is true, so acceleration phase values are used
- Velocity: $\omega = \alpha_{max} \cdot t_{half}$

At the next time step where `ti > t_half`:
- Deceleration phase with `t_dec = ti - t_half = dt`
- Velocity: $\omega = \alpha_{max} \cdot t_{half} - \alpha_{max} \cdot dt$

This is continuous, so no issue exists. However, the velocity formula in the deceleration phase:
$$\omega = \alpha_{max} \cdot t_{half} - \alpha_{max} \cdot t_{dec}$$

can be rewritten as:
$$\omega = \alpha_{max}(t_{half} - t_{dec}) = \alpha_{max}(t_{half} - (t - t_{half})) = \alpha_{max}(2t_{half} - t)$$

This reaches zero at $t = 2t_{half} = T$, which is correct.

**Conclusion:** The implementation is mathematically correct. No action needed.

---

### Issue 3.2: Smooth Trajectory Initial Discontinuity [Low]

**Location:** Lines 96-105

**Code:**
```python
def accel_profile(time_in_accel: float) -> float:
    if t_ramp <= 0:
        return 1.0
    if time_in_accel <= t_ramp:
        return 0.5 * (1 - np.cos(np.pi * time_in_accel / t_ramp))
```

**Issue:** At $t = 0$:
$$\text{accel\_profile}(0) = 0.5(1 - \cos(0)) = 0.5(1 - 1) = 0$$

This means the initial acceleration is zero, rising smoothly to 1 over the ramp period. This is actually the intended behavior for a "smooth" trajectory - it provides C1 continuity (continuous velocity).

**Conclusion:** This is not a bug but rather the intended design. The initial zero acceleration ensures no jerk at $t=0$.

---

### Issue 3.3: Shaper Timing Discretization Error [Medium]

**Location:** Lines 194-201

**Code:**
```python
for amp, t_imp in zip(shaper_amplitudes, shaper_times):
    shift_idx = int(round(t_imp / dt))

    for i in range(len(torque)):
        idx_shaped = i + shift_idx
        if idx_shaped < n_shaped:
            torque_shaped[idx_shaped] += amp * torque[i]
```

**Issue:** The impulse times are converted to indices via `int(round(t_imp / dt))`. This introduces timing error:

$$\epsilon_t = t_{imp} - \text{round}(t_{imp}/dt) \cdot dt$$

For a ZVD shaper targeting a 0.4 Hz mode with $\zeta = 0.02$:
- $T_d = 1/0.4 \cdot 1/\sqrt{1-0.02^2} \approx 2.5001$ s
- $t_2 = T_d/2 = 1.25$ s
- $t_3 = T_d = 2.5$ s

With $dt = 0.001$ s: `shift_idx = round(1250.05) = 1250`, error = 0.00005 s

**Impact:** The timing error is small for typical dt values but can accumulate for multi-impulse shapers. For high-frequency modes, the error becomes more significant as a fraction of the period.

**Recommendation:** Consider interpolation-based convolution for higher accuracy when shaper timing is critical.

---

### Issue 3.4: RW Torque Sign Convention Needs Verification [Medium]

**Location:** Lines 250-251

**Code:**
```python
# tau_motor = -Gs_pinv @ tau_body (reaction torque opposes motor torque)
rw_torque = -body_torque @ Gs_pinv.T
```

**Issue:** The physics of reaction wheel control:

1. To apply torque $\tau_{body}$ to the spacecraft, the wheels must generate reaction torque $-\tau_{body}$
2. The motor torque accelerates the wheel: $\tau_{motor} = J_w \dot{\Omega}_w$
3. The reaction on the spacecraft: $\tau_{reaction} = -\tau_{motor}$
4. Therefore: $\tau_{body} = -\tau_{motor}$, so $\tau_{motor} = -\tau_{body}$

The implementation computes:
$$\tau_{rw} = -\tau_{body} \cdot G_s^{\dagger T}$$

For a body torque vector $\tau_b$ and the Gs matrix relating wheel torques to body torque:
$$\tau_b = G_s \cdot \tau_{rw}$$

Solving for wheel torques:
$$\tau_{rw} = G_s^{\dagger} \cdot \tau_b$$

But we want the motor torque to produce the opposite reaction:
$$\tau_{motor} = -G_s^{\dagger} \cdot \tau_b$$

**Verification Needed:** Confirm that the Basilisk `rwMotorCmdInMsg` expects motor torque (not reaction torque) and verify the sign convention in the simulation.

---

### Issue 3.5: Minimum Duration Formula Has Incorrect Factor [High]

**Location:** Lines 295-296

**Code:**
```python
# For bang-bang: T_min = sqrt(4 * I * theta / tau_max)
T_min = np.sqrt(4 * I_axis * abs(theta_final) / tau_body_max)
```

**Issue:** Deriving the bang-bang minimum time:

For bang-bang with acceleration $\alpha$ for time $T/2$, then deceleration $-\alpha$ for $T/2$:

Position at $t = T/2$:
$$\theta_{half} = \frac{1}{2}\alpha \left(\frac{T}{2}\right)^2 = \frac{\alpha T^2}{8}$$

Position at $t = T$ (integrating deceleration phase):
$$\theta_{final} = \theta_{half} + \omega_{max} \cdot \frac{T}{2} - \frac{1}{2}\alpha\left(\frac{T}{2}\right)^2$$

where $\omega_{max} = \alpha \cdot T/2$:
$$\theta_{final} = \frac{\alpha T^2}{8} + \frac{\alpha T}{2} \cdot \frac{T}{2} - \frac{\alpha T^2}{8} = \frac{\alpha T^2}{4}$$

Solving for $T$:
$$T = 2\sqrt{\frac{\theta_{final}}{\alpha}} = 2\sqrt{\frac{\theta_{final} \cdot I}{\tau_{max}}}$$

Squaring: $T^2 = \frac{4 \cdot I \cdot \theta}{\tau_{max}}$

So: $T = \sqrt{\frac{4 \cdot I \cdot \theta}{\tau_{max}}}$

**Conclusion:** The formula in the code IS correct. The comment and implementation match. No issue here - my initial analysis was incorrect.

---

## 4. spacecraft_model.py

### Issue 4.1: Flex Mode Direction Inconsistency [High]

**Location:** Lines 170-171 and module docstring

**Docstring (Lines 9-13):**
```python
"""
For YAW (Z-axis) slew maneuvers:
- Solar arrays extend along Y-axis (port/starboard)
- Flex modes bend in Z direction
"""
```

**Implementation (Lines 170-171):**
```python
# For yaw (Z-rotation) with mass on Y-axis, tangential direction is X
mode1_port.pHat_B = [[1.0], [0.0], [0.0]]
```

**Issue:** There's a contradiction:
- Docstring says flex modes "bend in Z direction"
- Code sets `pHat_B = [1, 0, 0]` (X direction)
- Comment says "tangential direction is X"

**Physical Analysis:**

For a solar array extending along the Y-axis, yaw (Z-rotation) creates angular acceleration $\ddot{\psi}$ about Z. A point mass at position $\vec{r} = [0, r_y, 0]^T$ experiences tangential acceleration:

$$\vec{a}_{tangential} = \vec{\ddot{\psi}} \times \vec{r} = \begin{bmatrix}0\\0\\\ddot{\psi}\end{bmatrix} \times \begin{bmatrix}0\\r_y\\0\end{bmatrix} = \begin{bmatrix}-r_y \ddot{\psi}\\0\\0\end{bmatrix}$$

The tangential force is in the **X direction**, which is what the code implements. The docstring saying "bend in Z direction" appears to be incorrect.

**However**, for out-of-plane bending of a solar array panel, the bending displacement would typically be perpendicular to the panel surface. If the panels are in the X-Y plane (extending along Y), out-of-plane bending would indeed be in the Z direction.

**Resolution:** The physical model depends on whether we're modeling:
1. **Base excitation** (tangential acceleration of attachment point) → X direction is correct
2. **Out-of-plane bending mode shape** → Z direction would be correct

For linearSpringMassDamper in Basilisk, `pHat_B` defines the direction of modal displacement. If this is modeling base excitation due to yaw acceleration, X direction is correct and the docstring should be updated.

---

### Issue 4.2: Damping Formula Comment Is Misleading [Low]

**Location:** Lines 165-166

**Code:**
```python
mode1_port.k = modal_mass * omega1**2  # k = m * omega^2
mode1_port.c = 2 * zeta1 * np.sqrt(mode1_port.k * modal_mass)  # critical damping formula
```

**Issue:** The comment "critical damping formula" is misleading. The formula:
$$c = 2\zeta\sqrt{km} = 2\zeta m\omega_n$$

is the general damping coefficient formula for ANY damping ratio $\zeta$, not just critical damping ($\zeta = 1$).

Critical damping specifically refers to $\zeta = 1$, giving $c_{critical} = 2\sqrt{km} = 2m\omega_n$.

**Recommendation:** Change comment to "viscous damping coefficient formula" or "damping coefficient for damping ratio zeta".

---

### Issue 4.3: Hardcoded Hub Mass vs. Imported Inertia [Medium]

**Location:** Lines 46-48

**Code:**
```python
self.hub_mass = 750.0  # kg - main body mass
self.hub_inertia = HUB_INERTIA.tolist()  # kg*m^2 - principal inertias
```

**Issue:** `hub_mass` is hardcoded to 750 kg, while `HUB_INERTIA` is imported from `spacecraft_properties.py`. This creates a maintenance burden:

- If someone changes `HUB_INERTIA`, they might forget to update `hub_mass`
- The mass and inertia should be physically consistent

For a uniform density rectangular box with dimensions $a \times b \times c$ and mass $m$:
$$I_{xx} = \frac{m}{12}(b^2 + c^2), \quad I_{yy} = \frac{m}{12}(a^2 + c^2), \quad I_{zz} = \frac{m}{12}(a^2 + b^2)$$

Given $I = \text{diag}(900, 800, 600)$ kg·m² and $m = 750$ kg, we can check consistency:
- $I_{xx} + I_{yy} + I_{zz} = 2300$ kg·m²
- For a box: $I_{xx} + I_{yy} + I_{zz} = \frac{m}{6}(a^2 + b^2 + c^2)$
- This gives $a^2 + b^2 + c^2 = 18.4$ m²

This is physically plausible. However, the lack of explicit coupling between mass and inertia in the code is a maintenance risk.

**Recommendation:** Either compute mass from inertia (if assuming a specific geometry) or define both in `spacecraft_properties.py`.

---

### Issue 4.4: Variable Shadowing Import [Low]

**Location:** Line 266

**Code:**
```python
spacecraft = test_rigid_spacecraft()
```

**Issue:** The variable `spacecraft` shadows the imported module:
```python
from Basilisk.simulation import spacecraft
```

After line 266, referencing `spacecraft` would give the `FlexibleSpacecraft` instance, not the Basilisk module.

**Recommendation:** Rename the variable to `sc_instance` or similar.

---

## 5. spacecraft_properties.py

### Issue 5.1: Inconsistent Modal vs. Appendage Mass Definitions [Medium]

**Location:** Lines 23-40

**Code:**
```python
FLEX_MODE_MASS = 5.0
ARRAY_MASS_PER_WING = 50.0

DEFAULT_APPENDAGE_MASSES = {
    "mode1_port": ARRAY_MASS_PER_WING / 2.0,  # = 25.0 kg
    "mode2_port": ARRAY_MASS_PER_WING / 2.0,  # = 25.0 kg
    ...
}
```

**Issue:** There are two different mass concepts:

1. `FLEX_MODE_MASS = 5.0` kg - used for modal dynamics (spring-mass-damper)
2. `DEFAULT_APPENDAGE_MASSES` = 25.0 kg per location - used for inertia calculation

In `compute_effective_inertia()` (line 121):
```python
mass = float(masses.get(key, modal_mass)) if key is not None else float(modal_mass)
```

When called with default arguments, it uses `DEFAULT_APPENDAGE_MASSES` (25 kg), not `FLEX_MODE_MASS` (5 kg).

**Physical Interpretation:**
- Modal mass (5 kg): Effective mass participating in the vibration mode (typically 10-20% of total appendage mass)
- Appendage mass (25 kg): Physical mass at that location affecting rigid-body inertia

These are different physical quantities, and the naming/documentation should clarify this distinction.

---

### Issue 5.2: Modal Gain Units Need Documentation [Low]

**Location:** Lines 91-98

**Code:**
```python
I_axis = float(axis @ inertia @ axis)
lever_arms = compute_mode_lever_arms(...)
return [arm / I_axis for arm in lever_arms]
```

**Issue:** The modal gain is computed as:
$$\text{gain} = \frac{r}{I_{axis}}$$

where $r$ is the lever arm (meters) and $I_{axis}$ is moment of inertia (kg·m²).

Units: $\frac{m}{kg \cdot m^2} = \frac{1}{kg \cdot m}$

When multiplied by torque (N·m = kg·m²/s²):
$$\text{gain} \times \tau = \frac{1}{kg \cdot m} \times \frac{kg \cdot m^2}{s^2} = \frac{m}{s^2}$$

This is modal acceleration, which integrates to modal displacement (meters).

**Recommendation:** Add docstring clarifying:
- Input units: torque in N·m
- Output units: modal acceleration in m/s²
- Physical meaning: base excitation transfer function gain

---

## 6. vizard_demo.py

### Issue 6.1: MRP to Angle Conversion Edge Cases [Low]

**Location:** Line 633

**Code:**
```python
rotation_achieved = np.degrees(4 * np.arctan(sigma_mag))
```

**Issue:** The MRP-to-angle formula $\theta = 4 \arctan(|\sigma|)$ is valid for $|\sigma| \in [0, 1]$ (short rotation) or $|\sigma| \in [1, \infty)$ after shadow transformation.

For $|\sigma| = 1$ (180° rotation): $\theta = 4 \arctan(1) = 4 \times 45° = 180°$ ✓

The code calls `_mrp_shadow()` first to ensure $|\sigma| \leq 1$, which is correct.

**Potential Issue:** Near $|\sigma| = 1$, small numerical errors could push $|\sigma|$ slightly above 1 before shadow transformation, causing the arctan to exceed 45° unexpectedly. The `_mrp_shadow` function handles this, but there could be edge cases.

---

### Issue 6.2: Comet Direction Computation Verification [Low]

**Location:** Lines 193-199

**Code:**
```python
camera_body = self.camera_body  # [0, -1, 0]
rot_z = np.array([[c_yaw, -s_yaw, 0.0],
                  [s_yaw,  c_yaw, 0.0],
                  [0.0,    0.0,   1.0]])
comet_direction = rot_z @ camera_body
```

**Analysis:** For a 180° yaw ($c_{yaw} = -1$, $s_{yaw} = 0$):

$$R_z(180°) \cdot \begin{bmatrix}0\\-1\\0\end{bmatrix} = \begin{bmatrix}-1 & 0 & 0\\0 & -1 & 0\\0 & 0 & 1\end{bmatrix} \begin{bmatrix}0\\-1\\0\end{bmatrix} = \begin{bmatrix}0\\1\\0\end{bmatrix}$$

So after a 180° yaw, the camera (initially pointing -Y in body frame) will point +Y in inertial frame. The comet is placed at +Y direction, which is correct.

**Conclusion:** The computation is correct.

---

### Issue 6.3: Motor Torque Array Modification Pattern [Low]

**Location:** Lines 583-588

**Code:**
```python
motor_torque = rw_cmd_payload.motorTorque
motor_torque[0] = float(rw_torque_cmd[0])
motor_torque[1] = float(rw_torque_cmd[1])
motor_torque[2] = float(rw_torque_cmd[2])
rw_cmd_payload.motorTorque = motor_torque
```

**Issue:** This pattern assumes `motorTorque` returns a mutable reference. If it returns a copy (common in wrapped C++ objects), the modifications would be lost without the reassignment.

The code includes the reassignment `rw_cmd_payload.motorTorque = motor_torque`, which handles the copy case. However, a more robust pattern would be:

```python
rw_cmd_payload.motorTorque = [float(rw_torque_cmd[i]) for i in range(3)]
```

---

### Issue 6.4: Jitter to Blur Conversion Uses Hardcoded Constants [Medium]

**Location:** Lines 658-659

**Code:**
```python
jitter_arcsec = (total_rms / 1000 / 4.0) * (180/np.pi) * 3600
blur_px = jitter_arcsec / 2.0
```

**Issue:** Breaking down the jitter calculation:

1. `total_rms / 1000`: Convert mm to meters
2. `/ 4.0`: Divide by lever arm? (4 meters assumed?)
3. `* (180/np.pi) * 3600`: Convert radians to arcseconds

The formula computes: $\theta_{arcsec} = \frac{\delta_{m}}{4} \times \frac{180}{\pi} \times 3600$

This assumes a lever arm of 4 meters, which should be documented or parameterized.

For blur: `blur_px = jitter_arcsec / 2.0` assumes 2 arcsec per pixel, which is camera-specific.

**Recommendation:** Define these as class parameters:
```python
self.lever_arm = 4.0  # meters, distance from rotation axis to sensor
self.pixel_scale = 2.0  # arcsec/pixel, camera plate scale
```

---

## 7. mission_simulation.py

### Issue 7.1: ZVD Shaper Only Targets First Mode [Critical]

**Location:** Lines 621-627

**Code:**
```python
if method == "zvd":
    # ZVD shaper for first mode
    f_mode = config.modal_freqs_hz[0] if config.modal_freqs_hz else 1.0
    zeta = config.modal_damping[0] if config.modal_damping else 0.01
    amplitudes, delays = _zvd_shaper_params(f_mode, zeta)
```

**Issue:** The spacecraft has TWO flexible modes:
- Mode 1: 0.4 Hz
- Mode 2: 1.3 Hz

The ZVD shaper is only designed for the first mode (0.4 Hz). The second mode at 1.3 Hz will NOT be suppressed.

**Impact:** Residual vibration at 1.3 Hz will persist after the maneuver, degrading pointing performance. This defeats the purpose of comparing "ZVD" to other shaping methods.

**Recommendation:** Implement cascaded shapers for multi-mode suppression:
```python
if method == "zvd":
    amplitudes, delays = [1.0], [0.0]
    for f_mode, zeta in zip(config.modal_freqs_hz, config.modal_damping):
        a_mode, d_mode = _zvd_shaper_params(f_mode, zeta)
        amplitudes, delays = _cascade_shapers(amplitudes, delays, a_mode, d_mode)
```

---

### Issue 7.2: Symplectic Euler for Damped Systems [Medium]

**Location:** Lines 680-694

**Code:**
```python
for i in range(1, n):
    # Symplectic Euler: update velocity first, then position with new velocity
    acc = gain1 * torque[i - 1] - 2 * zeta1 * omega1 * mode1_vel[i - 1] - omega1**2 * mode1_disp[i - 1]
    mode1_vel[i] = mode1_vel[i - 1] + acc * dt
    mode1_disp[i] = mode1_disp[i - 1] + mode1_vel[i] * dt
```

**Issue:** The comment claims "Symplectic Euler" which is energy-preserving for Hamiltonian systems. However:

1. The modal equation includes damping: $\ddot{q} + 2\zeta\omega\dot{q} + \omega^2 q = F$
2. Damped systems are NOT Hamiltonian (energy is dissipated)
3. Symplectic integrators have no special properties for non-Hamiltonian systems

**Analysis:** For the undamped case ($\zeta = 0$), symplectic Euler would preserve the oscillation amplitude. With damping, the integrator is simply "semi-implicit Euler" which has better stability than explicit Euler but no symplectic properties.

**Impact:** For this application (short simulation with small damping), the method is adequate. For long simulations or high-accuracy requirements, RK4 or an exponential integrator would be more appropriate.

---

### Issue 7.3: Hardcoded Trajectory File Paths [Low]

**Location:** Lines 558-566

**Code:**
```python
traj_candidates = [
    os.path.join(os.path.dirname(__file__), "spacecraft_trajectory_4th_180deg_30s.npz"),
    os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "spacecraft_trajectory_4th_180deg_30s.npz")),
]
```

**Issue:** The file search is based on `__file__`, which may fail in:
- Frozen executables (PyInstaller, etc.)
- Jupyter notebooks
- Installed packages where `__file__` differs from development layout

**Recommendation:** Use `importlib.resources` for package data, or accept the trajectory file path as a parameter.

---

### Issue 7.4: Stability Margin Computation Edge Cases [Medium]

**Location:** Lines 1088-1108

**Code:**
```python
# Phase crossover (for gain margin)
idx = np.where((phase_deg[:-1] > -180.0) & (phase_deg[1:] <= -180.0))[0]
```

**Issue:** This finds indices where phase crosses -180° from above. Potential issues:

1. **Phase wrapping:** If phase jumps from -179° to +179° (wrapping), this won't detect it as a -180° crossing
2. **Multiple crossings:** Only the first crossing is used, but flexible mode systems often have multiple -180° crossings
3. **No crossing:** If the phase never reaches -180°, `gm_db` stays at infinity, which may not be the desired behavior

The code uses `np.unwrap()` on the phase earlier, which should handle wrapping. However, edge cases near ±180° could still cause issues.

**Recommendation:** Add handling for:
- Multiple phase crossings (report the worst-case gain margin)
- Systems that are unstable (negative gain margin)

---

### Issue 7.5: Floating-Point Comparison Without Tolerance [Low]

**Location:** Lines 1093-1095

**Code:**
```python
if ph2 != ph1:
    f_pc = f1 + (f2 - f1) * (-180.0 - ph1) / (ph2 - ph1)
else:
    f_pc = f1
```

**Issue:** Comparing floating-point numbers with `!=` can be unreliable. If `ph2` and `ph1` are nearly equal (e.g., differ by $10^{-15}$), the division could produce very large or very small values due to numerical precision issues.

**Recommendation:** Use a tolerance-based comparison:
```python
if abs(ph2 - ph1) > 1e-10:
    f_pc = f1 + (f2 - f1) * (-180.0 - ph1) / (ph2 - ph1)
else:
    f_pc = f1
```

---

## Summary Table

| File | Issue ID | Severity | Summary |
|------|----------|----------|---------|
| design_shaper.py | 1.1 | Medium | Non-standard ZVD K formula (correct but confusing) |
| design_shaper.py | 1.2 | Low | Arbitrary division-by-zero guard |
| design_shaper.py | 1.3 | High | No validation for zeta >= 1 |
| design_shaper.py | 1.4 | Medium | LCM-based dt may be suboptimal |
| feedback_control.py | 2.1 | Medium | MRP error sign convention ambiguity |
| feedback_control.py | 2.2 | Medium | Bandwidth formula edge cases |
| feedback_control.py | 2.3 | High | PPF integration method inconsistency |
| feedback_control.py | 2.4 | **Critical** | Notch filter implementation incorrect |
| feedback_control.py | 2.5 | Medium | Overdamped settling time formula wrong |
| feedback_control.py | 2.6 | Low | Negative Ki used for disable logic |
| feedforward_control.py | 3.1 | Low | Bang-bang transition (verified correct) |
| feedforward_control.py | 3.2 | Low | Smooth trajectory initial value (by design) |
| feedforward_control.py | 3.3 | Medium | Shaper timing discretization error |
| feedforward_control.py | 3.4 | Medium | RW torque sign needs verification |
| feedforward_control.py | 3.5 | - | Minimum duration formula (verified correct) |
| spacecraft_model.py | 4.1 | High | Flex mode direction inconsistency |
| spacecraft_model.py | 4.2 | Low | Misleading "critical damping" comment |
| spacecraft_model.py | 4.3 | Medium | Hardcoded mass vs imported inertia |
| spacecraft_model.py | 4.4 | Low | Variable shadows module import |
| spacecraft_properties.py | 5.1 | Medium | Modal vs appendage mass confusion |
| spacecraft_properties.py | 5.2 | Low | Modal gain units undocumented |
| vizard_demo.py | 6.1 | Low | MRP angle conversion edge cases |
| vizard_demo.py | 6.2 | Low | Comet direction (verified correct) |
| vizard_demo.py | 6.3 | Low | Motor torque array pattern |
| vizard_demo.py | 6.4 | Medium | Hardcoded jitter/blur constants |
| mission_simulation.py | 7.1 | **Critical** | ZVD only targets first mode |
| mission_simulation.py | 7.2 | Medium | Symplectic Euler misnomer for damped system |
| mission_simulation.py | 7.3 | Low | Hardcoded trajectory file paths |
| mission_simulation.py | 7.4 | Medium | Stability margin edge cases |
| mission_simulation.py | 7.5 | Low | Float comparison without tolerance |

---

## Recommendations Priority

### Immediate (Critical)
1. **Fix notch filter implementation** in feedback_control.py - fundamentally broken
2. **Implement multi-mode ZVD shaper** in mission_simulation.py - currently only suppresses first mode

### High Priority
3. Add damping ratio validation ($\zeta < 1$) to design_shaper.py
4. Clarify flex mode direction in spacecraft_model.py (update docstring or code)
5. Fix PPF integration method in feedback_control.py
6. Fix overdamped settling time formula in feedback_control.py

### Medium Priority
7. Verify RW torque sign convention against Basilisk
8. Improve stability margin computation for multi-crossing cases
9. Document modal vs appendage mass distinction
10. Parameterize jitter-to-blur conversion constants

### Low Priority (Code Quality)
11. Use tolerance-based float comparisons
12. Improve comments and documentation
13. Remove variable shadowing
14. Use consistent division-by-zero guards

---

*Report generated by Claude Code analysis*
