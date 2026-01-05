# Mathematical Derivations of Input Shapers

## Fundamentals

### Single-Mode Flexible System

Consider a flexible structure with a single dominant mode modeled as a damped harmonic oscillator:

$$\ddot{x} + 2\zeta\omega_n\dot{x} + \omega_n^2 x = F(t)$$

where:
- $x$ = modal displacement
- $\omega_n$ = natural frequency (rad/s)
- $\zeta$ = damping ratio (dimensionless)
- $F(t)$ = applied force

The **damped natural frequency** is:

$$\omega_d = \omega_n\sqrt{1-\zeta^2}$$

For spacecraft solar arrays, typical values are:
- $\omega_n \in [0.1, 1.0]$ Hz (very flexible)
- $\zeta \in [0.01, 0.05]$ (very light damping in vacuum)

---

## Impulse Response

For a unit impulse $F(t) = \delta(t)$ applied at $t=0$, the displacement response is:

$$h(t) = \frac{1}{\omega_d}e^{-\zeta\omega_n t}\sin(\omega_d t) \quad \text{for } t \geq 0$$

**Key insight:** The response oscillates at $\omega_d$ with exponentially decaying amplitude.

For a lightly damped system ($\zeta \ll 1$):
- $\omega_d \approx \omega_n$ (damped frequency ≈ natural frequency)
- Decay time constant: $\tau = \frac{1}{\zeta\omega_n}$

---

## Residual Vibration

When multiple impulses are applied at different times, the responses superpose. For $N$ impulses with amplitudes $A_i$ at times $t_i$:

$$x(t) = \sum_{i=1}^{N} \frac{A_i}{\omega_d}e^{-\zeta\omega_n(t-t_i)}\sin(\omega_d(t-t_i)) \cdot u(t-t_i)$$

where $u(t-t_i)$ is the unit step function.

After all impulses have been applied (for $t > t_N$), the **residual vibration amplitude** can be expressed as:

$$V(\omega, \zeta) = \left|\sum_{i=1}^{N} A_i e^{-\zeta\omega t_i} e^{j\omega t_i}\right|$$

This complex magnitude represents the vibration amplitude as a function of frequency.

---

## ZV Shaper Derivation

**Design Goal:** Find 2 impulses that produce zero residual vibration at the nominal frequency.

### Constraints

1. **Zero vibration at $\omega_n$:**
   $$V(\omega_n, \zeta) = 0$$

2. **Unity gain** (preserve command magnitude):
   $$A_1 + A_2 = 1$$

### Solution

Using complex exponential notation, for two impulses at $t_1 = 0$ and $t_2$:

$$V = |A_1 + A_2 e^{-\zeta\omega_n t_2} e^{j\omega_n t_2}|$$

For zero vibration:
$$A_2 e^{-\zeta\omega_n t_2} e^{j\omega_n t_2} = -A_1$$

This requires:
- **Phase condition:** $e^{j\omega_n t_2} = -1 = e^{j\pi}$
  
  Therefore: $\omega_n t_2 = \pi \Rightarrow t_2 = \frac{\pi}{\omega_d}$ (half damped period)

- **Amplitude condition:** With damping decay over time $t_2$:
  
  $$A_2 e^{-\zeta\omega_n t_2} = A_1$$
  
  Let $K = e^{-\zeta\omega_n t_2} = e^{-\zeta\pi/\sqrt{1-\zeta^2}}$

Combined with unity gain constraint:

$$A_1 = \frac{1}{1+K}, \quad A_2 = \frac{K}{1+K}$$

### ZV Shaper Summary

$$\boxed{
\begin{align}
\text{Times:} \quad & t_1 = 0, \quad t_2 = \frac{\pi}{\omega_d} \\
\text{Amplitudes:} \quad & A_1 = \frac{1}{1+K}, \quad A_2 = \frac{K}{1+K} \\
\text{where:} \quad & K = e^{-\zeta\pi/\sqrt{1-\zeta^2}}
\end{align}
}$$

**For undamped systems** ($\zeta = 0$): $K = 1 \Rightarrow A_1 = A_2 = 0.5$

---

## ZVD Shaper Derivation

**Design Goal:** Zero vibration AND zero derivative at nominal frequency (improved robustness).

### Constraints

1. $V(\omega_n) = 0$ (zero vibration)
2. $\frac{\partial V}{\partial \omega}\bigg|_{\omega_n} = 0$ (zero slope → flat valley)
3. $A_1 + A_2 + A_3 = 1$ (unity gain)

### Solution

With 3 impulses equally spaced by half-periods:
$$t_1 = 0, \quad t_2 = \frac{\pi}{\omega_d}, \quad t_3 = \frac{2\pi}{\omega_d}$$

The amplitudes that satisfy both derivative constraints are:

$$\boxed{
\begin{align}
A_1 &= \frac{1}{1 + 2K + K^2} \\
A_2 &= \frac{2K}{1 + 2K + K^2} \\
A_3 &= \frac{K^2}{1 + 2K + K^2}
\end{align}
}$$

**For undamped systems:** $K = 1 \Rightarrow [A_1, A_2, A_3] = [1/4, 1/2, 1/4]$

**Key property:** The derivative constraint creates a wider "notch" in the frequency response, making the shaper more robust to frequency errors.

---

## ZVDD Shaper Derivation

**Design Goal:** Zero vibration, zero first derivative, AND zero second derivative (maximum robustness).

### Constraints

1. $V(\omega_n) = 0$
2. $\frac{\partial V}{\partial \omega}\bigg|_{\omega_n} = 0$
3. $\frac{\partial^2 V}{\partial \omega^2}\bigg|_{\omega_n} = 0$
4. $A_1 + A_2 + A_3 + A_4 = 1$

### Solution

With 4 impulses:
$$t_1 = 0, \quad t_2 = \frac{\pi}{\omega_d}, \quad t_3 = \frac{2\pi}{\omega_d}, \quad t_4 = \frac{3\pi}{\omega_d}$$

Amplitudes follow a **binomial pattern**:

$$\boxed{
\begin{align}
A_1 &= \frac{1}{1 + 3K + 3K^2 + K^3} \\
A_2 &= \frac{3K}{1 + 3K + 3K^2 + K^3} \\
A_3 &= \frac{3K^2}{1 + 3K + 3K^2 + K^3} \\
A_4 &= \frac{K^3}{1 + 3K + 3K^2 + K^3}
\end{align}
}$$

**For undamped systems:** $[1, 3, 3, 1]/8$ (binomial coefficients!)

**Observation:** Each additional derivative constraint:
- Adds one impulse
- Increases duration by one half-period
- Improves robustness significantly

---

## EI Shaper Formulation

**Design Goal:** Maintain vibration below tolerance $V_{tol}$ across frequency uncertainty $[\omega_n(1-\Delta), \omega_n(1+\Delta)]$.

### Optimization Problem

$$
\begin{align}
\min_{A_i, t_i} \quad & t_N \\
\text{subject to:} \quad & V(\omega_n) = 0 \\
& V(\omega_n(1-\Delta)) = V_{tol} \\
& V(\omega_n(1+\Delta)) = V_{tol} \\
& \sum A_i = 1 \\
& A_i > 0, \quad t_i > 0
\end{align}
$$

**No closed-form solution** - requires numerical optimization (SLSQP, trust-constr).

**Key insight:** EI creates a "flat-top" response:
- Perfect suppression at nominal frequency
- Exactly $V_{tol}$ at edges of uncertainty band
- Guaranteed performance across entire range

---

## Robustness Analysis

For a shaper designed at nominal $\omega_n$ but actual frequency $\omega_n + \Delta\omega$:

$$V(\omega_n + \Delta\omega) = \left|\sum A_i e^{-\zeta(\omega_n + \Delta\omega)t_i} e^{j(\omega_n + \Delta\omega)t_i}\right|$$

**Performance degradation depends on:**
1. Frequency error magnitude $\Delta\omega / \omega_n$
2. Number of derivative constraints (ZV < ZVD < ZVDD)
3. Damping ratio $\zeta$

### Sensitivity Metric

Define **sensitivity** as the rate of performance degradation:

$$S = \frac{dV}{d\omega}\bigg|_{\omega_n}$$

- ZV: $S \neq 0$ (sharp valley)
- ZVD: $S = 0$ but $S' \neq 0$ (flat valley)
- ZVDD: $S = S' = 0$ (very flat valley)

---

## Trade-off: Duration vs. Robustness

For a mode with frequency $f_n$ Hz:

| Shaper | Impulses | Duration | Max $V$ @ ±20% | Use Case |
|--------|----------|----------|----------------|----------|
| ZV | 2 | $1/(2f_n)$ | ~30% | Known frequency, speed critical |
| ZVD | 3 | $1/f_n$ | ~10% | Moderate uncertainty |
| ZVDD | 4 | $3/(2f_n)$ | ~3% | High uncertainty, precision required |
| EI | 3 | $\sim 1/f_n$ | $V_{tol}$ | Specified tolerance |

**Example:** For $f_n = 0.5$ Hz:
- ZV: 1 second
- ZVD: 2 seconds  
- ZVDD: 3 seconds

**Engineering decision:** Choose minimum duration that meets performance requirements.

---

## References

1. Singer, N. C., & Seering, W. P. (1990). Preshaping command inputs to reduce system vibration. *Journal of Dynamic Systems, Measurement, and Control*, 112(1), 76-82.

2. Singhose, W., Seering, W., & Singer, N. (1994). Residual vibration reduction using vector diagrams to generate shaped inputs. *Journal of Mechanical Design*, 116(2), 654-659.

3. Singh, T., & Vadali, S. R. (1993). Robust time-delay control. *Journal of Dynamic Systems, Measurement, and Control*, 115(2), 303-306.