# Multi-Mode Input Shaping

## The Multi-Mode Problem

Real spacecraft have multiple flexible modes that must all be suppressed simultaneously. For example, a spacecraft with solar arrays:

- **Mode 1:** 0.3 Hz (first bending) - largest amplitude
- **Mode 2:** 0.8 Hz (second bending) - medium amplitude
- **Mode 3:** 1.5 Hz (third bending) - smaller amplitude

A single-mode shaper only suppresses one frequency. Multi-mode shaping addresses all critical modes.

---

## Two Approaches

### Approach 1: Cascaded Convolution

**Method:** Design separate shapers for each mode and convolve them.

**Algorithm:**
```python
S_total = S_mode1 ⊗ S_mode2 ⊗ S_mode3
```

**Properties:**
- Simple to implement
- Modular (add/remove modes easily)
- Guaranteed perfect suppression of each mode
- **Drawback:** Duration = sum of individual durations
- **Drawback:** Impulses multiply: N_total = N₁ × N₂ × N₃

**Example (2 modes, ZVD):**
- Mode 1: 3 impulses, 3.33s
- Mode 2: 3 impulses, 1.25s
- **Result: 9 impulses, 4.58s**

---

### Approach 2: Simultaneous Optimization

**Method:** Design one shaper that suppresses all modes with specified tolerance.

**Optimization Problem:**
```
minimize: t_N (duration)
subject to:
    V_i(ω_i, ζ_i) ≤ V_tol  for all modes i
    Σ A_j = 1              (unity gain)
    A_j > 0                (positive amplitudes)
```

**Properties:**
- Optimal duration for given tolerance
- Typically 30-60% shorter than cascaded
- Requires numerical optimization (~50ms computation)
- Achieves tolerance spec, not zero vibration

**Example (2 modes, 5% tolerance):**
- **Result: 6 impulses, 1.79s** (61% shorter!)

---

## Performance Comparison

### Test Case: 2-Mode Spacecraft (0.3 Hz, 0.8 Hz)

| Approach | Impulses | Duration | Suppression | Computation |
|----------|----------|----------|-------------|-------------|
| Cascaded (ZVD) | 9 | 4.58s | Perfect (0%) | 2.6 ms |
| Simultaneous (5%) | 6 | 1.79s | 5% residual | 49.5 ms |

**Key Result:** 2.57x faster maneuvers with acceptable 5% residual vibration.

---

## Tolerance vs Duration Trade-off

For simultaneous optimization, tolerance specification directly impacts duration:

| Tolerance | Duration | vs Cascaded |
|-----------|----------|-------------|
| 2% | 1.81s | -60.5% |
| 5% | 1.79s | -61.0% |
| 10% | 1.75s | -61.8% |
| 20% | 1.68s | -63.3% |

**Insight:** Most benefit comes from relaxing to 10-15% tolerance. Beyond that, diminishing returns.

---

## When to Use Each Approach

### Use Cascaded When:
-  Perfect suppression required (0% vibration)
-  Modularity important (frequent mode updates)
-  Implementation simplicity valued
-  Duration not critical

**Example:** High-precision science observations (Hubble, JWST)

### Use Simultaneous When:
-  Speed critical (fast slew requirements)
-  Tolerance specification exists (e.g., <5% acceptable)
-  Optimal performance needed
-  Can afford optimization computation

**Example:** Rapid target acquisition, communication slews

---

## Implementation Notes

### Cascaded Convolution
```python
from input_shaping import design_multimode_cascaded

# Simple one-line call
A, t = design_multimode_cascaded(
    mode_frequencies=[0.3, 0.8, 1.5],  # Hz
    damping_ratios=[0.02, 0.02, 0.03],
    method='ZVD'
)
```

### Simultaneous Optimization
```python
from input_shaping import design_multimode_simultaneous

# Specify tolerance and impulse count
A, t = design_multimode_simultaneous(
    mode_frequencies=[0.3, 0.8],
    damping_ratios=[0.02, 0.02],
    n_impulses=6,        # More impulses → better performance
    Vtol=0.05            # 5% tolerance
)
```

---

## Design Guidelines

### 1. Choose Number of Impulses (Simultaneous)

**Rule of thumb:** `n_impulses ≥ 2 × n_modes`

- 2 modes → 4-6 impulses
- 3 modes → 6-9 impulses
- More impulses → better suppression, longer duration

### 2. Select Tolerance

**Mission-driven approach:**
1. Calculate acceptable vibration from image blur requirements
2. Add safety margin (15-20%)
3. Use as `Vtol` parameter

**Example:**
- Camera blur analysis: <8% vibration acceptable
- Safety margin: 8% × 0.8 = 6.4%
- **Use: Vtol = 0.06**

### 3. Verify Suppression

Always validate that achieved suppression meets requirements:
```python
def verify_suppression(shaper, modes):
    for freq, zeta in modes:
        V = calculate_residual(shaper, freq, zeta)
        print(f"{freq} Hz: {V*100:.2f}% residual")
```

---

## Computational Performance

| Operation | Time | Real-time? |
|-----------|------|------------|
| Cascaded design | ~3 ms | ✓ Yes |
| Simultaneous (2 modes) | ~50 ms | ✓ Yes |
| Simultaneous (3 modes) | ~200 ms | ✓ Yes |

Both approaches are fast enough for onboard recomputation if system ID updates during mission.

---

## References

1. Singer, N. C., & Seering, W. P. (1990). Preshaping command inputs to reduce system vibration.

2. Singhose, W., Seering, W., & Singer, N. (1996). Input shaping for vibration reduction with specified insensitivity to modeling errors.

3. Singh, T., & Vadali, S. R. (1993). Robust time-delay control of multimode systems.