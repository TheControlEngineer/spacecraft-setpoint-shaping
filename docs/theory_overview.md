# Input Shaping: Theory Overview

## The Problem

When commanding a flexible structure (like a spacecraft with solar arrays) to move quickly, vibrations are induced that persist long after the motion completes. These residual vibrations:

- Degrade pointing accuracy
- Blur camera images during long exposures
- Require time to settle before next operation
- May violate mission requirements

**Traditional solution:** Move slowly and wait for vibrations to damp out naturally.

**Input shaping solution:** Shape the command to avoid exciting vibrations in the first place.

---

## The Core Idea

Instead of applying a step command, split it into a **sequence of impulses** timed to create destructive interference of the vibration response.

### Visual Intuition

**Unshaped Step Command:**
```
Force: |████████████|________
Vibration: ~~~~~~~~~~~~~~~~~~~~~~~~~~~~ (long decay)
```

**ZV Shaped Command:**
```
Force: |████|____|████|________
Vibration:    ~~~↓↑~~~  →  ___________ (cancelled!)
```

The second impulse arrives when the first impulse's vibration is 180° out of phase, causing cancellation.

---

## Why It Works

**Phasor diagram perspective:**

Each impulse creates a vibration that can be represented as a rotating vector (phasor). The residual vibration is the vector sum of all impulse responses.

**ZV (2 impulses):**
- First impulse creates phasor at 0°
- Second impulse (at half-period) creates phasor at 180°
- Equal magnitudes → perfect cancellation

**ZVD (3 impulses):**
- Three phasors arranged for cancellation
- Creates a "flat" frequency response near nominal
- More robust to frequency errors

---

## Key Design Parameters

### Input: System Properties
- $\omega_n$: Natural frequency (from modal analysis or system ID)
- $\zeta$: Damping ratio (typically very low for spacecraft)

### Output: Shaper Design
- $N$: Number of impulses
- $[A_1, A_2, ..., A_N]$: Impulse amplitudes (sum to 1.0)
- $[t_1, t_2, ..., t_N]$: Impulse times

### Trade-off
- More impulses → Better robustness BUT longer maneuver duration

---

## Application to Spacecraft

### Why Spacecraft Need Input Shaping

Spacecraft have:
1. **Very flexible appendages** (solar arrays, antennas)
   - Large surface area, low mass
   - Natural frequencies: 0.1-1 Hz
   
2. **Very light damping** (vacuum environment)
   - $\zeta \approx 0.01-0.05$
   - Vibrations persist for minutes without control

3. **Strict pointing requirements**
   - Science observations (Hubble, JWST)
   - Communication (high-gain antennas)
   - Planet/comet imaging

### How Input Shaping Helps

**Scenario:** Slew spacecraft 30° to track a comet

**Without shaping:**
- Fast slew (5 seconds)
- Solar arrays ring for 60+ seconds
- Camera images are blurred
- Total time: 65 seconds

**With ZVD shaping:**
- Shaped slew (7 seconds with 2-second shaper)
- Solar arrays settle in ~5 seconds
- Camera ready faster
- Total time: 12 seconds
- **Better AND faster!**

---

## Multi-Mode Extension

Real spacecraft have multiple flexible modes. Two approaches:

### 1. Cascaded Convolution
Design separate shapers for each mode, then convolve:
$$\text{Shaper}_{total} = \text{Shaper}_1 * \text{Shaper}_2 * \text{Shaper}_3$$

**Pro:** Simple, modular
**Con:** Duration adds up

### 2. Simultaneous Optimization
Design one multi-mode shaper via optimization:
$$\min t_N \quad \text{s.t.} \quad V_i(\omega_i) \leq V_{tol} \quad \forall i$$

**Pro:** Shorter duration
**Con:** More complex, no closed form

---

## Comparison with Feedback Control

| Aspect | Input Shaping | LQR Feedback |
|--------|---------------|--------------|
| Type | Feedforward | Feedback |
| Requires sensors? | No | Yes (rate gyros, etc.) |
| Handles disturbances? | No | Yes |
| Computational load | Low (offline design) | High (real-time) |
| Robustness source | Derivative constraints | State feedback gains |
| Best for | Known moves | Unknown disturbances |

**Best practice:** Combine both!
- Input shaping for commanded maneuvers
- LQR for disturbance rejection

---

## Historical Context

**Developed:** Late 1950s (Smith, O.J.M. - Posicast control)
**Revived:** Late 1980s (Singer & Seering - ZV/ZVD shapers)
**Modern applications:**
- Cranes and gantries
- Hard disk drives
- Robotic manipulators
- Spacecraft (MRO, JWST, etc.)

---

## Further Reading

- [Singer & Seering 1990] - Original ZV/ZVD derivation
- [Singhose 2009] - Comprehensive review article
- [Wie 2008] - Space Vehicle Dynamics textbook