"""
spacecraft_properties.py

Shared spacecraft physical constants, inertia data, and modal parameter helpers.

This module is the single source of truth for all physical parameters that
characterise the spacecraft bus and its flexible solar array appendages.
Every other module (feedforward, feedback, design_shaper, etc.) imports
these values to guarantee consistency across the simulation.

Keep the values here aligned with FlexibleSpacecraft in spacecraft_model.py
so that feedforward torque sizing matches the dynamics that Basilisk integrates.
"""

from __future__ import annotations

from typing import Iterable, Optional

import numpy as np

# ============================================================================
# Hub inertia tensor [kg*m^2]
# ============================================================================
# Principal inertia of the rigid spacecraft bus without appendage masses.
# Diagonal because the body frame is aligned with the principal axes.
HUB_INERTIA = np.array(
    [
        [900.0, 0.0, 0.0],
        [0.0, 800.0, 0.0],
        [0.0, 0.0, 600.0],
    ],
    dtype=float,
)

# ============================================================================
# Modal mass parameters
# ============================================================================
# Each flexible mode is modelled as a spring mass damper with this effective
# participating mass.  This is NOT the total panel mass; it is the fraction
# of panel mass that participates in each bending mode (roughly 10% of the
# physical 50 kg wing mass).
FLEX_MODE_MASS = 5.0  # kg, per mode

# Physical mass of one deployable wing (used for reference only, not in the
# default inertia calculation which uses modal masses instead).
ARRAY_MASS_PER_WING = 50.0  # kg

# ============================================================================
# Modal attachment locations
# ============================================================================
# Position vectors [x, y, z] in the body frame (metres) where each spring
# mass damper attaches to the hub.  Solar arrays extend along the Y axis
# (port at negative Y, starboard at positive Y).  Two modes per wing
# represent the first and second bending harmonics.
FLEX_MODE_LOCATIONS = {
    "mode1_port": np.array([0.0, -3.5, 0.0], dtype=float),   # 1st bending, port wing
    "mode2_port": np.array([0.0, -4.5, 0.0], dtype=float),   # 2nd bending, port wing
    "mode1_stbd": np.array([0.0, 3.5, 0.0], dtype=float),    # 1st bending, starboard wing
    "mode2_stbd": np.array([0.0, 4.5, 0.0], dtype=float),    # 2nd bending, starboard wing
}

# Default subset of mode keys used when only one wing needs analysis
# (port side is enough due to symmetry).
DEFAULT_MODE_KEYS = ("mode1_port", "mode2_port")

# Default masses assigned to each modal attachment point.  These match the
# Basilisk LinearSpringMassDamper effector setup in spacecraft_model.py.
DEFAULT_APPENDAGE_MASSES = {
    "mode1_port": FLEX_MODE_MASS,
    "mode2_port": FLEX_MODE_MASS,
    "mode1_stbd": FLEX_MODE_MASS,
    "mode2_stbd": FLEX_MODE_MASS,
}


def compute_mode_lever_arms(
    rotation_axis: Iterable[float],
    mode_keys: Iterable[str] = DEFAULT_MODE_KEYS,
    mode_locations: Optional[dict] = None,
) -> list[float]:
    """
    Compute perpendicular lever arms from a rotation axis to each mode location.

    For each mode attachment point, the lever arm is the perpendicular distance
    from the rotation axis to the point location.  This distance governs the
    strength of the base excitation coupling (larger lever arm means stronger
    torque to modal excitation coupling).

    Args:
        rotation_axis: 3 element unit vector defining the rotation axis.
        mode_keys: which modes to include (keys into mode_locations).
        mode_locations: dict mapping mode key to position vector; defaults to
                        the package level FLEX_MODE_LOCATIONS.

    Returns:
        List of perpendicular distances (metres) for the requested mode keys.
    """
    # Normalise the rotation axis to unit length.
    axis = np.array(rotation_axis, dtype=float).reshape(3)
    axis_norm = np.linalg.norm(axis)
    if axis_norm <= 0:
        axis = np.array([0.0, 0.0, 1.0])
        axis_norm = 1.0
    axis /= axis_norm

    locations = mode_locations if mode_locations is not None else FLEX_MODE_LOCATIONS

    lever_arms = []
    for key in mode_keys:
        loc = locations.get(key)
        if loc is None:
            continue
        # Perpendicular distance = magnitude of (axis x location_vector).
        lever_arm = np.linalg.norm(np.cross(axis, np.array(loc, dtype=float).reshape(3)))
        lever_arms.append(float(lever_arm))
    return lever_arms


def compute_modal_gains(
    inertia: np.ndarray,
    rotation_axis: Iterable[float],
    mode_keys: Iterable[str] = DEFAULT_MODE_KEYS,
    mode_locations: Optional[dict] = None,
) -> list[float]:
    """
    Compute modal gains that map torque to modal acceleration.

    For base excitation of a spring mass damper on a rotating hub, the
    equation of motion for the modal coordinate q is:

        q_ddot + 2*zeta*omega*q_dot + omega^2*q = (r / I_axis) * torque

    where r is the lever arm and I_axis is the hub inertia about the
    rotation axis.  The modal gain returned here is (r / I_axis).

    Args:
        inertia: 3x3 spacecraft inertia tensor [kg*m^2].
        rotation_axis: 3 element unit vector for the slew axis.
        mode_keys: modes to compute gains for.
        mode_locations: position dict (defaults to FLEX_MODE_LOCATIONS).

    Returns:
        List of modal gains [1/(kg*m)] for each requested mode.
    """
    # Normalise axis.
    axis = np.array(rotation_axis, dtype=float).reshape(3)
    axis_norm = np.linalg.norm(axis)
    if axis_norm <= 0:
        axis = np.array([0.0, 0.0, 1.0])
        axis_norm = 1.0
    axis /= axis_norm

    # Scalar inertia about the rotation axis: I_axis = axis^T * J * axis.
    inertia = np.array(inertia, dtype=float).reshape(3, 3)
    I_axis = float(axis @ inertia @ axis)
    if I_axis <= 0:
        return []

    # Gain for each mode is lever_arm / I_axis.
    lever_arms = compute_mode_lever_arms(
        rotation_axis=axis, mode_keys=mode_keys, mode_locations=mode_locations
    )
    return [arm / I_axis for arm in lever_arms]


def compute_effective_inertia(
    hub_inertia: Optional[np.ndarray] = None,
    mode_locations: Optional[Iterable[np.ndarray]] = None,
    modal_mass: float = FLEX_MODE_MASS,
    appendage_masses: Optional[dict] = None,
) -> np.ndarray:
    """
    Compute the total rigid body inertia including appendage modal masses.

    Uses the parallel axis theorem to add each point mass contribution
    to the hub inertia.  The contribution of a point mass m at position
    vector r is:  delta_I = m * (r.r * I3  -  r (x) r)

    This is the inertia used by the feedforward controller for torque sizing
    so that the commanded torque correctly accounts for the mass of the
    deployed solar arrays.

    Args:
        hub_inertia: 3x3 rigid hub inertia (defaults to HUB_INERTIA).
        mode_locations: position vectors or dict of mode locations
                        (defaults to FLEX_MODE_LOCATIONS).
        modal_mass: fallback mass per mode if not found in appendage_masses.
        appendage_masses: dict mapping mode key to its mass [kg].

    Returns:
        3x3 effective inertia tensor [kg*m^2].
    """
    inertia = np.array(HUB_INERTIA if hub_inertia is None else hub_inertia, dtype=float)
    locations = mode_locations if mode_locations is not None else FLEX_MODE_LOCATIONS
    masses = appendage_masses if appendage_masses is not None else DEFAULT_APPENDAGE_MASSES

    # Handle both dict and list inputs for mode locations.
    if isinstance(locations, dict):
        items = locations.items()
    else:
        items = [(None, loc) for loc in locations]

    for key, location in items:
        r_vec = np.array(location, dtype=float).reshape(3)
        # Look up the mass for this mode, falling back to the default modal mass.
        mass = float(masses.get(key, modal_mass)) if key is not None else float(modal_mass)
        # Parallel axis theorem: I += m * (r^2 * I3  -  r outer r)
        inertia += mass * ((r_vec @ r_vec) * np.eye(3) - np.outer(r_vec, r_vec))

    return inertia
