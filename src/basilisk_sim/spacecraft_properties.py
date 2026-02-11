"""
Shared spacecraft properties and inertia helpers.

Keep these values aligned with FlexibleSpacecraft so feedforward sizing
matches the simulated configuration.
"""

from __future__ import annotations

from typing import Iterable, Optional

import numpy as np

HUB_INERTIA = np.array(
    [
        [900.0, 0.0, 0.0],
        [0.0, 800.0, 0.0],
        [0.0, 0.0, 600.0],
    ],
    dtype=float,
)

# Modal mass is the effective mass for each flexible mode (spring mass damper).
FLEX_MODE_MASS = 5.0
# Array mass is the physical appendage mass (not used by default in inertia sizing).
ARRAY_MASS_PER_WING = 50.0

FLEX_MODE_LOCATIONS = {
    "mode1_port": np.array([0.0, -3.5, 0.0], dtype=float),
    "mode2_port": np.array([0.0, -4.5, 0.0], dtype=float),
    "mode1_stbd": np.array([0.0, 3.5, 0.0], dtype=float),
    "mode2_stbd": np.array([0.0, 4.5, 0.0], dtype=float),
}

DEFAULT_MODE_KEYS = ("mode1_port", "mode2_port")

# Default inertia sizing uses the modal masses to match the Basilisk model.
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

    Returns a list of distances in meters for the provided mode keys.
    """
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
    Compute modal gains that map torque to modal acceleration (gain * torque).

    For base excitation of a flex mode at lever arm r:
        q_ddot + 2*zeta*omega*q_dot + omega^2*q = (r / I_axis) * torque

    Units:
        torque: N*m
        gain: 1/(kg*m)
        q_ddot: m/s^2
    """
    axis = np.array(rotation_axis, dtype=float).reshape(3)
    axis_norm = np.linalg.norm(axis)
    if axis_norm <= 0:
        axis = np.array([0.0, 0.0, 1.0])
        axis_norm = 1.0
    axis /= axis_norm

    inertia = np.array(inertia, dtype=float).reshape(3, 3)
    I_axis = float(axis @ inertia @ axis)
    if I_axis <= 0:
        return []

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
    Return the rigid-body inertia including appended array mass.
    """
    inertia = np.array(HUB_INERTIA if hub_inertia is None else hub_inertia, dtype=float)
    locations = mode_locations if mode_locations is not None else FLEX_MODE_LOCATIONS
    masses = appendage_masses if appendage_masses is not None else DEFAULT_APPENDAGE_MASSES

    if isinstance(locations, dict):
        items = locations.items()
    else:
        items = [(None, loc) for loc in locations]

    for key, location in items:
        r_vec = np.array(location, dtype=float).reshape(3)
        mass = float(masses.get(key, modal_mass)) if key is not None else float(modal_mass)
        inertia += mass * ((r_vec @ r_vec) * np.eye(3) - np.outer(r_vec, r_vec))

    return inertia
