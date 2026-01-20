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

FLEX_MODE_MASS = 5.0

FLEX_MODE_LOCATIONS = {
    "mode1_port": np.array([0.0, -3.5, 0.0], dtype=float),
    "mode2_port": np.array([0.0, -4.5, 0.0], dtype=float),
    "mode1_stbd": np.array([0.0, 3.5, 0.0], dtype=float),
    "mode2_stbd": np.array([0.0, 4.5, 0.0], dtype=float),
}


def compute_effective_inertia(
    hub_inertia: Optional[np.ndarray] = None,
    mode_locations: Optional[Iterable[np.ndarray]] = None,
    modal_mass: float = FLEX_MODE_MASS,
) -> np.ndarray:
    """
    Return the rigid-body inertia including the flexible mode masses.
    """
    inertia = np.array(HUB_INERTIA if hub_inertia is None else hub_inertia, dtype=float)
    locations = mode_locations if mode_locations is not None else FLEX_MODE_LOCATIONS.values()

    for location in locations:
        r_vec = np.array(location, dtype=float).reshape(3)
        inertia += modal_mass * ((r_vec @ r_vec) * np.eye(3) - np.outer(r_vec, r_vec))

    return inertia
