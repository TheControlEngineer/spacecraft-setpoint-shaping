"""
Basilisk Spacecraft Input Shaping Simulation

A high-fidelity simulation framework for spacecraft attitude control
with flexible appendages, featuring input shaping for vibration suppression.

Modules:
    spacecraft_properties: Physical constants, inertia, and modal parameters
    spacecraft_model: Basilisk FlexibleSpacecraft model
    feedforward_control: Input shaping and trajectory generation
    feedback_control: LQR, MRP, and notch filter controllers
    design_shaper: ZV/ZVD/ZVDD/EI shaper design algorithms
    star_camera_simulator: Star camera models for attitude sensing
    comet_camera_simulator: Comet tracking and image blur analysis
    basilisk_star_camera_integration: Integration layer for camera analysis
"""

__version__ = "0.1.0"

# Core spacecraft parameters
from .spacecraft_properties import (
    HUB_INERTIA,
    FLEX_MODE_MASS,
    FLEX_MODE_LOCATIONS,
    compute_effective_inertia,
    compute_modal_gains,
    compute_mode_lever_arms,
)

# Spacecraft model
from .spacecraft_model import FlexibleSpacecraft

# Controllers
from .feedforward_control import FeedforwardController
from .feedback_control import (
    MRPFeedbackController,
    FilteredDerivativeController,
    NotchFilterController,
    TrajectoryTrackingController,
    HybridController,
)

# Shaper design
from .design_shaper import compute_residual_vibration_continuous

# Camera simulators
from .star_camera_simulator import StarCameraSimulator
from .comet_camera_simulator import CometCameraSimulator

__all__ = [
    # Version
    "__version__",
    # Spacecraft properties
    "HUB_INERTIA",
    "FLEX_MODE_MASS",
    "FLEX_MODE_LOCATIONS",
    "compute_effective_inertia",
    "compute_modal_gains",
    "compute_mode_lever_arms",
    # Spacecraft model
    "FlexibleSpacecraft",
    # Controllers
    "FeedforwardController",
    "MRPFeedbackController",
    "FilteredDerivativeController",
    "NotchFilterController",
    "TrajectoryTrackingController",
    "HybridController",
    # Shaper design
    "compute_residual_vibration_continuous",
    # Camera simulators
    "StarCameraSimulator",
    "CometCameraSimulator",
]
