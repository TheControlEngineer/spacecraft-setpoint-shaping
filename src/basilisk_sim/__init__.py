"""
Basilisk Spacecraft Input Shaping Simulation

A high fidelity simulation framework for spacecraft attitude control
with flexible appendages, featuring input shaping for vibration suppression.

This package provides the following modules:

    spacecraft_properties : Physical constants, inertia tensors, and modal parameters
    spacecraft_model      : Basilisk FlexibleSpacecraft model with solar array dynamics
    feedforward_control   : Input shaping and trajectory generation for open loop control
    feedback_control      : LQR, MRP, and notch filter closed loop controllers
    design_shaper         : ZV/ZVD/ZVDD/EI shaper design algorithms
    star_camera_simulator : Star camera models for attitude sensing
    comet_camera_simulator: Comet tracking simulation and image blur analysis
    basilisk_star_camera_integration : Integration layer bridging Basilisk and camera analysis
    state_estimator       : Phasor based modal state estimator for narrowband flex modes
"""

__version__ = "0.1.0"

# ============================================================================
# Core spacecraft parameters
# ============================================================================
# Import physical properties shared across all simulation modules.
from .spacecraft_properties import (
    HUB_INERTIA,
    FLEX_MODE_MASS,
    FLEX_MODE_LOCATIONS,
    compute_effective_inertia,
    compute_modal_gains,
    compute_mode_lever_arms,
)

# ============================================================================
# Spacecraft model
# ============================================================================
# The FlexibleSpacecraft class wraps the Basilisk spacecraft object
# and adds solar array bending mode dynamics.
from .spacecraft_model import FlexibleSpacecraft

# ============================================================================
# Controllers
# ============================================================================
# Feedforward (open loop) and feedback (closed loop) controllers.
from .feedforward_control import FeedforwardController
from .feedback_control import (
    MRPFeedbackController,
    FilteredDerivativeController,
    NotchFilterController,
    TrajectoryTrackingController,
    HybridController,
)

# ============================================================================
# Shaper design utilities
# ============================================================================
# Functions for computing residual vibration and reference trajectories.
from .design_shaper import (
    compute_residual_vibration_continuous,
    design_trapezoidal_trajectory,
    design_s_curve_trajectory,
)

# ============================================================================
# Camera simulators
# ============================================================================
# Synthetic camera renderers used for motion blur visualisation.
from .star_camera_simulator import StarCameraSimulator
from .comet_camera_simulator import CometCameraSimulator

# Public API list for wildcard imports (from basilisk_sim import *).
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
    "design_trapezoidal_trajectory",
    "design_s_curve_trajectory",
    # Camera simulators
    "StarCameraSimulator",
    "CometCameraSimulator",
]
