"""
Input Shaping Library for Flexible Structure Vibration Suppression

A comprehensive library implementing various input shaping techniques for
suppressing residual vibrations in flexible structures, with applications
to spacecraft attitude control, precision positioning systems, and robotics.
"""

from .shapers import (
    ZV, ZVD, ZVDD, EI, 
    design_shaper,
    convolve_shapers,
    design_multimode_cascaded,
    design_multimode_simultaneous
)

__version__ = "0.1.0"
__author__ = "Jomin Joseph Karukakalam"

__all__ = [
    'ZV',
    'ZVD',
    'ZVDD',
    'EI',
    'design_shaper',
    'convolve_shapers',
    'design_multimode_cascaded',
    'design_multimode_simultaneous',
]