"""
Input Shaping Library for Flexible Structure Vibration Suppression

A comprehensive library implementing various input shaping techniques for
suppressing residual vibrations in flexible structures, with applications
to spacecraft attitude control, precision positioning systems, and robotics.
"""

from .shapers import ZV, ZVD, ZVDD, EI, design_shaper

__version__ = "0.1.0"
__author__ = "Your Name"

__all__ = [
    'ZV',
    'ZVD',
    'ZVDD',
    'EI',
    'design_shaper',
]