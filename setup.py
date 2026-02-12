"""
setup.py

Package installer configuration for the Basilisk spacecraft simulation.
This file defines the package metadata, dependencies, and build settings
required for installing the basilisk_sim library.

The source modules live under src/ and are discovered automatically.
"""

from setuptools import setup, find_packages

# Read the long description from the project README for PyPI display.
with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="basilisk-sim",
    version="0.1.0",
    author="Jomin Joseph Karukakalam",
    description="Spacecraft input shaping simulation with Basilisk",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/TheControlEngineer/spacecraft_input_shaping",

    # All importable packages are located inside the src/ directory.
    package_dir={"": "src"},
    packages=find_packages(where="src"),

    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Physics",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.10",

    # Core scientific computing libraries required at runtime.
    install_requires=[
        "numpy>=1.20.0",
        "scipy>=1.7.0",
        "matplotlib>=3.3.0",
    ],

    # Optional dependency groups for development and Basilisk integration.
    extras_require={
        "dev": ["pytest>=6.0", "pytest-cov", "black", "flake8"],
        "basilisk": ["Basilisk"],
    },
)
