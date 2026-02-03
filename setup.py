from setuptools import setup, find_packages

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
    install_requires=[
        "numpy>=1.20.0",
        "scipy>=1.7.0",
        "matplotlib>=3.3.0",
    ],
    extras_require={
        "dev": ["pytest>=6.0", "pytest-cov", "black", "flake8"],
        "basilisk": ["Basilisk"],
    },
)
