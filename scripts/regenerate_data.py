"""Regenerate all simulation data files.

Runs run_vizard_demo.py for every combination of shaping method,
controller variant, and run mode to produce fresh NPZ output files.
Once complete, run scripts/run_mission.py to update plots and CSVs.
"""
import os
import subprocess
import sys
from pathlib import Path

# Shaping methods and control configurations to sweep
methods = ["s_curve", "fourth"]
controllers = ["standard_pd", "filtered_pd"]
run_modes = ["combined", "fb_only", "ff_only"]

# Locate the vizard demo script relative to this file
script_dir = Path(__file__).parent.resolve()
vizard_demo_path = script_dir / "run_vizard_demo.py"

# Use the Python executable from the active virtual environment
python_exec = sys.executable

# Regenerate NPZ files for every (method, controller, mode) combination
for method in methods:
    for controller in controllers:
        for mode in run_modes:
            args = [python_exec, str(vizard_demo_path), method, "--controller", controller, "--mode", mode]
            print("Running:", " ".join(args))
            subprocess.run(args, check=False)
    # Also generate ff_only with the default controller (legacy naming)
    args = [python_exec, str(vizard_demo_path), method, "--mode", "ff_only"]
    print("Running:", " ".join(args))
    subprocess.run(args, check=False)

print("All NPZ files regenerated. Now run scripts/run_mission.py to update plots.")
