"""Regenerate all simulation data files."""
import os
import subprocess
import sys
from pathlib import Path

methods = ["s_curve", "fourth"]
controllers = ["standard_pd", "filtered_pd"]
run_modes = ["combined", "fb_only", "ff_only"]

# Use the path relative to this script
script_dir = Path(__file__).parent.resolve()
vizard_demo_path = script_dir / "run_vizard_demo.py"

# Use the current Python executable (from venv)
python_exec = sys.executable

# Regenerate all NPZ files for all methods/controllers/modes
for method in methods:
    for controller in controllers:
        for mode in run_modes:
            args = [python_exec, str(vizard_demo_path), method, "--controller", controller, "--mode", mode]
            print("Running:", " ".join(args))
            subprocess.run(args, check=False)
    # Also generate ff_only for method (legacy naming)
    args = [python_exec, str(vizard_demo_path), method, "--mode", "ff_only"]
    print("Running:", " ".join(args))
    subprocess.run(args, check=False)

print("All NPZ files regenerated. Now run scripts/run_mission.py to update plots.")
