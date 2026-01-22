import subprocess
import itertools
import sys

methods = ["unshaped", "fourth"]
controllers = ["standard_pd", "filtered_pd"]
run_modes = ["combined", "fb_only", "ff_only"]

# Use the correct path to vizard_demo.py
vizard_demo_path = "basilisk_simulation/vizard_demo.py"

# Use the current Python executable (from venv)
python_exec = sys.executable

# Regenerate all NPZ files for all methods/controllers/modes
for method in methods:
    for controller in controllers:
        for mode in run_modes:
            args = [python_exec, vizard_demo_path, method, "--controller", controller, "--mode", mode]
            print("Running:", " ".join(args))
            subprocess.run(args, check=False)
    # Also generate ff_only for method (legacy naming)
    args = [python_exec, vizard_demo_path, method, "--mode", "ff_only"]
    print("Running:", " ".join(args))
    subprocess.run(args, check=False)

print("All NPZ files regenerated. Now rerun mission_simulation.py to update plots.")
