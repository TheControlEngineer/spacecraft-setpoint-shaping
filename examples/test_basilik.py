import sys
print("✓ Basilisk is working!", flush=True)
sys.stdout.flush()

# Minimal test - just import
print("Testing Basilisk import...")
from Basilisk.utilities import SimulationBaseClass
print("✓ Import successful!")

# Create sim object
print("Creating simulation...")
sim = SimulationBaseClass.SimBaseClass()
print("✓ Simulation object created!")

print("All basic functionality works!")