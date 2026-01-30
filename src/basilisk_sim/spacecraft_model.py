"""
Flexible Spacecraft Model for Basilisk Simulation

This module defines a spacecraft with:
- Rigid hub with specified mass/inertia
- 3-axis reaction wheel array
- 2 flexible solar array appendages with modal dynamics

For YAW (Z-axis) slew maneuvers:
- Solar arrays extend along Y-axis (port/starboard)
- Flex modes bend in X direction (tangential to yaw)
- Slew start/stop (angular acceleration) excites tangential bending via base excitation
- This is the classic input shaping problem for flexible spacecraft
"""

from Basilisk.utilities import SimulationBaseClass
from Basilisk.utilities import macros
from Basilisk.simulation import spacecraft
from Basilisk.simulation import reactionWheelStateEffector
from Basilisk.simulation import simpleNav
from Basilisk.architecture import messaging

import numpy as np

from .spacecraft_properties import (
    FLEX_MODE_LOCATIONS,
    FLEX_MODE_MASS,
    HUB_INERTIA,
    compute_effective_inertia as compute_effective_inertia_base,
)


class FlexibleSpacecraft:
    """
    Spacecraft model with flexible solar arrays.
    
    This is based on a typical small satellite with deployable arrays.
    The modal frequencies and damping values are representative of
    real solar array dynamics.
    """
    
    def __init__(self):
        """Set up spacecraft parameters."""
        
        # Hub properties - these are typical for a small observation satellite
        self.hub_mass = 750.0  # kg - main body mass
        self.hub_inertia = HUB_INERTIA.tolist()  # kg*m^2 - principal inertias
        self.modal_mass = FLEX_MODE_MASS
        self.flex_mode_locations = {name: loc.copy() for name, loc in FLEX_MODE_LOCATIONS.items()}
        
        # Solar array parameters
        # Each array is about 50 kg deployed mass
        self.array_mass = 50.0  # kg each
        
        # Modal data - these are the two dominant bending modes we want to suppress
        # First mode is typically the fundamental bending, second is a higher harmonic
        self.array_modes = [
            {'frequency': 0.4, 'damping': 0.02, 'name': 'First bending'},
            {'frequency': 1.3, 'damping': 0.015, 'name': 'Second bending'}
        ]
        
        # Reaction wheel sizing
        # 70 Nm is enough to do a 180 deg slew in about 30 seconds
        self.rw_max_torque = 70.0  # Nm
        self.rw_max_momentum = 10.0  # Nms
        self.rw_inertia = 0.05  # kg*m^2 per wheel
        
        # Basilisk objects (created later)
        self.scObject = None
        self.rwStateEffector = None
        self.simpleNavObject = None
        
    def create_rigid_spacecraft(self):
        """
        Create the rigid hub - this is the main body without flexible dynamics.
        Flexible modes are added separately via add_flexible_solar_arrays().
        """
        scObject = spacecraft.Spacecraft()
        scObject.ModelTag = "FlexibleSpacecraft"
        
        # Hub mass and inertia
        scObject.hub.mHub = self.hub_mass
        scObject.hub.r_BcB_B = [[0.0], [0.0], [0.0]]  # CoM at geometric center
        scObject.hub.IHubPntBc_B = self.hub_inertia
        
        self.scObject = scObject
        return scObject
    
    def add_reaction_wheels(self):
        """
        Add a 3-wheel pyramid configuration.
        
        Two wheels are canted at 45 deg in the X-Y plane, one is along Z.
        This gives full 3-axis control authority.
        """
        rwStateEffector = reactionWheelStateEffector.ReactionWheelStateEffector()
        rwStateEffector.ModelTag = "RW_Pyramid"
        
        # Wheel 1: Canted +X+Y
        RW1 = reactionWheelStateEffector.RWConfigPayload()
        RW1.gsHat_B = [[np.sqrt(2)/2], [np.sqrt(2)/2], [0.0]]
        RW1.Js = self.rw_inertia
        RW1.u_max = self.rw_max_torque
        
        # Wheel 2: Canted -X+Y
        RW2 = reactionWheelStateEffector.RWConfigPayload()
        RW2.gsHat_B = [[-np.sqrt(2)/2], [np.sqrt(2)/2], [0.0]]
        RW2.Js = self.rw_inertia
        RW2.u_max = self.rw_max_torque
        
        # Wheel 3: Body Z
        RW3 = reactionWheelStateEffector.RWConfigPayload()
        RW3.gsHat_B = [[0.0], [0.0], [1.0]]
        RW3.Js = self.rw_inertia
        RW3.u_max = self.rw_max_torque
        
        # Add wheels to effector
        rwStateEffector.addReactionWheel(RW1)
        rwStateEffector.addReactionWheel(RW2)
        rwStateEffector.addReactionWheel(RW3)
        
        # Attach to spacecraft
        self.scObject.addStateEffector(rwStateEffector)
        
        self.rwStateEffector = rwStateEffector
        return rwStateEffector
    
    def add_simple_nav(self):
        """
        Add navigation sensor with perfect knowledge.
        For testing we use ideal sensing to isolate control effects.
        """
        simpleNavObject = simpleNav.SimpleNav()
        simpleNavObject.ModelTag = "SimpleNav"
        
        self.simpleNavObject = simpleNavObject
        return simpleNavObject
    
    def add_flexible_solar_arrays(self):
        """
        Add flexible solar arrays modeled as spring-mass-damper systems.
        
        Each array has two modes:
            Mode 1: 0.4 Hz (first bending)
            Mode 2: 1.3 Hz (second bending)
        
        Array geometry for YAW coupling:
            - Arrays extend along Y-axis (port/starboard at y = ±3.5m, ±4.5m)
            - Yaw is rotation about Z-axis
            - For masses on Y-axis, yaw acceleration creates tangential force in X direction
            - Therefore pHat_B = [1,0,0] for proper yaw-to-bending coupling
        
        This coupling is what makes input shaping necessary for precision pointing.
        """
        from Basilisk.simulation import linearSpringMassDamper
        
        # Effective mass participating in each mode (about 10% of array mass)
        modal_mass = self.modal_mass  # kg
        
        # Mode 1: First bending (0.4 Hz) - Port side array
        mode1_port = linearSpringMassDamper.LinearSpringMassDamper()
        mode1_port.ModelTag = "mode1_port"
        omega1 = 2 * np.pi * self.array_modes[0]['frequency']
        zeta1 = self.array_modes[0]['damping']
        mode1_port.k = modal_mass * omega1**2  # k = m * omega^2
        mode1_port.c = 2 * zeta1 * np.sqrt(mode1_port.k * modal_mass)  # damping coefficient formula
        mode1_port.massInit = modal_mass
        r_mode1_port = self.flex_mode_locations["mode1_port"]
        mode1_port.r_PB_B = [[float(r_mode1_port[0])], [float(r_mode1_port[1])], [float(r_mode1_port[2])]]
        # For yaw (Z-rotation) with mass on Y-axis, tangential direction is X
        mode1_port.pHat_B = [[1.0], [0.0], [0.0]]
        mode1_port.rhoInit = 0.0
        mode1_port.rhoDotInit = 0.0
        self.scObject.addStateEffector(mode1_port)
        
        # Mode 2: Second bending (1.3 Hz) - Port side
        mode2_port = linearSpringMassDamper.LinearSpringMassDamper()
        mode2_port.ModelTag = "mode2_port"
        omega2 = 2 * np.pi * self.array_modes[1]['frequency']
        zeta2 = self.array_modes[1]['damping']
        mode2_port.k = modal_mass * omega2**2
        mode2_port.c = 2 * zeta2 * np.sqrt(mode2_port.k * modal_mass)
        mode2_port.massInit = modal_mass
        r_mode2_port = self.flex_mode_locations["mode2_port"]
        mode2_port.r_PB_B = [[float(r_mode2_port[0])], [float(r_mode2_port[1])], [float(r_mode2_port[2])]]
        mode2_port.pHat_B = [[1.0], [0.0], [0.0]]
        mode2_port.rhoInit = 0.0
        mode2_port.rhoDotInit = 0.0
        self.scObject.addStateEffector(mode2_port)
        
        # Mode 1: First bending - Starboard side
        mode1_stbd = linearSpringMassDamper.LinearSpringMassDamper()
        mode1_stbd.ModelTag = "mode1_starboard"
        mode1_stbd.k = modal_mass * omega1**2
        mode1_stbd.c = 2 * zeta1 * np.sqrt(mode1_stbd.k * modal_mass)
        mode1_stbd.massInit = modal_mass
        r_mode1_stbd = self.flex_mode_locations["mode1_stbd"]
        mode1_stbd.r_PB_B = [[float(r_mode1_stbd[0])], [float(r_mode1_stbd[1])], [float(r_mode1_stbd[2])]]
        mode1_stbd.pHat_B = [[1.0], [0.0], [0.0]]
        mode1_stbd.rhoInit = 0.0
        mode1_stbd.rhoDotInit = 0.0
        self.scObject.addStateEffector(mode1_stbd)
        
        # Mode 2: Second bending - Starboard side
        mode2_stbd = linearSpringMassDamper.LinearSpringMassDamper()
        mode2_stbd.ModelTag = "mode2_starboard"
        mode2_stbd.k = modal_mass * omega2**2
        mode2_stbd.c = 2 * zeta2 * np.sqrt(mode2_stbd.k * modal_mass)
        mode2_stbd.massInit = modal_mass
        r_mode2_stbd = self.flex_mode_locations["mode2_stbd"]
        mode2_stbd.r_PB_B = [[float(r_mode2_stbd[0])], [float(r_mode2_stbd[1])], [float(r_mode2_stbd[2])]]
        mode2_stbd.pHat_B = [[1.0], [0.0], [0.0]]
        mode2_stbd.rhoInit = 0.0
        mode2_stbd.rhoDotInit = 0.0
        self.scObject.addStateEffector(mode2_stbd)
        
        # Store references for later access
        self.flexModes = {
            'mode1_port': mode1_port,
            'mode2_port': mode2_port,
            'mode1_stbd': mode1_stbd,
            'mode2_stbd': mode2_stbd
        }
        
        return self.flexModes
    
    def get_info(self):
        """Print spacecraft configuration summary."""
        print("\nFlexible Spacecraft Configuration:")
        print(f"  Hub: {self.hub_mass} kg, Inertia: {self.hub_inertia[0][0]}, {self.hub_inertia[1][1]}, {self.hub_inertia[2][2]} kg*m^2")
        print(f"  Modes: {self.array_modes[0]['frequency']} Hz (zeta={self.array_modes[0]['damping']}), {self.array_modes[1]['frequency']} Hz (zeta={self.array_modes[1]['damping']})")
        print(f"  RW: {self.rw_max_torque} Nm max torque, {self.rw_max_momentum} Nms max momentum")

    def compute_effective_inertia(self, include_flex=True):
        """Compute inertia including nominal flexible mode masses."""
        if not include_flex:
            return np.array(self.hub_inertia, dtype=float)
        return compute_effective_inertia_base(
            hub_inertia=self.hub_inertia,
            mode_locations=self.flex_mode_locations,
            modal_mass=self.modal_mass,
        )


def test_rigid_spacecraft():
    """Quick test to verify the spacecraft model builds correctly."""
    
    # Create and configure spacecraft
    sc = FlexibleSpacecraft()
    sc.get_info()
    
    # Build Basilisk objects
    scObject = sc.create_rigid_spacecraft()
    rwStateEffector = sc.add_reaction_wheels()
    flexModes = sc.add_flexible_solar_arrays()
    navObject = sc.add_simple_nav()
    
    print("\nSpacecraft model built successfully!")
    print(f"  Hub: {scObject.ModelTag}")
    print(f"  Flexible modes: {len(flexModes)}")
    
    return sc


if __name__ == "__main__":
    spacecraft = test_rigid_spacecraft()
