"""
Flexible Spacecraft Model for Basilisk Simulation

Assembles a complete Basilisk spacecraft object that includes:
    - Rigid hub with principal inertia aligned to body axes
    - 3 axis reaction wheel pyramid (two canted + one axial)
    - 2 deployable solar array wings, each carrying two bending modes
      modeled as linear spring mass damper (LSMD) state effectors
    - Perfect navigation sensor (ideal attitude/rate knowledge)

Physics of yaw coupling (Z axis slew):
    Solar arrays extend along the body Y axis (port/starboard).  When the
    hub undergoes angular acceleration about Z, the array masses experience
    a tangential inertial load in the X direction.  This base excitation
    drives transverse bending of the panels, producing the classic residual
    vibration problem that input shaping is designed to suppress.

All physical constants (inertia, modal mass, attachment locations) are
imported from spacecraft_properties.py so that feedforward and feedback
modules share a single source of truth.
"""

# ── Basilisk framework imports ───────────────────────────────────────────────
from Basilisk.utilities import SimulationBaseClass
from Basilisk.utilities import macros
from Basilisk.simulation import spacecraft
from Basilisk.simulation import reactionWheelStateEffector
from Basilisk.simulation import simpleNav
from Basilisk.architecture import messaging

import numpy as np

# Shared physical constants so every module uses the same numbers
from .spacecraft_properties import (
    FLEX_MODE_LOCATIONS,
    FLEX_MODE_MASS,
    HUB_INERTIA,
    compute_effective_inertia as compute_effective_inertia_base,
)


# ═══════════════════════════════════════════════════════════════════════════
#  Flexible Spacecraft Model
# ═══════════════════════════════════════════════════════════════════════════

class FlexibleSpacecraft:
    """
    Spacecraft model with flexible solar array appendages.

    Wraps Basilisk's ``Spacecraft``, ``ReactionWheelStateEffector``, and
    ``LinearSpringMassDamper`` objects into a single configurable class.
    Parameters are representative of a small observation satellite (~750 kg)
    with two deployable solar wings (~50 kg each).  Modal frequencies and
    damping ratios match published ranges for honeycomb panel arrays.

    Typical usage::

        sc = FlexibleSpacecraft()
        sc.create_rigid_spacecraft()
        sc.add_reaction_wheels()
        sc.add_flexible_solar_arrays()
        sc.add_simple_nav()
    """
    
    def __init__(self):
        """Initialise spacecraft parameters from shared constants.

        No Basilisk objects are created here; call the ``create_*`` and
        ``add_*`` methods to build the simulation tree.
        """

        # ── Hub properties ───────────────────────────────────────────────
        # Typical small observation satellite main body
        self.hub_mass = 750.0  # kg main body mass
        self.hub_inertia = HUB_INERTIA.tolist()  # kg*m^2 principal inertias
        self.modal_mass = FLEX_MODE_MASS
        # Deep copy so callers can modify locations without affecting the module constant
        self.flex_mode_locations = {name: loc.copy() for name, loc in FLEX_MODE_LOCATIONS.items()}

        # ── Solar array parameters ───────────────────────────────────────
        # Each wing is about 50 kg deployed mass; only a fraction (modal mass)
        # participates in each bending mode
        self.array_mass = 50.0  # kg per wing

        # Two dominant bending modes targeted by the input shaper.
        # Frequencies and damping ratios are within the range typical of
        # honeycomb panel arrays on LEO spacecraft.
        self.array_modes = [
            {'frequency': 0.4, 'damping': 0.02, 'name': 'First bending'},
            {'frequency': 1.3, 'damping': 0.015, 'name': 'Second bending'}
        ]

        # ── Reaction wheel sizing ────────────────────────────────────────
        # 70 Nm max torque provides enough authority for a 180 deg yaw
        # slew in approximately 30 s with margin for flex compensation
        self.rw_max_torque = 70.0  # Nm peak torque per wheel
        self.rw_max_momentum = 10.0  # Nms storage capacity
        self.rw_inertia = 0.05  # kg*m^2 spin axis inertia per wheel

        # ── Basilisk object handles (populated by builder methods) ───────
        self.scObject = None
        self.rwStateEffector = None
        self.simpleNavObject = None
        
    # ─────────────────────────────────────────────────────────────────────
    #  Builder methods — call in order: rigid hub -> RW -> flex -> nav
    # ─────────────────────────────────────────────────────────────────────

    def create_rigid_spacecraft(self):
        """Create the rigid hub (main body) without flexible dynamics.

        Flexible modes are attached later via
        :meth:`add_flexible_solar_arrays`, which adds
        ``LinearSpringMassDamper`` state effectors to this hub.

        Returns
        -------
        spacecraft.Spacecraft
            The Basilisk spacecraft object with rigid hub configured.
        """
        scObject = spacecraft.Spacecraft()
        scObject.ModelTag = "FlexibleSpacecraft"

        # Set hub mass properties.  Centre of mass coincides with the body
        # frame origin so that RW and flex torques have no cross coupling
        # from lever arm effects.
        scObject.hub.mHub = self.hub_mass
        scObject.hub.r_BcB_B = [[0.0], [0.0], [0.0]]  # CoM at body origin
        scObject.hub.IHubPntBc_B = self.hub_inertia

        self.scObject = scObject
        return scObject
    
    def add_reaction_wheels(self):
        """Add a 3 wheel pyramid reaction wheel array.

        Layout:
            - Wheel 1: canted +X +Y at 45 deg in the X Y plane
            - Wheel 2: canted -X +Y at 45 deg (orthogonal to Wheel 1)
            - Wheel 3: aligned with body Z axis

        The two canted wheels provide coupled X and Y torque, while Wheel 3
        gives direct yaw authority.  Together they deliver full 3 axis
        control.  For a pure yaw slew, most of the torque comes from
        Wheel 3.

        Returns
        -------
        reactionWheelStateEffector.ReactionWheelStateEffector
            The configured RW effector, already attached to the spacecraft.
        """
        rwStateEffector = reactionWheelStateEffector.ReactionWheelStateEffector()
        rwStateEffector.ModelTag = "RW_Pyramid"

        # Wheel 1: spin axis in the +X +Y direction (45 deg cant)
        RW1 = reactionWheelStateEffector.RWConfigPayload()
        RW1.gsHat_B = [[np.sqrt(2)/2], [np.sqrt(2)/2], [0.0]]
        RW1.Js = self.rw_inertia
        RW1.u_max = self.rw_max_torque

        # Wheel 2: spin axis in the -X +Y direction (symmetric to RW1)
        RW2 = reactionWheelStateEffector.RWConfigPayload()
        RW2.gsHat_B = [[-np.sqrt(2)/2], [np.sqrt(2)/2], [0.0]]
        RW2.Js = self.rw_inertia
        RW2.u_max = self.rw_max_torque

        # Wheel 3: aligned with body Z, primary yaw actuator
        RW3 = reactionWheelStateEffector.RWConfigPayload()
        RW3.gsHat_B = [[0.0], [0.0], [1.0]]
        RW3.Js = self.rw_inertia
        RW3.u_max = self.rw_max_torque

        # Register each wheel with the effector
        rwStateEffector.addReactionWheel(RW1)
        rwStateEffector.addReactionWheel(RW2)
        rwStateEffector.addReactionWheel(RW3)

        # Attach the effector to the hub so Basilisk propagates the
        # coupled wheel + body dynamics
        self.scObject.addStateEffector(rwStateEffector)

        self.rwStateEffector = rwStateEffector
        return rwStateEffector
    
    def add_simple_nav(self):
        """Add a perfect navigation sensor (no noise, no latency).

        Using ideal attitude and rate knowledge isolates the effect of
        the feedforward trajectory and input shaper from navigation
        errors, making it easier to evaluate shaping performance.

        Returns
        -------
        simpleNav.SimpleNav
            The navigation object providing truth state output messages.
        """
        simpleNavObject = simpleNav.SimpleNav()
        simpleNavObject.ModelTag = "SimpleNav"

        self.simpleNavObject = simpleNavObject
        return simpleNavObject
    
    def add_flexible_solar_arrays(self):
        """Attach flexible solar arrays as spring mass damper state effectors.

        Each wing carries two bending modes (fundamental + second harmonic),
        giving four LSMD effectors total (port x2 + starboard x2).

        Modal frequencies and damping:
            Mode 1  0.4 Hz, zeta = 0.02  (fundamental out of plane bending)
            Mode 2  1.3 Hz, zeta = 0.015 (second harmonic)

        Yaw coupling geometry:
            Arrays lie along the body Y axis.  Yaw (Z axis rotation) imposes
            a tangential inertial load on the array masses in the X direction,
            so the oscillation direction ``pHat_B`` is set to [1, 0, 0].
            This is the coupling path that input shaping must cancel.

        Spring and damper coefficients are derived from the standard
        relations for a single degree of freedom oscillator:
            k = m * omega^2
            c = 2 * zeta * sqrt(k * m)

        Returns
        -------
        dict
            Mapping of mode name to ``LinearSpringMassDamper`` object.
        """
        from Basilisk.simulation import linearSpringMassDamper

        # Effective participating mass per mode (~10% of physical wing mass)
        modal_mass = self.modal_mass  # kg
        
        # ── Precompute natural frequencies and damping ratios ────────────
        omega1 = 2 * np.pi * self.array_modes[0]['frequency']  # rad/s, mode 1
        zeta1 = self.array_modes[0]['damping']
        omega2 = 2 * np.pi * self.array_modes[1]['frequency']  # rad/s, mode 2
        zeta2 = self.array_modes[1]['damping']

        # ── Mode 1, port wing (fundamental bending at 0.4 Hz) ───────────
        mode1_port = linearSpringMassDamper.LinearSpringMassDamper()
        mode1_port.ModelTag = "mode1_port"
        mode1_port.k = modal_mass * omega1**2           # restoring stiffness
        mode1_port.c = 2 * zeta1 * np.sqrt(mode1_port.k * modal_mass)  # viscous damping
        mode1_port.massInit = modal_mass
        r_mode1_port = self.flex_mode_locations["mode1_port"]
        mode1_port.r_PB_B = [[float(r_mode1_port[0])], [float(r_mode1_port[1])], [float(r_mode1_port[2])]]
        # Oscillation along X: tangential to yaw rotation for masses on Y axis
        mode1_port.pHat_B = [[1.0], [0.0], [0.0]]
        mode1_port.rhoInit = 0.0     # zero initial displacement
        mode1_port.rhoDotInit = 0.0  # zero initial velocity
        self.scObject.addStateEffector(mode1_port)
        
        # ── Mode 2, port wing (second bending at 1.3 Hz) ─────────────
        mode2_port = linearSpringMassDamper.LinearSpringMassDamper()
        mode2_port.ModelTag = "mode2_port"
        mode2_port.k = modal_mass * omega2**2
        mode2_port.c = 2 * zeta2 * np.sqrt(mode2_port.k * modal_mass)
        mode2_port.massInit = modal_mass
        r_mode2_port = self.flex_mode_locations["mode2_port"]
        mode2_port.r_PB_B = [[float(r_mode2_port[0])], [float(r_mode2_port[1])], [float(r_mode2_port[2])]]
        mode2_port.pHat_B = [[1.0], [0.0], [0.0]]
        mode2_port.rhoInit = 0.0
        mode2_port.rhoDotInit = 0.0
        self.scObject.addStateEffector(mode2_port)
        
        # ── Mode 1, starboard wing (symmetric to port mode 1) ────────
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
        
        # ── Mode 2, starboard wing (symmetric to port mode 2) ────────
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

        # Keep handles so other modules can read back modal state (rho, rhoDot)
        self.flexModes = {
            'mode1_port': mode1_port,
            'mode2_port': mode2_port,
            'mode1_stbd': mode1_stbd,
            'mode2_stbd': mode2_stbd
        }

        return self.flexModes
    
    # ─────────────────────────────────────────────────────────────────────
    #  Query / utility methods
    # ─────────────────────────────────────────────────────────────────────

    def get_info(self):
        """Print a one line summary of hub, modal, and actuator parameters."""
        print("\nFlexible Spacecraft Configuration:")
        print(f"  Hub: {self.hub_mass} kg, Inertia: {self.hub_inertia[0][0]}, {self.hub_inertia[1][1]}, {self.hub_inertia[2][2]} kg*m^2")
        print(f"  Modes: {self.array_modes[0]['frequency']} Hz (zeta={self.array_modes[0]['damping']}), {self.array_modes[1]['frequency']} Hz (zeta={self.array_modes[1]['damping']})")
        print(f"  RW: {self.rw_max_torque} Nm max torque, {self.rw_max_momentum} Nms max momentum")

    def compute_effective_inertia(self, include_flex=True):
        """Return the 3x3 inertia tensor, optionally augmented by flex masses.

        When *include_flex* is True the parallel axis contribution of each
        modal mass at its attachment location is added to the hub inertia.
        This gives the total effective inertia that the feedforward
        trajectory planner should use for torque sizing.

        Parameters
        ----------
        include_flex : bool
            If False, return the bare hub inertia with no appendages.

        Returns
        -------
        np.ndarray
            3x3 inertia tensor in [kg*m^2].
        """
        if not include_flex:
            return np.array(self.hub_inertia, dtype=float)
        # Delegates to the shared helper so the parallel axis calculation
        # stays consistent with all other modules
        return compute_effective_inertia_base(
            hub_inertia=self.hub_inertia,
            mode_locations=self.flex_mode_locations,
            modal_mass=self.modal_mass,
        )


# ═══════════════════════════════════════════════════════════════════════════
#  Standalone smoke test
# ═══════════════════════════════════════════════════════════════════════════

def test_rigid_spacecraft():
    """Smoke test: build the full model and verify all objects instantiate."""

    # Instantiate with default parameters and print summary
    sc = FlexibleSpacecraft()
    sc.get_info()

    # Walk through the builder sequence (order matters: hub first, then
    # effectors that attach to the hub, then standalone nav sensor)
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
