import math
from dataclasses import dataclass

import numpy as np

WAYPOINTS: list[tuple[int, ...]] = [
    # -------- Boundary & Sweeping Turns --------
    (1100, 200),    # Test boundary approach
    (1000, 650),    # Test steep 45-degree dive
    (400, 700),     # Test long lateral translation

    # -------- Extreme Altitudes & Verticals --------
    (100, 150),     # Test extreme boundary
    (100, 650),     # Test pure vertical drop

    # -------- Symmetry & Precision --------
    (1000, 150),    # Test horizontal sweep across the top
    (1000, 650),    # Test pure vertical drop

    # -------- Center Precision & Hovering --------
    (600, 200),     # Test deadzone and noise on Y-axis
    (600, 600),     # Test pure vertical descent
    (600, 400),
]


@dataclass
class Screen:
    WIDTH: int = 1200
    HEIGHT: int = 800
    MARGIN: int = 20


@dataclass
class Wind:
    """
    Wind parameters simulating aerodynamic forces.
    """
    MAX_WIND_DIR: float = math.pi   # Maximum wind direction in radians (360 degrees)
    SPEED: float = 150.0            # Base wind speed (pixels/s^2)
    DIRECTION: float = 0.0          # Wind angle in radians (0.0 = Right, pi/2 = Up, pi = Left)
    TURBULENCE: float = 0.1         # How rapidly the wind changes speed and direction over time
    GUST_STRENGTH: float = 8.0      # Maximum gust strength added to the base wind (pixels/s^2)
    DRIFT_SPEED: float = 0.005      # Drift speed for slow, constant wind changes (pixels/s^2)
    DRIFT_DIR_SPEED: float = 0.003  # How quickly the drift direction changes (radians/s)


@dataclass
class Physics:
    """
    Pixel-based Drone Physics
    We use pixels instead of meters to decouple visual rendering from strict SI units,
    making the simulation stable and intuitive for RL.
    """
    GRAVITY: float = 880.0              # Effective Gravity  (pixels/s^2)
    MASS: float = 1.0                   # Mass of the drone (kg)
    INERTIA: float = 20.0               # Moment of inertia for rotation (kg*pixels^2)
    MAX_THRUST: float = 600.0           # Max force per rotor (pixels/s^2)
    MAX_TILT: float = np.radians(40)    # Max tilt angle (radians)
    ARM_LENGTH: int = 30                # Visual length of drone arm (pixels)
    ROTOR_RADIUS: int = 8               # Visual size of rotors
    FPS: float = 60.0
    DT: float = 1.0 / FPS


@dataclass
class Thrust:
    """
    Thrust values for hover and movement.
    HOVER is the thrust needed to counteract gravity and maintain altitude.
    MOVE is the additional thrust applied when the player gives input.
    """
    HOVER: float = Physics.GRAVITY
    MOVE: float = 100.0


@dataclass
class Drag:
    """
    Drag coefficients for linear and angular movement.
    These values help stabilize the drone and prevent perpetual motion.
    """
    LINEAR: float = 0.75
    ANGULAR: float = 0.77


@dataclass
class Action:
    """
    'D Abstracted Action Space for the Drone.
    Values are expected to be between 0.0 (off) and 1.0 (full thrust).
    """
    LEFT: float = 0.0
    RIGHT: float = 0.0

    def toNumpy(self) -> np.ndarray:
        """Converts the named actions into a 1D NumPy array for Physics/PyTorch."""
        return np.array([self.LEFT, self.RIGHT], dtype=np.float64)

    def __iter__(self):
        """Allows unpacking like: up, down, left, right = action"""
        return iter((self.LEFT, self.RIGHT))


@dataclass
class Color:
    """
    Simple RGB color representation.
    Values range from 0 to 255.
    """
    R: int
    G: int
    B: int


@dataclass
class Colors:
    """
    Predefined colors for rendering the drone and environment.
    Used to visually indicate different states (e.g., thrust levels).
    """
    BLACK = Color(0, 0, 0)
    WHITE = Color(255, 255, 255)
    RED = Color(255, 50, 50)
    BLUE = Color(50, 50, 255)
    GRAY = Color(100, 100, 100)
    GREEN = Color(50, 255, 50)
    ORANGE = Color(255, 165, 0)
    YELLOW = Color(255, 255, 50)
