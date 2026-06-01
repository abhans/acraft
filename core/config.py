import itertools
from dataclasses import dataclass

import numpy as np

_XGRID: list[int] = [400, 600, 1000, 1100]
_YGRID: list[int] = [400, 600, 650, 700]

WAYPOINTS: list[tuple[int, ...]] = [
    (X, Y) for X, Y in itertools.product(_XGRID, _YGRID)
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
    SPEED: float = 50.0             # Base wind speed (pixels/s^2)
    DIRECTION: float = 0.0          # Wind angle in radians (0.0 = Right, pi/2 = Up, pi = Left)
    GUST_STRENGTH: float = 20.0     # Maximum gust strength added to the base wind (pixels/s^2)
    DRIFT_SPEED: float = 0.005      # Drift speed for slow, constant wind changes (pixels/s^2)
    DRIFT_DIR_SPEED: float = 0.003  # How quickly the drift direction changes (radians/s)


@dataclass
class Physics:
    """
    Pixel-based Drone Physics
    We use pixels instead of meters to decouple visual rendering from strict SI units,
    making the simulation stable and intuitive for RL.
    """
    GRAVITY: float = 880.0          # Effective Gravity  (pixels/s^2)
    MASS: float = 1.0               # Mass of the drone (kg)
    INERTIA: float = 25.0           # Moment of inertia for rotation (kg*pixels^2)
    MAX_THRUST: float = 990.0       # Max force per rotor (pixels/s^2)
    ARM_LENGTH: int = 30            # Visual length of drone arm (pixels)
    ROTOR_RADIUS: int = 8           # Visual size of rotors
    FPS: float = 60.0
    DT: float = 1.0 / FPS


@dataclass
class Drag:
    """
    Drag coefficients for linear and angular movement.
    These values help stabilize the drone and prevent perpetual motion.
    """
    LINEAR: float = 0.5
    ANGULAR: float = 0.65


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
    BLACK = Color(40, 40, 40)
    WHITE = Color(255, 255, 255)
    RED = Color(252, 73, 52)
    BLUE = Color(131, 165, 152)
    GRAY = Color(100, 100, 100)
    GREEN = Color(184, 187, 38)
    ORANGE = Color(255, 165, 0)
    YELLOW = Color(250, 189, 47)
