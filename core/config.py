from dataclasses import dataclass

import numpy as np


@dataclass
class Screen:
    WIDTH: int = 1200
    HEIGHT: int = 800
    MARGIN: int = 20


@dataclass
class Physics:
    """
    Pixel-based Drone Physics
    We use pixels instead of meters to decouple visual rendering from strict SI units,
    making the simulation stable and intuitive for RL.
    """

    GRAVITY: float = 980.0  # Effective Gravity  (pixels/s^2)
    MASS: float = 5.0  # Mass of the drone (kg)
    INERTIA: float = 20.0  # Moment of inertia for rotation (kg*pixels^2)
    MAX_THRUST: float = 600.0  # Max force per rotor (pixels/s^2)
    ARM_LENGTH: int = 30  # Visual length of drone arm in pixels
    ROTOR_RADIUS: int = 8  # Visual size of rotors
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

    LINEAR: float = 0.99
    ANGULAR: float = 0.98


@dataclass
class Action:
    """
    'D Abstracted Action Space for the Drone.
    Values are expected to be between 0.0 (off) and 1.0 (full thrust).
    """

    LEFT: float = 0.0
    RIGHT: float = 0.0

    def to_numpy(self) -> np.ndarray:
        """Converts the named actions into a 1D NumPy array for Physics/PyTorch."""
        return np.array([self.LEFT, self.RIGHT], dtype=np.float64)

    def __iter__(self):
        """Allows unpacking like: up, down, left, right = action"""
        return iter((self.LEFT, self.RIGHT))


@dataclass
class Colors:
    BLACK: tuple = (0, 0, 0)
    WHITE: tuple = (255, 255, 255)
    RED: tuple = (255, 50, 50)
    BLUE: tuple = (50, 50, 255)
    GRAY: tuple = (100, 100, 100)
    GREEN: tuple = (50, 255, 50)
    YELLOW: tuple = (255, 255, 50)
