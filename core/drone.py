import math

import numpy as np

from core.config import Action, Drag, Physics

"""
2D Drone Model with Pixel-Based Physics
This class simulates a 2D drone with realistic physics using pixel units.

------------ STATES ------------
- x, y: Position of the drone's center of mass (pixels)
- vx, vy: Linear velocity (pixels/s)
- theta: Orientation angle (radians)
- omega: Angular velocity (radians/s)

------------ ACTIONS ------------
- LEFT: Thrust level for the left rotor (0.0 to 1.0)
- RIGHT: Thrust level for the right rotor (0.0 to 1.0)
"""


class Drone:
    def __init__(self, x, y):
        self.state = np.array([x, y, 0.0, 0.0, 0.0, 0.0], dtype=np.float64)
        self.action = Action()

    def step(self, action):
        self.action = self._handleAction(action)
        # Convert 0-1 actions to actual force values
        thrustLeft = self.action.LEFT * Physics.MAX_THRUST
        thrustRight = self.action.RIGHT * Physics.MAX_THRUST

        x, y, vx, vy, theta, omega = self.state

        # ------------------------ ROTATIONAL DYNAMICS ------------------------
        # ! The hard part for GAIL to learn.
        # Calculate the effective torque (Force * Arm Length)
        # * Positive = Clockwise
        torque = Physics.ARM_LENGTH * (thrustRight - thrustLeft)

        # Angular Acceleration (Torque / Inertia)
        alpha = torque / Physics.INERTIA
        omega += (alpha * Physics.DT) * Drag.ANGULAR

        theta += omega * Physics.DT
        theta = (theta + math.pi) % (2 * math.pi) - math.pi

        # ------------------------ LINEAR DYNAMICS ------------------------
        thrustTotal = thrustLeft + thrustRight

        # Decompose thrust into world X and Y coordinates
        fThrustX = thrustTotal * math.sin(theta)
        fThrustY = -thrustTotal * math.cos(theta)

        # Linear acceleration (Force / Mass)
        ax = fThrustX / Physics.MASS
        ay = (fThrustY + Physics.GRAVITY) / Physics.MASS

        # Update linear velocity
        vx += (ax * Physics.DT) * Drag.LINEAR
        vy += (ay * Physics.DT) * Drag.LINEAR

        # Update position
        x += vx * Physics.DT
        y += vy * Physics.DT

        self.state = np.array([x, y, vx, vy, theta, omega], dtype=np.float32)
        return self.state

    def _handleAction(self, action):
        """Converts abstracted Action into drone control inputs."""
        if isinstance(action, Action):
            return action

        # If the action is a 4D NumPy array (from PyTorch),
        # convert it to Action dataclass
        elif isinstance(action, np.ndarray) and action.shape == (4,):
            return Action(*action)

        else:
            raise ValueError(
                "[ERROR] Invalid action format."
                "Expected Action dataclass or 4D NumPy array but got: {}".format(
                    type(action)
                )
            )
