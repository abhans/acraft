import math

import numpy as np

from core.config import Action, Drag, Physics


class Drone2D:
    def __init__(self, x, y):
        self.state = np.array([x, y, 0.0, 0.0, 0.0, 0.0], dtype=np.float64)
        self.action = Action()

    def step(self, action):
        self.action = self._handleAction(action)
        # 1. Convert 0-1 actions to actual force values
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
        omega += alpha * Physics.DT
        # Add the effective angular drag
        omega *= Drag.ANGULAR

        theta += omega * Physics.DT
        theta = (theta + math.pi) % (2 * math.pi) - math.pi

        # ------------------------ LINEAR DYNAMICS ------------------------
        thrustTotal = thrustLeft + thrustRight

        # Decompose thrust into world X and Y coordinates
        fThrustX = thrustTotal * math.sin(theta)
        fThrustY = -thrustTotal * math.cos(theta)  # Negative because PyGame Y is down

        # Add the influence of gravity
        fGravityY = Physics.GRAVITY

        # Calculate linear acceleration (Force / Mass)
        ax = fThrustX / Physics.MASS
        ay = (fThrustY + fGravityY) / Physics.MASS

        # Update linear velocity
        vx += ax * Physics.DT
        vy += ay * Physics.DT

        # Apply linear drag
        vx *= Drag.LINEAR
        vy *= Drag.LINEAR

        # Update position
        x += vx * Physics.DT
        y += vy * Physics.DT

        self.state = np.array([x, y, vx, vy, theta, omega], dtype=np.float64)
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
