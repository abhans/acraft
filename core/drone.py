import math

import numpy as np

from core.config import Action, Drag, Physics, Thrust


class Drone2D:
    def __init__(self, x, y):
        self.state = np.array([x, y, 0.0, 0.0, 0.0, 0.0], dtype=np.float64)
        self.action = Action()

    def step(self, action):
        self.action = self._handleAction(action)
        # User Actions
        uVertical = self.action.UP - self.action.DOWN  # Up/Down
        uLateral = self.action.RIGHT - self.action.LEFT  # Left/Right

        x, y, vx, vy, theta, omega = self.state

        # ---------------------------------------- Auto-Stabilization ----------------------------------------
        #  Automatic tilting in the direction (Inner-Loop P).
        targetTheta = uLateral * Physics.MAXTILT
        errorTheta = targetTheta - theta

        # Apply angular acceleration and air drag
        omega += errorTheta * Physics.ACCELERATION * Physics.DT
        omega *= Drag.ANGULAR

        # Update angle
        theta += omega * Physics.DT
        theta = (theta + math.pi) % (2 * math.pi) - math.pi

        # Calculate Forces based on tilt
        # Total thrust is hover base + player vertical input
        totalThrust = Thrust.HOVER + (uVertical * Thrust.MOVE)

        # Decompose thrust into X and Y world coordinates based on current tilt
        fThrustX = totalThrust * math.sin(theta)
        fThrustY = -totalThrust * math.cos(theta)

        # Add the force of gravity
        fGravityY = Physics.GRAVITY

        # Update velocities
        vx += (fThrustX / Physics.MASS) * Physics.DT
        vy += ((fThrustY + fGravityY) / Physics.MASS) * Physics.DT

        # Apply air drag
        #  Helps stabilize the drone and prevents perpetual motion.
        vx *= Drag.LINEAR
        vy *= Drag.LINEAR

        # Update Position
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
