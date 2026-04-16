import numpy as np

from core.config import WAYPOINTS, Action, Physics
from core.controller import PIDController

"""
Expert Controller for 2D Drone Positioning Simulator

This class implements a PID-based expert controller to simulate a skilled human pilot 
with some imperfections.

It's used to collect expert data to train the GAIL agent.

------------ Human-Like Flaws ------------
- Deadzones: Small errors are ignored to simulate human control imperfections.
- Motor Noise: Random noise added to motor outputs to simulate imperfect human control.
"""


class Expert:
    def __init__(self):
        # ------------------------ Outer Loop (Tilt & Thrust) ------------------------
        self.yPID = PIDController(kP=3.5, kD=3.0, limit=400.0)
        self.xPID = PIDController(kP=0.012, kD=0.04, limit=0.5)

        # ------------------------ Inner Loop (Desired Tilt to Motor Differential) ------------------------
        self.thetaPID = PIDController(kP=200.0, kD=95.0, limit=700.0)

        # Waypoints for the expert to follow.
        self.waypoints = WAYPOINTS
        self.currWaypointIdx = 0
        self.waypointThresh = 10.0  # In Pixels

        # ------------------------ Human-Expert Flaws ------------------------
        # Deadzones to simulate imperfect human control.
        #   Small errors within the deadzone are ignored by expert.
        self._positionDeadzone = 1.0
        self._angleDeadzone = np.radians(2.5)
        # Random noise added to motor outputs to simulate imperfect human control
        self._motorNoise = 0.0125

    def _applyDeadzone(self, error, threshold=0.05):
        if abs(error) < threshold:
            return 0.0

        return error - (np.sign(error) * threshold)

    def getAction(self, state, dt):
        x, y, vx, vy, theta, omega = state
        targetX, targetY = self.waypoints[self.currWaypointIdx]

        # Switch waypoint if reached
        dist = np.hypot(x - targetX, y - targetY)

        if dist < self.waypointThresh:
            self.currWaypointIdx = (self.currWaypointIdx + 1) % len(self.waypoints)
            targetX, targetY = self.waypoints[self.currWaypointIdx]

            # Reset PID integrals when changing targets
            self.yPID.integral = 0
            self.xPID.integral = 0

        # ------------------------ Outer Loop ------------------------
        errY = y - targetY  # Positive if drone is below the target (needs more thrust)
        errX = targetX - x

        # Apply deadzones to simulate human control imperfections
        errY = self._applyDeadzone(errY, self._positionDeadzone)
        errX = self._applyDeadzone(errX, self._positionDeadzone)

        # Altitude PID
        #   Total required thrust force
        totalThrust = Physics.GRAVITY + self.yPID.compute(errY, dt)

        # Lateral PID
        #   Desired tilt angle
        thetaTarget = self.xPID.compute(errX, dt)

        # ------------------------ Inner Loop ------------------------
        # Attitude PID
        #   Required torque
        errTheta = thetaTarget - theta
        torque = self.thetaPID.compute(errTheta, dt)

        # Torque = Arm * (Right - Left)
        # Thrust = Right + Left
        diffThrust = torque / Physics.ARM_LENGTH

        motorR = (totalThrust + diffThrust) / (2.0 * Physics.MAX_THRUST)
        motorL = (totalThrust - diffThrust) / (2.0 * Physics.MAX_THRUST)

        # Add random noise to simulate imperfect human control
        motorR += np.random.uniform(-self._motorNoise, self._motorNoise)
        motorL += np.random.uniform(-self._motorNoise, self._motorNoise)

        # Clamp to valid ranges
        motorR = np.clip(motorR, 0.0, 1.0)
        motorL = np.clip(motorL, 0.0, 1.0)

        # Store last status for HUD display
        self.currTargetX = targetX
        self.currTargetY = targetY
        self.currDist = dist
        self.currTotalThrust = totalThrust
        self.currMotorR = motorR
        self.currMotorL = motorL
        self.currThetaTarget = thetaTarget
        self.currTorque = torque

        # Print expert status
        status = self._getStatusLines(
            targetX, targetY, dist, totalThrust, motorR, motorL, thetaTarget, torque
        )
        print(f"\r\033[K{status[0]}")
        for line in status[1:]:
            print(line)

        print("\033[8A", end="")

        return Action(motorL, motorR)

    def _getStatusLines(
        self, targetX, targetY, dist, totalThrust, motorR, motorL, thetaTarget, torque
    ):
        """Return list of status lines for display."""
        return [
            "[EXPERT]",
            f"Target: ({targetX:5.1f}, {targetY:5.1f})",
            f"Dist: {dist:6.1f}",
            f"Thrust: {totalThrust:7.1f}",
            f"├── Right: {motorR:5.2f}",
            f"└── Left:  {motorL:5.2f}",
            f"Theta: {np.degrees(thetaTarget):6.1f}°",
            f"Torque: {torque:7.1f}",
        ]
