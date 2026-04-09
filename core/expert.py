import numpy as np

from core.config import Action, Physics
from core.controller import PIDController


class Expert:
    def __init__(self):
        # ------------------------ Outer Loop (Tilt & Thrust) ------------------------
        self.yPID = PIDController(kP=3.4, kD=2.7, limit=500.0)
        self.xPID = PIDController(kP=0.017, kD=0.038, limit=0.6)

        # ------------------------ Inner Loop (Desired Tilt to Motor Differential) ------------------------
        self.thetaPID = PIDController(kP=200.0, kD=87.0, limit=1000.0)

        # Waypoints for the expert to follow.
        self.waypoints = [
            (600, 400),
            (800, 300),
            (900, 500),
            (600, 600),
            (300, 400),
            (400, 200),
        ]
        self.currWaypointIdx = 0
        self.waypointThresh = 25.0  # In Pixels

        # ------------------------ Human-Expert Flaws ------------------------
        # Deadzones to simulate imperfect human control.
        #   Small errors within the deadzone are ignored by expert.
        self._positionDeadzone = 2.0
        self._angleDeadzone = np.radians(2.0)
        # Random noise added to motor outputs to simulate imperfect human control
        self._motorNoise = 0.025

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

        print(
            f"[EXPERT] Target: ({targetX:.1f}, {targetY:.1f})",
            f"Dist: {dist:.1f}",
            f"Thrust: {totalThrust:.1f}",
            f"ThetaTarget: {np.degrees(thetaTarget):.1f} deg",
            f"Torque: {torque:.1f}",
            f"MotorL: {motorL:.2f}",
            f"MotorR: {motorR:.2f}",
            sep=" | ",
            end="\r",
        )

        return Action(motorL, motorR)
