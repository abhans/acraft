import numpy as np


class PIDController:
    def __init__(self, kP, kD, kI=0.0, limit=None):
        self.kP = kP
        self.kD = kD
        self.kI = kI
        self.limit = limit
        self.integral = 0.0
        self.errPrev = 0.0

    def compute(self, error, dt):
        # Accumulated integral for kI
        self.integral += error * dt
        # Derivative in error for kD
        derivative = (error - self.errPrev) / dt if dt > 0 else 0.0

        self.errPrev = error

        # PID: kP* error + kI * integral + kD * derivative
        output = (self.kP * error) + (self.kI * self.integral) + (self.kD * derivative)

        if self.limit:
            output = np.clip(output, -self.limit, self.limit)

        return output
