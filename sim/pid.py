"""Generic PID controller."""
import numpy as np


class PID:
    """
    Discrete PID with optional integral anti-windup.

    integral_limit clamps the raw accumulator to ±integral_limit,
    bounding the Ki contribution to ±Ki*integral_limit.
    """

    def __init__(self, Kp, Ki, Kd, dt, integral_limit=None):
        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd
        self.dt = dt
        self.integral = 0.0
        self.prev_error = 0.0
        self.integral_limit = integral_limit

    def update(self, error):
        self.integral += error * self.dt
        if self.integral_limit is not None:
            self.integral = np.clip(self.integral,
                                    -self.integral_limit, self.integral_limit)
        derivative = (error - self.prev_error) / self.dt if self.dt > 0 else 0.0
        output = self.Kp * error + self.Ki * self.integral + self.Kd * derivative
        self.prev_error = error
        return output

    def reset(self):
        self.integral = 0.0
        self.prev_error = 0.0

    def __repr__(self):
        return (f'PID(Kp={self.Kp}, Ki={self.Ki}, Kd={self.Kd}, '
                f'lim=±{self.integral_limit})')
