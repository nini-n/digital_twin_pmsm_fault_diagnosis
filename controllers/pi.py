"""
Simple PI controller used in the closed-loop PMSM simulation.

The controller includes optional output saturation and is used for the speed
and current loops in the digital-twin dataset generation script.
"""


class PIController:
    """Discrete-time proportional-integral controller with optional saturation."""

    def __init__(self, Kp: float, Ki: float, limit: float | None = None):
        self.Kp = float(Kp)
        self.Ki = float(Ki)
        self.limit = float(limit) if limit is not None else None
        self.integrator = 0.0

    def update(self, error: float, dt: float) -> float:
        """
        Update the controller output for one simulation step.

        Parameters
        ----------
        error : float
            Tracking error.
        dt : float
            Simulation time step.

        Returns
        -------
        float
            Controller output after optional saturation.
        """
        error = float(error)
        dt = float(dt)

        self.integrator += error * dt
        output = self.Kp * error + self.Ki * self.integrator

        if self.limit is not None:
            output = max(-self.limit, min(self.limit, output))

        return output