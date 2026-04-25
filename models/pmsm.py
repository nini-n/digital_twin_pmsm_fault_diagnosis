"""
PMSM model used by the digital-twin simulation.

The model is written in the d-q reference frame and is used both as the faulty
plant and as the nominal healthy twin. Fault injection is handled outside this
class by modifying the plant-side parameters during simulation.
"""

from __future__ import annotations

from typing import Dict, Tuple


class PMSM:
    """Permanent magnet synchronous motor model in the d-q reference frame."""

    def __init__(self, params: Dict[str, float]):
        self.Rs = float(params["Rs"])
        self.Ld = float(params["Ld"])
        self.Lq = float(params["Lq"])
        self.lambda_m = float(params["lambda_m"])
        self.p = float(params["p"])
        self.J = float(params["J"])
        self.B = float(params["B"])

        # Stored nominal values are used when the plant parameters are reset.
        self.Rs_nominal = self.Rs
        self.Ld_nominal = self.Ld
        self.Lq_nominal = self.Lq

        # State variables: d-axis current, q-axis current, and mechanical speed.
        self.id = 0.0
        self.iq = 0.0
        self.omega = 0.0

    def step(self, vd: float, vq: float, Tl: float, dt: float) -> Tuple[float, float, float, float]:
        """
        Advance the motor state by one Euler integration step.

        Parameters
        ----------
        vd, vq : float
            d-axis and q-axis stator voltages.
        Tl : float
            Load torque.
        dt : float
            Simulation time step.

        Returns
        -------
        tuple
            Updated d-axis current, q-axis current, mechanical speed, and
            electromagnetic torque.
        """
        vd = float(vd)
        vq = float(vq)
        Tl = float(Tl)
        dt = float(dt)

        omega_e = self.p * self.omega

        did_dt = (
            vd
            - self.Rs * self.id
            + omega_e * self.Lq * self.iq
        ) / self.Ld

        diq_dt = (
            vq
            - self.Rs * self.iq
            - omega_e * self.Ld * self.id
            - omega_e * self.lambda_m
        ) / self.Lq

        torque = 1.5 * self.p * (
            self.lambda_m * self.iq
            + (self.Ld - self.Lq) * self.id * self.iq
        )

        domega_dt = (torque - Tl - self.B * self.omega) / self.J

        self.id += did_dt * dt
        self.iq += diq_dt * dt
        self.omega += domega_dt * dt

        return self.id, self.iq, self.omega, torque