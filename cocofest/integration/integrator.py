from bioptim import OdeSolver
from bioptim.dynamics.integrator import RK4
from casadi import MX, SX, vertcat
import math


class ModifiedOdeSolverRK4(OdeSolver.RK4):
    def __init__(self, n_integration_steps: int = 20):
        super().__init__(n_integration_steps)

    @property
    def integrator(self):
        return ModifiedRK4


class ModifiedRK4(RK4):
    """
    Modified Runge-Kutta method with a customized k4 computation.
    """

    def __init__(self, ode: dict, ode_opt: dict):
        """
        Parameters
        ----------
        ode: dict
            The ode description
        ode_opt: dict
            The ode options
        """
        super(RK4, self).__init__(ode, ode_opt)

    def next_x(self, t0: float | MX | SX, x_prev: MX | SX, u: MX | SX, p: MX | SX, a: MX | SX, d: MX | SX) -> MX | SX:
        h = self.h
        dt = self.dt
        offset = math.exp(-15)

        k1 = self.fun(vertcat(t0, dt), x_prev, self.get_u(u, t0), p, a, d)[:, self.ode_idx]
        k2 = self.fun(vertcat(t0 + h / 2, dt), x_prev + h / 2 * k1, self.get_u(u, t0 + h / 2), p, a, d)[:, self.ode_idx]
        k3 = self.fun(vertcat(t0 + h / 2, dt), x_prev + h / 2 * k2, self.get_u(u, t0 + h / 2), p, a, d)[:, self.ode_idx]

        # Customize the k4 computation
        k4 = self.fun(vertcat(t0 + h-offset, dt), x_prev + h-offset * (k2 + k3) / 2, self.get_u(u, t0 + h-offset), p, a, d)[:, self.ode_idx]

        return x_prev + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)

