import torch
from torch import Tensor

from mechanic.aerial_turn_ml.policy import Policy
# ??
j = 10.5

# air control torque coefficients
t = torch.tensor([-400.0, -130.0, 95.0], dtype=torch.float)


class Simulation:
    o: Tensor = None
    w: Tensor = None

    def __init__(self, policy: Policy):
        self.policy = policy

    def simulate(self, steps: int, dt: float):
        for _ in range(steps):
            self.step(dt)

    def step(self, dt):
        rpy = self.policy(self.o, self.w)

        # air damping torque coefficients
        h = torch.stack((
            torch.tensor(-50.0)[None],
            -30.0 * (1.0 - rpy[:, 1].abs()),
            -20.0 * (1.0 - rpy[:, 2].abs())
        ), dim=1)

        w_local = torch.sum(self.o * self.w[:, :, None], 1)
        self.w += torch.sum(self.o * (t[None, :] * rpy + h * w_local)[:, None, :], 2) * (dt / j)
        self.o = torch.sum(self.o[:, None, :, :] * axis_to_rotation(self.w * dt)[:, :, :, None], 2)


def axis_to_rotation(omega: Tensor):
    norm_omega = torch.norm(omega, dim=1)

    u = omega / norm_omega[:, None]

    c = torch.cos(norm_omega)
    s = torch.sin(norm_omega)

    result = u[:, :, None] * u[:, None, :] * (-c[:, None, None] + 1.0)
    result += c[:, None, None] * torch.diag(torch.ones(3))[None, :, :]
    result[:, 0, 1] -= u[:, 2] * s
    result[:, 0, 2] += u[:, 1] * s
    result[:, 1, 0] += u[:, 2] * s
    result[:, 1, 2] -= u[:, 0] * s
    result[:, 2, 0] -= u[:, 1] * s
    result[:, 2, 1] += u[:, 0] * s

    return result
