import torch
from torch import Tensor
from torch.distributions.normal import Normal

from mechanic.aerial_turn_ml.policy import Policy

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# ??
j = 10.5

# air control torque coefficients
t = torch.tensor([-400.0, -130.0, 95.0], dtype=torch.float, device=device)
m = torch.diag(torch.ones(3)).byte().to(device)

w_max = 5.5
batch_size = 8000


class Simulation:
    o: Tensor = None
    w: Tensor = None

    def __init__(self, policy: Policy):
        self.policy = policy

    def random_state(self):
        self.o = Normal(0, 1).sample((batch_size, 3, 3)).to(device)
        self.w = torch.zeros((batch_size, 3), device=device)

        self.normalize()

    def simulate(self, steps: int, dt: float):
        for _ in range(steps):
            self.step(dt)

    def normalize(self):
        self.o = self.o / self.o.norm(dim=2, keepdim=True)
        self.o[:, 2] = torch.cross(self.o[:, 0], self.o[:, 1], dim=1)
        self.o[:, 1] = torch.cross(self.o[:, 2], self.o[:, 0], dim=1)

    def step(self, dt):
        rpy = self.policy(self.o, self.w)

        # air damping torque coefficients
        h = torch.stack((
            torch.full_like(rpy[:, 0], -50.0),
            -30.0 * (1.0 - rpy[:, 1].abs()),
            -20.0 * (1.0 - rpy[:, 2].abs())
        ), dim=1)

        w_local = torch.sum(self.o * self.w[:, :, None], 1)
        self.w = self.w + torch.sum(self.o * (t[None, :] * rpy + h * w_local)[:, None, :], 2) * (dt / j)
        self.o = torch.sum(self.o[:, None, :, :] * axis_to_rotation(self.w * dt)[:, :, :, None], 2)

        self.w = self.w / torch.clamp_min(torch.norm(self.w, dim=1) / w_max, 1)[:, None]

        # self.normalize()


def axis_to_rotation(omega: Tensor):
    norm_omega = torch.norm(omega, dim=1)

    u = omega / norm_omega[:, None]

    c = torch.cos(norm_omega)
    s = torch.sin(norm_omega)

    result = u[:, :, None] * u[:, None, :] * (-c[:, None, None] + 1.0)
    result += c[:, None, None] * torch.diag(torch.ones(3, device=device))[None, :, :]
    result[:, torch.cat((m[-1:], m[:-1]))] += torch.cat((u[:, 1:], u[:, :1]), 1) * s[:, None]
    result[:, torch.cat((m[1:], m[:1]))] -= torch.cat((u[:, -1:], u[:, :-1]), 1) * s[:, None]

    return result
