from mechanic.base_mechanic import BaseMechanic
from rlbot.agents.base_agent import SimpleControllerState
from rlutilities.simulation import Car
from rlutilities.linear_algebra import vec3, dot, look_at, transpose
from rlutilities.mechanics import AerialTurn
from rlutilities.mechanics import Jump
from rlutilities.simulation import Game, Input
from util.coordinates import spherical
import math

if 'torch' not in globals():
    import torch


class Flick(BaseMechanic):

    def __init__(self, info: Game):
        super().__init__()
        self.car = info.my_car
        self.ball = info.ball
        self.time = info.time

        # 60 frames, rpy + jump + release
        self.plan = torch.zeros(60, 5, requires_grad=True)
        self.optimizer = torch.optim.Adam([self.plan])

        torch.autograd.set_detect_anomaly(True)

    def step(self, target: vec3, dt) -> SimpleControllerState:
        if self.car.on_ground:
            self.controls = SimpleControllerState(jump=True)

        new_car = self.simulate()
        new_car.location[0].backward()

        return self.controls

    def simulate(self):
        car = TensorCar(self.car)
        # ball_offset = dot(transpose(self.car.rotation), self.ball.location - self.car.location)
        # ball_offset = torch.tensor(ball_offset)

        for controls in self.plan.split(1)[0]:
            controls = TensorControls(controls)
            car.step(controls)

        return car


class TensorControls:
    rpy: torch.Tensor
    jump: torch.Tensor
    use_item: torch.Tensor

    def __init__(self, controls):
        if isinstance(controls, torch.Tensor):
            self.rpy = controls[:3]
            self.jump, self.use_item = controls[3:][:, None]

        if isinstance(controls, Input):
            self.rpy = torch.tensor([controls.roll, controls.pitch, controls.yaw])
            self.jump = torch.tensor([controls.jump]).float()
            self.use_item = torch.zeros(1)


class TensorCar:
    def __init__(self, car: Car):
        self.location = torch.tensor([car.location[i] for i in range(3)])
        self.velocity = torch.tensor([car.velocity[i] for i in range(3)])
        self.rotation = torch.tensor([[car.rotation[i, j] for j in range(3)] for i in range(3)])
        self.angular_velocity = torch.tensor([car.angular_velocity[i] for i in range(3)])
        self.double_jumped = torch.tensor([car.double_jumped]).float()
        self.controls = TensorControls(car.controls)

        self.time = torch.tensor(car.time)
        self.jump_timer = torch.tensor(car.jump_timer)
        self.dodge_timer = torch.tensor(car.dodge_timer)
        if self.dodge_timer.item() < 0:
            self.dodge_timer += 10

        self.dodge_torque = torch.zeros(3)

        self.enable_jump_acceleration = torch.ones(1)

    def step(self, controls: TensorControls):
        dt = torch.tensor(0.01666)

        if self.jump_timer.item() < 1.5:
            dodge_multiplier = andt(
                controls.jump,
                nott(self.controls.jump),
                nott(self.double_jumped),
            )
            self.air_dodge(controls, dt, dodge_multiplier)
            self.aerial_control(controls, dt * nott(dodge_multiplier))
        else:
            self.aerial_control(controls, dt)

        self.velocity[2] += -650 * dt

        self.location += self.velocity * dt
        self.rotation = torch.matmul(axis_to_rotation(self.angular_velocity * dt), self.rotation)

        self.velocity = self.velocity / torch.clamp_min(torch.norm(self.velocity) / 2300, 1)
        self.angular_velocity = self.angular_velocity / torch.clamp_min(torch.norm(self.angular_velocity) / 5.5, 1)

        self.time += dt
        self.dodge_timer += dt
        self.jump_timer += dt

        if self.jump_timer.item() < 0.2:
            self.enable_jump_acceleration = andt(
                controls.jump,
                self.enable_jump_acceleration,
            )
        else:
            self.enable_jump_acceleration = torch.zeros(1)

        self.controls = controls

    def aerial_control(self, controls: TensorControls, dt):
        J = torch.tensor(10.5)
        T = torch.tensor([-400.0, -130.0, 95.0])
        H = torch.tensor([-50.0, -30.0 * nott(controls.rpy[1].abs()), -20.0 * nott(controls.rpy[2].abs())])

        accelerate = andt(
            controls.jump,
            self.enable_jump_acceleration,
        )
        if self.jump_timer.item() < 0.025:
            self.velocity += (0.75 * 1458.3333 * self.rotation[:, 2] - 510.0 * self.rotation[:, 0]) * dt * accelerate
        else:
            self.velocity += 1458.3333 * self.rotation[:, 2] * dt * accelerate

        if self.dodge_timer.item() >= 0.15 and (self.velocity[2].item() < 0.0 or self.dodge_timer.item() < 0.21):
            self.velocity[2] -= self.velocity[2] * 0.35

        if self.dodge_timer.item() <= 0.3:
            controls.rpy[1] = 0

        if self.dodge_timer.item() <= 0.65:
            self.angular_velocity += self.dodge_torque * dt
        else:
            angular_local = torch.matmul(self.angular_velocity, self.rotation)
            self.angular_velocity += torch.matmul(self.rotation, T * controls.rpy + H * angular_local) * (dt / J)

    def air_dodge(self, controls, dt, multiplier):
        dt = dt * multiplier

        vf = torch.dot(self.velocity, self.rotation[:, 0])
        s = vf.abs() / 2300

        dodge_dir = controls.rpy[1:] / torch.clamp_min(torch.norm(controls.rpy[1:]), 0.001)

        dodge_torque_local = torch.cat([-dodge_dir[1:2] * 260, dodge_dir[0:1] * 224, torch.zeros(1)])
        dodge_torque = torch.matmul(self.rotation, dodge_torque_local)

        if vf.abs() < 100.0:
            backward_dodge = dodge_dir[0] < 0.0
        else:
            backward_dodge = (dodge_dir[0] >= 0.0) != (vf > 0.0)

        dv = 500.0 * dodge_dir

        if backward_dodge:
            dv[0] *= (16.0 / 15.0) * (1.0 + 1.5 * s)

        dv[1] *= (1.0 + 0.9 * s)

        theta = torch.atan2(self.rotation[1, 0], self.rotation[0, 0])
        o_dodge = torch.tensor([
            [torch.cos(theta), -torch.sin(theta)],
            [torch.sin(theta), torch.cos(theta)],
        ])
        self.velocity[:2] += torch.matmul(o_dodge, dv)

        self.angular_velocity += dodge_torque * dt

        self.double_jumped += nott(self.double_jumped) * multiplier
        self.dodge_timer -= self.dodge_timer * multiplier


def nott(t: torch.Tensor):
    return t.neg() + 1


def andt(*t: torch.Tensor):
    t = torch.cat(t)
    mask = t == 0
    if mask.any():
        return t[mask].sum()
    else:
        return torch.ones()


def ort(*t: torch.Tensor):
    t = torch.cat(t)
    mask = t == 1
    if mask.any():
        return torch.ones()
    else:
        return t.sum()


def axis_to_rotation(omega):
    norm_omega = torch.norm(omega)

    u = omega / norm_omega

    c = torch.cos(norm_omega)
    s = torch.sin(norm_omega)

    result = u[:, None] * u[None, :] * (-c[None, None] + 1.0)
    result += c[None, None] * torch.diag(torch.ones(3))[:, :]

    result += torch.cross(s[None, None] * torch.diag(torch.ones(3))[:, :],
                          u[None, :].repeat(3, 1), dim=1)

    return result
