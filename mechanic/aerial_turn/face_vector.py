from mechanic.base_mechanic import BaseMechanic
from rlbot.agents.base_agent import SimpleControllerState
from rlutilities.simulation import Car
from rlutilities.linear_algebra import vec3, dot, look_at
from rlutilities.mechanics import AerialTurn
from util.coordinates import spherical
import math


class BaseFaceVector(BaseMechanic):
    car = None
    target = None
    frames_done = 0

    def step(self, car: Car, target: vec3, dt) -> SimpleControllerState:
        self.car = car
        self.target = target

        local_omega = dot(self.car.angular_velocity, self.car.rotation)

        omega_error = math.sqrt(local_omega[1]**2 + local_omega[2]**2)

        if omega_error < 0.1:
            self.frames_done += 1
        else:
            self.frames_done = 0

        if self.frames_done >= 10:
            self.finished = True

        return self.rotate(car, target, dt)

    def rotate(self, car: Car, target: vec3, dt) -> SimpleControllerState:
        raise NotImplementedError


class FaceVectorRLU(BaseFaceVector):
    def rotate(self, car: Car, target: vec3, dt) -> SimpleControllerState:
        # up = vec3(*[self.car.theta[i, 2] for i in range(3)])

        target_rotation = look_at(self.target, vec3(0, 0, 1))
        action = AerialTurn(car)
        action.target = target_rotation

        action.step(dt)
        controls = action.controls

        return controls
