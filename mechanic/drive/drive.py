import math
import numpy as np
from rlbot.agents.base_agent import SimpleControllerState
from mechanic.base_mechanic import BaseMechanic
from util.numerics import clip

PI = math.pi
DT = 1 / 120


class DriveTurnToFaceTarget(BaseMechanic):

    def step(self, car, target_loc) -> SimpleControllerState:

        target_in_local_coords = np.dot(target_loc - car.location, car.rotation_matrix)

        # PD for steer
        yaw_angle_to_target = math.atan2(target_in_local_coords[1], target_in_local_coords[0])
        car_ang_vel_local_coords = np.dot(car.angular_velocity, car.rotation_matrix)
        car_yaw_ang_vel = -car_ang_vel_local_coords[2]

        porportional_steer = 10 * yaw_angle_to_target
        derivative_steer = 1 / 4 * car_yaw_ang_vel

        if yaw_angle_to_target >= 0:
            if yaw_angle_to_target + car_yaw_ang_vel / 4 > PI / 3:
                self.controls.handbrake = True
            else:
                self.controls.handbrake = False
        elif yaw_angle_to_target < 0:
            if yaw_angle_to_target + car_yaw_ang_vel / 4 < -PI / 3:
                self.controls.handbrake = True
            else:
                self.controls.handbrake = False

        self.controls.steer = clip(porportional_steer + derivative_steer)
        self.controls.throttle = 1
        self.controls.boost = not self.controls.handbrake

        # updating status
        error = abs(car_yaw_ang_vel) + abs(yaw_angle_to_target)

        if error < 0.01:
            self.finished = True
        else:
            self.finished = False

        return self.controls
