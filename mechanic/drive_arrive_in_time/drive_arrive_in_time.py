import math
import numpy as np

from rlbot.agents.base_agent import SimpleControllerState

from mechanic.base_mechanic import BaseMechanic

from util.numerics import clip, sign
from util.drive_physics import throttle_acc
from util.render_utils import render_hitbox

PI = math.pi
DT = 1 / 120


class DriveArriveInTime(BaseMechanic):

    def step(self, car, target_loc, time) -> SimpleControllerState:

        target_in_local_coords = (target_loc - car.location).dot(car.rotation_matrix)
        car_local_velocity = car.velocity.dot(car.rotation_matrix)
        distance = np.linalg.norm(target_in_local_coords)

        # PD for steer
        yaw_angle_to_target = math.atan2(target_in_local_coords[1], target_in_local_coords[0])
        car_ang_vel_local_coords = np.dot(car.angular_velocity, car.rotation_matrix)
        car_yaw_ang_vel = -car_ang_vel_local_coords[2]

        proportional_steer = 11 * yaw_angle_to_target
        derivative_steer = 1 / 3 * car_yaw_ang_vel

        if sign(yaw_angle_to_target) * (yaw_angle_to_target + car_yaw_ang_vel / 3) > PI / 5:
            self.controls.handbrake = True
        else:
            self.controls.handbrake = False

        self.controls.steer = clip(proportional_steer + derivative_steer)

        # arrive in time
        desired_velocity = clip(distance / max(time - 1 / 120, 0.001), -2300, 2300)

        # throttle to desired velocity
        current_velocity = car_local_velocity[0]

        self.controls.throttle = throttle_velocity(current_velocity, desired_velocity)

        # boost to desired velocity
        self.controls.boost = not self.controls.handbrake and boost_velocity(
            current_velocity, desired_velocity)

        # This makes sure we're not powersliding
        # if the car is spinning the opposite way we're steering towards
        if car_ang_vel_local_coords[2] * self.controls.steer < 0:
            self.controls.handbrake = False
        # and also not boosting if we're sliding the opposite way we're throttling towards.
        if car_local_velocity[0] * self.controls.throttle < 0:
            self.controls.handbrake = self.controls.boost = False

        # rendering
        strings = [f"desired_velocity : {desired_velocity:.2f}",
                   f"time : {time:.2f}",
                   f"throttle : {self.controls.throttle:.2f}"]
        color = self.agent.renderer.white()

        self.agent.renderer.begin_rendering()
        for i, string in enumerate(strings):
            self.agent.renderer.draw_string_2d(20, 150 + i * 30, 2, 2, string, color)
        self.agent.renderer.draw_rect_3d(target_loc, 20, 20, True, self.agent.renderer.red())
        self.agent.renderer.draw_line_3d(car.location, target_loc, color)
        render_hitbox(self.agent.renderer, car.location, car.rotation_matrix,
                      color, car.hitbox, car.hitbox_offset)
        self.agent.renderer.end_rendering()

        # updating status
        if distance < 200 and abs(time) < 0.1:
            self.finished = True
        else:
            self.finished = False

        return self.controls


def throttle_velocity(vel, dspeed, lthrottle=0):
    """PD throttle to velocity"""
    dacc = (dspeed - vel) / DT * sign(dspeed)
    if dacc > 0:
        return clip(dacc / max(throttle_acc(vel, 1), 0.001)) * sign(dspeed)
    elif -3600 < dacc <= 0:
        return 0
    else:
        return -1


def boost_velocity(vel, dvel, lboost=0):
    """P velocity boost control"""
    rel_vel = dvel - vel
    if vel < 1400:
        if dvel < 0:
            threshold = 4600
        else:
            threshold = 250
    else:
        threshold = 30
    return rel_vel > threshold
