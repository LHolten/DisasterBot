import math
import numpy as np
from rlbot.agents.base_agent import SimpleControllerState
from mechanic.base_mechanic import BaseMechanic
from util.numerics import clip, sign
from util.drive_physics import throttle_acc

PI = math.pi
DT = 1 / 120


class DriveTurnToFaceTarget(BaseMechanic):

    def step(self, car, target_loc) -> SimpleControllerState:

        target_in_local_coords = np.dot(target_loc - car.location, car.rotation_matrix)

        # PD for steer
        yaw_angle_to_target = math.atan2(target_in_local_coords[1], target_in_local_coords[0])
        car_ang_vel_local_coords = np.dot(car.angular_velocity, car.rotation_matrix)
        car_yaw_ang_vel = -car_ang_vel_local_coords[2]

        proportional_steer = 10 * yaw_angle_to_target
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

        self.controls.steer = clip(proportional_steer + derivative_steer)
        self.controls.throttle = 1
        self.controls.boost = not self.controls.handbrake

        # updating status
        error = abs(car_yaw_ang_vel) + abs(yaw_angle_to_target)

        if error < 0.01:
            self.finished = True
        else:
            self.finished = False

        return self.controls


class DriveArriveInTime(BaseMechanic):

    def step(self, car, target_loc, time) -> SimpleControllerState:

        target_in_local_coords = (target_loc - car.location).dot(car.rotation_matrix)
        car_local_velocity = car.velocity.dot(car.rotation_matrix)
        distance = np.linalg.norm(target_in_local_coords)

        # PD for steer
        yaw_angle_to_target = math.atan2(target_in_local_coords[1], target_in_local_coords[0])
        car_ang_vel_local_coords = np.dot(car.angular_velocity, car.rotation_matrix)
        car_yaw_ang_vel = -car_ang_vel_local_coords[2]

        proportional_steer = 10 * yaw_angle_to_target
        derivative_steer = 1 / 4 * car_yaw_ang_vel

        if yaw_angle_to_target >= 0:
            if yaw_angle_to_target + car_yaw_ang_vel / 4 > PI / 4:
                self.controls.handbrake = True
            else:
                self.controls.handbrake = False
        elif yaw_angle_to_target < 0:
            if yaw_angle_to_target + car_yaw_ang_vel / 4 < -PI / 4:
                self.controls.handbrake = True
            else:
                self.controls.handbrake = False

        self.controls.steer = clip(proportional_steer + derivative_steer)

        # arrive in time
        desired_velocity = distance / max(time - 1 / 60, 0.001)

        # throttle to desired velocity
        current_velocity = np.linalg.norm(car.velocity)

        self.controls.throttle = throttle_velocity(
            current_velocity, desired_velocity)

        # boost to  desired velocity
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
        strings = [f"desired_velocity : {desired_velocity}",
                   f"time : {time}",
                   f"throttle : {self.controls.throttle}"]

        self.agent.renderer.begin_rendering()
        for i, string in enumerate(strings):
            self.agent.renderer.draw_string_2d(20, 150 + i * 30, 2, 2, string,
                                               self.agent.renderer.white())
        self.agent.renderer.draw_rect_3d(target_loc, 20, 20, True, self.agent.renderer.red())
        self.agent.renderer.end_rendering()

        # updating status
        if distance < 200:
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
        threshold = 50
    return rel_vel > threshold
