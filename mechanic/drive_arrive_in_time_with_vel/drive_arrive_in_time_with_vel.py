import math
import numpy as np

from rlbot.agents.base_agent import SimpleControllerState

from mechanic.base_mechanic import BaseMechanic
from mechanic.drive_turn_face_target import DriveTurnFaceTarget

from util.numerics import clip, sign
from util.physics.drive_1d_simulation_utils import (
    throttle_acceleration,
    BOOST_MIN_ACCELERATION,
    BOOST_MIN_TIME,
    BREAK_ACCELERATION,
    MAX_CAR_SPEED,
)
from util.render_utils import render_hitbox, render_car_text
from util.physics.drive_1d_velocity import state_at_velocity
from util.physics.drive_1d_distance import state_at_distance

PI = math.pi


class DriveArriveInTimeWithVel(BaseMechanic):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.turn_mechanic = DriveTurnFaceTarget(self.agent, rendering_enabled=False)

    def step(self, car, target_loc, time, final_vel) -> SimpleControllerState:

        turn_mechanic_controls = self.turn_mechanic.step(car, target_loc)

        if not self.turn_mechanic.finished:
            # continue turning until we're facing the correct way
            return turn_mechanic_controls

        self.controls.steer = turn_mechanic_controls.steer

        # useful variables
        delta_time = car.time - car.last_time
        if delta_time == 0:
            delta_time = 1 / 60

        target_in_local_coords = (target_loc - car.location).dot(car.rotation_matrix)
        car_forward_velocity = car.velocity.dot(car.rotation_matrix[:, 0])
        distance = np.linalg.norm(target_in_local_coords)

        car_ang_vel_local_coords = np.dot(car.angular_velocity, car.rotation_matrix)

        # arrive in time
        desired_vel = clip(distance / max(time - delta_time, 1e-5), -2300, 2300)

        # arrive with velocity
        current_final_vel = state_at_distance(distance, car_forward_velocity, car.boost)[1]
        if current_final_vel >= final_vel:
            time_final_vel, dist_final_vel, _ = state_at_velocity(final_vel, car_forward_velocity, car.boost)
            if time_final_vel > time - delta_time:
                desired_vel = final_vel
            else:
                desired_vel = clip((distance - dist_final_vel) / max(time - time_final_vel, 1e-5), -2300, 2300)
        else:
            time_final_vel, dist_final_vel, _ = state_at_velocity(final_vel, 0, car.boost)
            time_0_vel, dist_0_vel, _ = state_at_velocity(0, car_forward_velocity, 0)
            time_to_target_from_full_stop = state_at_distance(distance, 0, car.boost)[0]
            if car_forward_velocity > 0:
                time_dist_0, vel_dist_0, _ = state_at_distance(dist_0_vel, 0, 0)
                time_to_target_from_full_stop = state_at_distance(distance, 0, car.boost)[0]
                if time_dist_0 + time_to_target_from_full_stop < time - delta_time:
                    desired_vel = -1
            else:
                if time_0_vel + time_to_target_from_full_stop < min(
                    time - delta_time, time_0_vel + time_final_vel - delta_time
                ):
                    desired_vel = -MAX_CAR_SPEED

        # throttle to desired velocity
        self.controls.throttle = throttle_velocity(car_forward_velocity, desired_vel, delta_time)

        # boost to desired velocity
        self.controls.boost = not self.controls.handbrake and boost_velocity(
            car_forward_velocity, desired_vel, delta_time
        )

        # This makes sure we're not powersliding
        # if the car is spinning the opposite way we're steering towards
        if car_ang_vel_local_coords[2] * self.controls.steer < 0:
            self.controls.handbrake = False
        # and also not boosting if we're sliding the opposite way we're throttling towards.
        if car_forward_velocity * self.controls.throttle < 0:
            self.controls.handbrake = self.controls.boost = False

        # rendering
        if self.rendering_enabled:
            text_list = [
                f"current_vel : {car_forward_velocity:.2f}",
                f"desired_vel : {desired_vel:.2f}",
                f"final_vel : {final_vel:.2f}",
                f"current_final_vel : {current_final_vel:.2f}",
                f"time : {time:.2f}",
                f"throttle : {self.controls.throttle:.2f}",
            ]

            color = self.agent.renderer.white()

            self.agent.renderer.begin_rendering()
            # rendering all debug text in 3d near the car
            render_car_text(self.agent.renderer, car, text_list, color)
            # rendering a line from the car to the target
            self.agent.renderer.draw_rect_3d(target_loc, 20, 20, True, self.agent.renderer.red())
            self.agent.renderer.draw_line_3d(car.location, target_loc, color)
            # hitbox rendering
            render_hitbox(
                self.agent.renderer, car.location, car.rotation_matrix, color, car.hitbox_corner, car.hitbox_offset,
            )
            self.agent.renderer.draw_rect_3d(car.location, 20, 20, True, self.agent.renderer.grey())
            self.agent.renderer.end_rendering()

        # updating status
        if distance < 20 and abs(time) < 0.05 and abs(final_vel - car_forward_velocity) < 50:
            self.finished = True
        if time < -0.05:
            self.failed = True
            print("distance", distance, "time", time, "velocity", car_forward_velocity)

        return self.controls


def throttle_velocity(vel, desired_vel, dt):
    """Model based throttle to velocity"""
    desired_accel = (desired_vel - vel) / dt * sign(desired_vel)
    if desired_accel > 0:
        return clip(desired_accel / max(throttle_acceleration(vel, 1), 0.001)) * sign(desired_vel)
    elif -BREAK_ACCELERATION < desired_accel <= 0:
        return 0
    else:
        return -1


def boost_velocity(vel, desired_vel, dt):
    """Model based velocity boost control"""
    if desired_vel < vel or vel < 0 or vel > MAX_CAR_SPEED - 1:
        # don't boost if we want to go or we're going backwards or we're already at max speed
        return False
    else:
        desired_accel = (desired_vel - vel) / dt
        return desired_accel > (BOOST_MIN_ACCELERATION + throttle_acceleration(vel, 1) * BOOST_MIN_TIME / dt) / 2
