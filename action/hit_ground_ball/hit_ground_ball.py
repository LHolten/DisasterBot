import numpy as np

from rlbot.agents.base_agent import SimpleControllerState

from skeleton.util.conversion import rotation_to_matrix

from action.base_action import BaseAction
from mechanic.drive_arrive_in_time import DriveArriveInTime

from util.collision_utils import box_ball_collision_distance, box_ball_low_location_on_collision
from util.physics.drive_1d_time import state_at_time_vectorized


class HitGroundBall(BaseAction):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.mechanic = DriveArriveInTime(self.agent, rendering_enabled=self.rendering_enabled)
        self.target_loc = None
        self.target_time = None

    def get_controls(self, game_data) -> SimpleControllerState:

        if self.target_loc is None or True:
            # remove "or True" to test the accuracy without recalculating each tick.
            self.target_loc, target_dt = self.get_target_ball_state(game_data)
            self.target_time = game_data.time + target_dt

        target_dt = self.target_time - game_data.time
        self.controls = self.mechanic.step(game_data.my_car, self.target_loc, target_dt)

        self.finished = self.mechanic.finished
        self.failed = self.mechanic.failed

        return self.controls

    @staticmethod
    def get_target_ball_state(game_data):

        ball_prediction = game_data.ball_prediction
        car = game_data.my_car
        car_rot = rotation_to_matrix([0, car.rotation[1], car.rotation[2]])

        ball = game_data.ball

        hitbox_height = car.hitbox_corner[2] + car.hitbox_offset[2]
        origin_height = 17  # the car's elevation from the ground due to wheels and suspension

        # only accurate if we're already moving towards the target
        boost = np.array([car.boost] * len(ball_prediction), dtype=np.float64)

        location_slices = ball_prediction["physics"]["location"]

        distance_slices = box_ball_collision_distance(
            location_slices, car.location, car_rot, car.hitbox_corner, car.hitbox_offset, ball.radius,
        )
        time_slices = np.array(ball_prediction["game_seconds"] - game_data.time, dtype=np.float64)

        not_too_high = location_slices[:, 2] < ball.radius + hitbox_height + origin_height

        velocity = car.velocity[None, :]
        direction_slices = location_slices - car.location
        velocity = np.sum(velocity * direction_slices, 1) / np.linalg.norm(direction_slices, 2, 1)
        velocity = np.array(velocity, dtype=np.float64)

        reachable = (state_at_time_vectorized(time_slices, velocity, boost)[0] > distance_slices) & not_too_high

        filtered_prediction = ball_prediction[reachable]

        target_loc = game_data.ball_prediction[-1]["physics"]["location"].copy()
        target_loc[2] = origin_height
        target_dt = game_data.ball_prediction[-1]["game_seconds"] - game_data.time

        if len(filtered_prediction) > 0:
            target_loc = filtered_prediction[0]["physics"]["location"]
            target_dt = filtered_prediction[0]["game_seconds"] - game_data.time
            target_loc = box_ball_low_location_on_collision(
                target_loc, car.location, car_rot, car.hitbox_corner, car.hitbox_offset, ball.radius,
            )

        return target_loc, target_dt

    def is_valid(self, game_data):
        return True
