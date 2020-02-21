import numpy as np

from rlbot.agents.base_agent import SimpleControllerState

from action.base_action import BaseAction
from mechanic.drive_arrive_in_time import DriveArriveInTime

from util.collision_utils import box_ball_collision_distance, box_ball_location_on_collision
from util.physics.drive_1d_time import state_at_time_vectorized


class HitGroundBall(BaseAction):

    mechanic = None

    def get_controls(self, game_data) -> SimpleControllerState:

        if self.mechanic is None:
            self.mechanic = DriveArriveInTime(self.agent, rendering_enabled=self.rendering_enabled)

        self.finished = self.mechanic.finished

        target_loc, target_dt = self.get_target_ball_state(game_data)
        return self.mechanic.step(game_data.my_car, target_loc, target_dt)

    @staticmethod
    def get_target_ball_state(game_data):

        ball_prediction = game_data.ball_prediction
        car = game_data.my_car
        ball = game_data.ball

        target_loc = game_data.ball_prediction[-1]["physics"]["location"]
        target_dt = game_data.ball_prediction[-1]["game_seconds"]

        hitbox_height = car.hitbox_corner[2] + car.hitbox_offset[2]
        origin_height = 16  # the car's elevation from the ground due to wheels and suspension

        # only accurate if we're already moving towards the target
        velocity = np.array([np.linalg.norm(car.velocity)] * len(ball_prediction))
        boost = np.array([car.boost] * len(ball_prediction))

        location_slices = ball_prediction["physics"]["location"]

        distance_slices = box_ball_collision_distance(
            location_slices, car.location, car.rotation_matrix, car.hitbox_corner, car.hitbox_offset, ball.radius,
        )
        time_slices = ball_prediction["game_seconds"] - game_data.time

        not_too_high = location_slices[:, 2] < ball.radius + hitbox_height + origin_height

        reachable = (state_at_time_vectorized(time_slices, velocity, boost)[0] > distance_slices) & (not_too_high)

        filtered_prediction = ball_prediction[reachable]

        if len(filtered_prediction) > 0:
            target_loc = filtered_prediction[0]["physics"]["location"]
            target_dt = filtered_prediction[0]["game_seconds"] - game_data.time
            target_loc = box_ball_location_on_collision(
                target_loc, car.location, car.rotation_matrix, car.hitbox_corner, car.hitbox_offset, ball.radius,
            )

        return target_loc, target_dt

    def is_valid(self, game_data):
        return True
