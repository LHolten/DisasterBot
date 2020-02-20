import numpy as np

from mechanic.base_test_agent import BaseTestAgent
from mechanic.drive_arrive_in_time import DriveArriveInTime

from util.collision_utils import box_ball_collision_distance, box_ball_location_on_collision


class TestAgent(BaseTestAgent):
    def create_mechanic(self):
        return DriveArriveInTime(self, rendering_enabled=True)

    def get_mechanic_controls(self):

        if not hasattr(self, "state_at_time_vectorized"):
            # importing compiled numba functions late works better.
            from util.physics.drive_1d_time import state_at_time_vectorized

            self.state_at_time_vectorized = state_at_time_vectorized

        ball_prediction = self.game_data.ball_prediction
        car = self.game_data.my_car
        ball = self.game_data.ball

        target_loc = ball.location
        target_dt = 0

        hitbox_height = car.hitbox_corner[2] + car.hitbox_offset[2]
        origin_height = 16  # the car's elevation from the ground due to wheels and suspension

        # only accurate if we're already moving towards the target
        velocity = np.array([np.linalg.norm(car.velocity)] * len(ball_prediction))
        boost = np.array([car.boost] * len(ball_prediction))

        location_slices = ball_prediction["physics"]["location"]

        distance_slices = box_ball_collision_distance(
            location_slices, car.location, car.rotation_matrix, car.hitbox_corner, car.hitbox_offset, ball.radius,
        )
        time_slices = ball_prediction["game_seconds"] - self.game_data.time

        not_too_high = location_slices[:, 2] < ball.radius + hitbox_height + origin_height

        reachable = (self.state_at_time_vectorized(time_slices, velocity, boost)[0] > distance_slices) & (not_too_high)

        filtered_prediction = ball_prediction[reachable]

        if len(filtered_prediction) > 0:
            target_loc = filtered_prediction[0]["physics"]["location"]
            target_dt = filtered_prediction[0]["game_seconds"] - self.game_data.time
            target_loc = box_ball_location_on_collision(
                target_loc, car.location, car.rotation_matrix, car.hitbox_corner, car.hitbox_offset, ball.radius,
            )

        return self.mechanic.step(self.game_data.my_car, target_loc, target_dt)
