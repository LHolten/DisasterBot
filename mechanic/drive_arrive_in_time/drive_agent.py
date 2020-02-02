import numpy as np

from mechanic.base_test_agent import BaseTestAgent
from mechanic.drive_arrive_in_time import DriveArriveInTime

from util.drive_physics_numpy import distance_traveled_numpy
from util.collision_utils import box_ball_collision_distance, box_ball_location_on_collision


class TestAgent(BaseTestAgent):

    def create_mechanic(self):
        return DriveArriveInTime(self)

    def get_mechanic_controls(self):

        ball_prediction = self.game_data.ball_prediction

        target_loc = self.game_data.ball.location
        target_dt = 0

        # only accurate if we're already moving towards the target
        velocity = np.array([np.linalg.norm(self.game_data.my_car.velocity)])
        boost = np.array([self.game_data.my_car.boost])

        location_slices = ball_prediction["physics"]["location"]
        distance_slices = box_ball_collision_distance(location_slices,
                                                      self.game_data.my_car.location,
                                                      self.game_data.my_car.rotation_matrix,
                                                      self.game_data.my_car.hitbox_corner,
                                                      self.game_data.my_car.hitbox_offset,
                                                      self.game_data.ball.radius)
        time_slices = ball_prediction["game_seconds"] - self.game_data.time

        reachable = (distance_traveled_numpy(time_slices, velocity, boost) >
                     distance_slices) & (location_slices[:, 2] < 120)

        filtered_prediction = ball_prediction[reachable]

        if len(filtered_prediction) > 0:
            target_loc = filtered_prediction[0]["physics"]["location"]
            target_dt = filtered_prediction[0]["game_seconds"] - self.game_data.time
            target_loc = box_ball_location_on_collision(target_loc,
                                                        self.game_data.my_car.location,
                                                        self.game_data.my_car.rotation_matrix,
                                                        self.game_data.my_car.hitbox_corner,
                                                        self.game_data.my_car.hitbox_offset,
                                                        self.game_data.ball.radius)

        return self.mechanic.step(self.game_data.my_car, target_loc, target_dt)
