from rlutilities.simulation import Car
from rlutilities.mechanics import AerialTurn
from rlutilities.linear_algebra import mat3, vec3, cross, normalize
import random


def score():
    car = Car()

    x_axis = normalize(vec3(*[random.normalvariate(0, 1) for _ in range(3)]))
    y_axis = vec3(*[random.normalvariate(0, 1) for _ in range(3)])
    z_axis = normalize(cross(x_axis, y_axis))
    y_axis = normalize(cross(z_axis, x_axis))

    car.rotation = mat3(x_axis[0], x_axis[1], x_axis[2],
                        y_axis[0], y_axis[1], y_axis[2],
                        z_axis[0], z_axis[1], z_axis[2])

    angular_velocity = normalize(vec3(*[random.normalvariate(0, 1) for _ in range(3)]))
    car.angular_velocity = angular_velocity * random.random() * 5.5

    turn = AerialTurn(car)
    turn.eps_omega = 10
    turn.target = mat3(1, 0, 0, 0, 1, 0, 0, 0, 1)

    end_turn = turn.simulate()

    return end_turn.time


if __name__ == '__main__':

    scores = [score() for _ in range(10000)]

    print(sum(scores) / 10000 / 0.01666)
