import numpy as np


def render_local_line_3d(renderer, local_line, origin_loc, origin_rot_matrix, color):
    """Uses renderer.draw_line_3d to draw a line from local coordinates."""
    point1 = origin_rot_matrix.dot(local_line[0]) + origin_loc
    point2 = origin_rot_matrix.dot(local_line[1]) + origin_loc
    renderer.draw_line_3d(point1, point2, color)


def render_hitbox(renderer, my_loc, my_rot, color, hitbox, hitbox_offset):
    """Uses the renderer to draw a wireframe view of the car's hitbox."""

    signs = ([1, 1, 1], [-1, -1, 1], [-1, 1, -1], [1, -1, -1])

    for s in signs:
        point = hitbox / 2 * np.array(s)
        for i in range(3):
            ss = np.array([1, 1, 1])
            ss[i] *= -1
            line = np.array([point + hitbox_offset, point * ss + hitbox_offset])
            render_local_line_3d(renderer, line, my_loc, my_rot, color)
