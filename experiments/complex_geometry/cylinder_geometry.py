from math import sin, cos, pi

from geofbpinn.geometry.plot import plot_decomposition2d
from geofbpinn.geometry.polygon_decomposition import Decomposition2DPolygon

scale = 1 / 1000  # mm -> m


def prepare_geometry():
    outer_rect = [
        (0.0, 0.0),
        (0.0, 1200.0 * scale),
        (2000.0 * scale, 1200.0 * scale),
        (2000.0 * scale, 0.0),
    ]  # m

    hole_center = (300 * scale, 600 * scale)  # m
    hole_radius = 100 * scale  # m
    n_points = 200
    angle = (2 * pi) / n_points

    hole_points = []
    for i in range(n_points):
        x = cos(angle * i) * hole_radius + hole_center[0]
        y = sin(angle * i) * hole_radius + hole_center[1]
        hole_points.append((x, y))

    return outer_rect, hole_points


if __name__ == "__main__":
    domain, hole = prepare_geometry()
    block_size = (2, 2)
    overlap = (0.4, 0.4)
    bbox_left = (-0.5, -0.5)
    bbox_right = (2000.5 * scale, 1200.5 * scale)
    points_per_block = 200
    boundary_points_per_block = 60
    eps_full = 1e-3

    dec = Decomposition2DPolygon(
        polygon_vertices=domain,
        bbox_left=bbox_left,
        bbox_right=bbox_right,
        block_scales=[1],
        block_shift=[0],
        block_size=block_size,
        overlap=overlap,
        points_per_block=points_per_block,
        eps_full=eps_full,
        holes=[hole],
    )
    print("Result items:", len(dec.blocks))
    plot_decomposition2d(
        dec.blocks,
        polygon_vertices=domain,
        figsize=(12, 4),
        holes=[hole],
        savepath="pde.png",
    )

    dec.remove_redundant_blocks(samples_per_block=2000, tol=0.0001, verbose=False)
    print("Result items:", len(dec.blocks))
    plot_decomposition2d(
        dec.blocks,
        polygon_vertices=domain,
        figsize=(12, 4),
        holes=[hole],
        savepath="pde_clear.png",
    )
