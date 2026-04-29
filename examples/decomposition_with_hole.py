import numpy as np
from geofbpinn.geometry.plot import plot_decomposition2d
from geofbpinn.geometry.polygon_decomposition import Decomposition2DPolygon
from examples.naca4 import naca4_polygon

outer_rect = [
    (-0.5, -0.5),
    (1.5, -0.5),
    (1.5, 0.8),
    (-0.5, 0.8),
]  # outer rectangle polygon
outer_poly = outer_rect

wing = naca4_polygon("2412", chord=1.0, n_pts=600)

# bounding box for decomposition (choose a rectangle)
xs = np.array([p[0] for p in outer_poly])
ys = np.array([p[1] for p in outer_poly])
bbox_left = (xs.min(), ys.min())
bbox_right = (xs.max(), ys.max())

dec = Decomposition2DPolygon(
    polygon_vertices=outer_poly,
    bbox_left=bbox_left,
    bbox_right=bbox_right,
    block_size=(0.3, 0.25),
    overlap=(0.06, 0.06),
    block_scales=[1],
    block_shift=[1],
    points_per_block=300,
    holes=[wing],
    eps_full=1e-3,
)
print("Blocks generated:", len(dec.blocks))
plot_decomposition2d(
    dec.blocks,
    polygon_vertices=outer_poly,
    holes=[wing],
    figsize=(10, 5),
    savepath="hole.png",
)
