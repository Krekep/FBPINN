import numpy as np
import math

from geofbpinn.geometry.polygon_decomposition import Decomposition2DPolygon


def naca4_polygon(
    naca: str, chord: float = 1.0, n_pts: int = 600
) -> list[tuple[float, float]]:
    assert len(naca) == 4, "naca must be a string of 4 digits, e.g. '2412' or '0012'"
    m = int(naca[0]) / 100.0
    p = int(naca[1]) / 10.0
    t = int(naca[2:]) / 100.0
    beta = np.linspace(0, math.pi, n_pts)
    x = (1 - np.cos(beta)) / 2.0
    yt = (
        5
        * t
        * (
            0.2969 * np.sqrt(x)
            - 0.1260 * x
            - 0.3516 * x**2
            + 0.2843 * x**3
            - 0.1015 * x**4
        )
    )
    yc = np.zeros_like(x)
    dyc_dx = np.zeros_like(x)
    for i, xi in enumerate(x):
        if p == 0:
            yc[i] = 0.0
            dyc_dx[i] = 0.0
        elif xi < p:
            yc[i] = m / (p**2) * (2 * p * xi - xi**2)
            dyc_dx[i] = 2 * m / (p**2) * (p - xi)
        else:
            yc[i] = m / ((1 - p) ** 2) * ((1 - 2 * p) + 2 * p * xi - xi**2)
            dyc_dx[i] = 2 * m / ((1 - p) ** 2) * (p - xi)
    theta = np.arctan(dyc_dx)
    xu = x - yt * np.sin(theta)
    yu = yc + yt * np.cos(theta)
    xl = x + yt * np.sin(theta)
    yl = yc - yt * np.cos(theta)
    coords = []
    for xi, yi in zip(xu, yu):
        coords.append((xi * chord, yi * chord))
    for xi, yi in zip(xl[::-1], yl[::-1]):
        coords.append((xi * chord, yi * chord))
    return coords


if __name__ == "__main__":
    from geofbpinn.geometry.plot import plot_decomposition2d

    np.random.seed(0)
    airfoil = naca4_polygon("2412", chord=1.0, n_pts=600)
    xs = np.array([p[0] for p in airfoil])
    ys = np.array([p[1] for p in airfoil])
    bbox_left = (xs.min() - 0.1, ys.min() - 0.1)
    bbox_right = (xs.max() + 0.1, ys.max() + 0.1)

    block_size = (0.18, 0.12)
    overlap = (0.04, 0.04)
    points_per_block = 200
    boundary_points_per_block = 60
    eps_full = 1e-3

    dec = Decomposition2DPolygon(
        polygon_vertices=airfoil,
        bbox_left=bbox_left,
        bbox_right=bbox_right,
        block_size=block_size,
        block_scales=[1],
        block_shift=[1],
        window_fn_type="sigmoid",
        overlap=overlap,
        points_per_block=points_per_block,
        eps_full=eps_full,
    )
    print("Result items:", len(dec.blocks))
    print("Blocks per axis:", list(map(len, dec.blocks_per_axis)))
    plot_decomposition2d(
        dec.blocks,
        polygon_vertices=airfoil,
        figsize=(12, 12),
        savepath="decomposition_airwing.png",
        title="Decomposition of airplane wing",
    )
    dec.remove_redundant_blocks(samples_per_block=2000, tol=0.0001, verbose=False)
    print("Result items after cleanup:", len(dec.blocks))
    print("Blocks per axis:", list(map(len, dec.blocks_per_axis)))
    plot_decomposition2d(
        dec.blocks,
        polygon_vertices=airfoil,
        figsize=(12, 12),
        savepath="decomposition_airwing_cleanup.png",
        title="Decomposition of airplane wing",
    )
