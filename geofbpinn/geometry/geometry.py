import math
from typing import Tuple, List, Optional
import numpy as np
from collections.abc import Sequence


def polygon_area(poly: List[Tuple[float, float]]) -> float:
    """
    Compute the area of a polygon using the shoelace formula.

    Parameters
    ----------
    poly : List[Tuple[float, float]]
        List of (x, y) coordinates representing the polygon vertices.
        The polygon is assumed to be closed (first and last points connected).

    Returns
    -------
    float
        The absolute area of the polygon.
    """
    if not poly or len(poly) < 3:
        return 0.0
    x = np.array([p[0] for p in poly])
    y = np.array([p[1] for p in poly])
    x2 = np.append(x, x[0])
    y2 = np.append(y, y[0])
    return 0.5 * abs(np.sum(x2[:-1] * y2[1:] - x2[1:] * y2[:-1]))


def sutherland_hodgman_clip(
    subject_polygon: List[Tuple[float, float]],
    clip_rect: Tuple[float, float, float, float],
):
    """
    Clip a polygon against a rectangular window using Sutherland-Hodgman algorithm.

    Parameters
    ----------
    subject_polygon : List[Tuple[float, float]]
        List of (x, y) coordinates representing the polygon to be clipped.
    clip_rect : Tuple[float, float, float, float]
        Clipping rectangle as (xmin, ymin, xmax, ymax).

    Returns
    -------
    List[Tuple[float, float]]
        Clipped polygon vertices as list of (x, y) coordinates.
        Returns empty list if no intersection or polygon has less than 3 vertices.

    Examples
    --------
    >>> polygon = [(0, 0), (2, 0), (2, 2), (0, 2)]
    >>> clip_rect = (0.5, 0.5, 1.5, 1.5)
    >>> clipped = sutherland_hodgman_clip(polygon, clip_rect)
    >>> print(clipped)
    [(0.5, 0.5), (1.5, 0.5), (1.5, 1.5), (0.5, 1.5)]
    """
    xmin, ymin, xmax, ymax = clip_rect

    def inside(p, edge):
        x, y = p
        if edge == "left":
            return x >= xmin - 1e-12
        if edge == "right":
            return x <= xmax + 1e-12
        if edge == "bottom":
            return y >= ymin - 1e-12
        if edge == "top":
            return y <= ymax + 1e-12

    def intersect(p1, p2, edge):
        x1, y1 = p1
        x2, y2 = p2
        if edge in ("left", "right"):
            x_clip = xmin if edge == "left" else xmax
            if abs(x2 - x1) < 1e-15:
                return (x_clip, y1)
            t = (x_clip - x1) / (x2 - x1)
            y = y1 + t * (y2 - y1)
            return (x_clip, y)
        else:
            y_clip = ymin if edge == "bottom" else ymax
            if abs(y2 - y1) < 1e-15:
                return (x1, y_clip)
            t = (y_clip - y1) / (y2 - y1)
            x = x1 + t * (x2 - x1)
            return (x, y_clip)

    outputList = subject_polygon
    for edge in ("left", "top", "right", "bottom"):
        inputList = outputList
        outputList = []
        if not inputList:
            break
        s = inputList[-1]
        for e in inputList:
            if inside(e, edge):
                if not inside(s, edge):
                    inter_pt = intersect(s, e, edge)
                    outputList.append(inter_pt)
                outputList.append(e)
            elif inside(s, edge):
                inter_pt = intersect(s, e, edge)
                outputList.append(inter_pt)
            s = e
    if not outputList:
        return []

    cleaned = [outputList[0]]
    for p in outputList[1:]:
        if (abs(p[0] - cleaned[-1][0]) > 1e-9) or (abs(p[1] - cleaned[-1][1]) > 1e-9):
            cleaned.append(p)
    if len(cleaned) < 3:
        return []
    return cleaned


def point_in_polygon(
    xs: np.ndarray, ys: np.ndarray, poly_arr: np.ndarray
) -> np.ndarray:
    """
    Determine if points is inside a polygon using ray casting algorithm.

    Parameters
    ----------
    xs: np.ndarray
        X-coordinates of the points to test (N, )
    ys: np.ndarray
        Y-coordinates of the point to test (N, )
    poly_arr: np.ndarray
        Array of (x, y) coordinates representing the polygon vertices.
        Polygon is treated as closed (first and last points connected).

    Returns
    -------
    np.ndarray
        Boolean array of shape (N,). True where point is inside the polygon.
    """
    px = poly_arr[:, 0]  # (V,)
    py = poly_arr[:, 1]  # (V,)
    px_next = np.roll(px, -1)  # (V,)
    py_next = np.roll(py, -1)  # (V,)

    xs_col = xs[:, np.newaxis]  # (N, 1)
    ys_col = ys[:, np.newaxis]  # (N, 1)

    cond1 = (py_next > ys_col) != (py > ys_col)  # (N, V)
    denom = np.where(np.abs(py_next - py) < 1e-30, 1e-30, py_next - py)
    x_intersect = (px_next - px) * (ys_col - py) / denom + px  # (N, V)
    cond2 = xs_col < x_intersect  # (N, V)

    return (np.sum(cond1 & cond2, axis=1) % 2) == 1  # (N,)


def sample_points_in_polygon_with_holes(
    poly: List[Tuple[float, float]], holes: List[List[Tuple[float, float]]], n: int
) -> np.ndarray[tuple[int, int], np.dtype[np.float32]]:
    """
    Generate random points inside a polygon but outside any holes.

    Parameters
    ----------
    poly : List[Tuple[float, float]]
        List of (x, y) coordinates representing the outer polygon boundary.
    holes : List[List[Tuple[float, float]]]
        List of polygons representing holes. Each hole is a list of (x, y) coordinates.
    n : int
        Number of points to sample.

    Returns
    -------
    np.ndarray[tuple[int, int], np.dtype[np.float32]]
        Array of shape (n, 2) containing sampled points as (x, y) coordinates.
        If polygon is invalid, returns zeros array of shape (0, 2).

    Examples
    --------
    >>> outer = [(0, 0), (3, 0), (3, 3), (0, 3)]
    >>> hole = [(1, 1), (2, 1), (2, 2), (1, 2)]
    >>> points = sample_points_in_polygon_with_holes(outer, [hole], 100)
    >>> points.shape
    (100, 2)
    """
    if not poly or len(poly) < 3:
        return np.zeros((0, 2), dtype=float)

    poly_arr = np.asarray(poly, dtype=float)
    holes_arr = [np.asarray(h, dtype=float) for h in holes if h and len(h) >= 3]

    minx, maxx = poly_arr[:, 0].min(), poly_arr[:, 0].max()
    miny, maxy = poly_arr[:, 1].min(), poly_arr[:, 1].max()

    bbox_area = (maxx - minx) * (maxy - miny)
    poly_area = polygon_area(poly)
    holes_area = sum(polygon_area(list(map(tuple, h))) for h in holes_arr)
    fill_ratio = (
        max((poly_area - holes_area) / bbox_area, 0.01) if bbox_area > 0 else 0.01
    )

    batch_size = int(n / fill_ratio * 1.5)

    collected: List[np.ndarray] = []
    total = 0
    max_iters = 50

    for _ in range(max_iters):
        if total >= n:
            break
        rx = np.random.uniform(minx, maxx, batch_size)
        ry = np.random.uniform(miny, maxy, batch_size)

        mask = point_in_polygon(rx, ry, poly_arr)
        for h_arr in holes_arr:
            mask &= ~point_in_polygon(rx, ry, h_arr)

        pts = np.stack([rx[mask], ry[mask]], axis=1)
        if len(pts) > 0:
            collected.append(pts)
            total += len(pts)

    if not collected:
        return np.zeros((0, 2), dtype=float)

    return np.concatenate(collected, axis=0)[:n]


def dist(a: Sequence[float], b: Sequence[float]) -> float:
    """
    Compute the Euclidean distance between two points.

    Parameters
    ----------
    a : Sequence[float]
        Coordinates of the first point.
    b : Sequence[float]
        Coordinates of the second point.

    Returns
    -------
    float
        Euclidean distance between `a` and `b`.

    Raises
    ------
    AssertionError
        If the lengths of `a` and `b` differ.
    """
    res = 0
    assert len(a) == len(
        b
    ), "Can not calculate distance between points with different dimensions"
    for i, j in zip(a, b):
        res += (i - j) * (i - j)
    return math.sqrt(res)


def point_on_segment(
    a: Sequence[float], b: Sequence[float], p: Sequence[float]
) -> bool:
    """
    Check if a point `p` lies on a line `ab` segment.

    Parameters
    ----------
    a: Sequence[float]
        First endpoint of the segment.
    b: Sequence[float]
        Second endpoint of the segment.
    p: Sequence[float]
        Point to test.

    Returns
    -------
    bool
        True if `p` lies on the closed segment `ab`, False otherwise.
    """
    dist_ap = dist(a, p)
    dist_bp = dist(b, p)
    dist_ab = dist(a, b)

    return math.isclose(dist_ap + dist_bp, dist_ab, abs_tol=1e-6)


def point_on_polygon_edge(
    xs: np.ndarray, ys: np.ndarray, poly_arr: np.ndarray, tol=1e-9
) -> np.ndarray:
    """
    Determine if points lie on any edge of a polygon.

    Parameters
    ----------
    xs: np.ndarray
        X-coordinates of the points to test (N, )
    ys: np.ndarray
        Y-coordinates of the point to test (N, )
    poly_arr: np.ndarray
        Array of (x, y) coordinates representing the polygon vertices.
        Polygon is treated as closed (first and last points connected).
    tol : float
        Tolerance for collinearity check.

    Returns
    -------
    np.ndarray
        Boolean array of shape (N,). True where point lie exactly on polygon edge.
    """
    ax = poly_arr[:, 0]  # (V,)
    ay = poly_arr[:, 1]
    bx = np.roll(ax, -1)  # (V,)
    by = np.roll(ay, -1)

    ab = np.hypot(bx - ax, by - ay)  # (V,)

    xs_col = xs[:, np.newaxis]  # (N, 1)
    ys_col = ys[:, np.newaxis]

    ap = np.hypot(xs_col - ax, ys_col - ay)  # (N, V)
    bp = np.hypot(xs_col - bx, ys_col - by)  # (N, V)

    return np.any(np.abs(ap + bp - ab) < tol, axis=1)  # (N,)


def sample_points_on_boundary(
    poly: List[Tuple[float, float]],
    holes: List[List[Tuple[float, float]]],
    n: int,
    block_rect: Optional[Tuple[float, float, float, float]] = None,
    include_holes: bool = False,
    original_polygon: Optional[List[Tuple[float, float]]] = None,
) -> np.ndarray:
    """
    Sample points uniformly along the boundary of a polygon.

    Parameters
    ----------
    poly : List[Tuple[float, float]]
        List of (x, y) coordinates representing the polygon vertices.
    holes : List[List[Tuple[float, float]]]
        List of polygons representing holes. Each hole is a list of (x, y) coordinates.
    n : int
        Number of points to sample along the boundary.
    block_rect : Optional[Tuple[float, float, float, float]]
        Block rectangle as (xmin, ymin, xmax, ymax). If provided, points on block boundaries are excluded
    include_holes : bool
        Will points samples on holes boundary
    original_polygon : Optional[List[Tuple[float, float]]]
        Original polygon before clipping. Used to check if point on block boundary is actually on polygon boundary.

    Returns
    -------
    np.ndarray
        Array of shape (n, 2) containing sampled points as (x, y) coordinates.
        Returns zeros array of shape (0, 2) for invalid polygon.

    Examples
    --------
    >>> triangle = [(0, 0), (2, 0), (1, 2)]
    >>> hole = []
    >>> points = sample_points_on_boundary(triangle, hole, 50)
    >>> points.shape
    (50, 2)
    """
    if not poly or len(poly) < 2:
        return np.zeros((0, 2), dtype=np.float32)

    edges_a: List[Tuple[float, float]] = []
    edges_b: List[Tuple[float, float]] = []
    edge_lengths: List[float] = []
    edge_is_outer: List[bool] = []

    for i in range(len(poly)):
        a = poly[i]
        b = poly[(i + 1) % len(poly)]
        L = math.hypot(b[0] - a[0], b[1] - a[1])
        if L > 1e-12:
            edges_a.append(a)
            edges_b.append(b)
            edge_lengths.append(L)
            edge_is_outer.append(True)

    if include_holes:
        for hole in holes:
            if not hole or len(hole) < 2:
                continue
            for i in range(len(hole)):
                a = hole[i]
                b = hole[(i + 1) % len(hole)]
                L = math.hypot(b[0] - a[0], b[1] - a[1])
                if L > 1e-12:
                    edges_a.append(a)
                    edges_b.append(b)
                    edge_lengths.append(L)
                    edge_is_outer.append(False)

    if not edge_lengths:
        return np.zeros((0, 2), dtype=float)

    len_arr = np.array(edge_lengths, dtype=float)  # (E,)
    cum_len = np.concatenate([[0.0], np.cumsum(len_arr)])  # (E+1,)
    total_len = cum_len[-1]

    a_arr = np.array(edges_a, dtype=float)  # (E, 2)
    b_arr = np.array(edges_b, dtype=float)  # (E, 2)
    is_outer_arr = np.array(edge_is_outer, dtype=bool)  # (E,)

    holes_arr = [np.asarray(h, dtype=float) for h in holes if h and len(h) >= 3]
    orig_arr = (
        np.asarray(original_polygon, dtype=float)
        if original_polygon is not None
        else None
    )

    collected: List[np.ndarray] = []
    total = 0
    max_iters = 20

    for _ in range(max_iters):
        if total >= n:
            break

        batch_n = n

        r_vals = np.random.uniform(0.0, total_len, batch_n)  # (B,)
        # searchsorted on cum_len[1:] gives index of the matching edge
        idx = np.searchsorted(cum_len[1:], r_vals, side="left")  # (B,)
        idx = np.clip(idx, 0, len(len_arr) - 1)

        t = (r_vals - cum_len[idx]) / len_arr[idx]  # (B,)
        xy = a_arr[idx] + t[:, np.newaxis] * (b_arr[idx] - a_arr[idx])  # (B, 2)
        xs, ys = xy[:, 0], xy[:, 1]

        mask = np.ones(batch_n, dtype=bool)

        # ---- block-boundary filter ------------------------------------ #
        if block_rect is not None:
            xmin, ymin, xmax, ymax = block_rect
            on_bb = (
                (np.abs(xs - xmin) < 1e-9)
                | (np.abs(xs - xmax) < 1e-9)
                | (np.abs(ys - ymin) < 1e-9)
                | (np.abs(ys - ymax) < 1e-9)
            )
            if orig_arr is not None:
                on_orig = point_on_polygon_edge(xs, ys, orig_arr)
                on_bb &= ~on_orig
            mask &= ~on_bb

        if holes_arr:
            outer_mask = is_outer_arr[idx]  # (B,) — only outer-edge points
            for h_arr in holes_arr:
                in_hole = point_in_polygon(xs, ys, h_arr)
                mask &= ~(outer_mask & in_hole)

        valid = xy[mask]
        if len(valid) > 0:
            collected.append(valid)
            total += len(valid)

    if not collected:
        return np.zeros((0, 2), dtype=np.float32)

    result = np.concatenate(collected, axis=0)
    assert len(result) >= n
    return result[:n].astype(np.float32)


def segment_intersects_rect(
    a: tuple[float, float], b: tuple[float, float], block: list[float]
) -> bool:
    """
    Checking the intersection of a segment and a rectangle

    Parameters
    ----------
    a: tuple[float, float]
        start point of segment
    b: tuple[float, float]
        end point of segment
    block: list[float]
        rectangle corners

    Returns
    -------
    is_intersection: bool
    """
    xmin, ymin, xmax, ymax = block
    if (
        (a[0] < xmin and b[0] < xmin)
        or (a[0] > xmax and b[0] > xmax)
        or (a[1] < ymin and b[1] < ymin)
        or (a[1] > ymax and b[1] > ymax)
    ):
        return False

    def orient(p, q, r):
        return (q[0] - p[0]) * (r[1] - p[1]) - (q[1] - p[1]) * (r[0] - p[0])

    def on_segment(p, q, r):
        return min(p[0], q[0]) <= r[0] <= max(p[0], q[0]) and min(p[1], q[1]) <= r[
            1
        ] <= max(p[1], q[1])

    def segments_intersect(
        p1: tuple[float, float],  # x of segment
        p2: tuple[float, float],  # y of segment
        p3: tuple[float, float],  # x of rectangle edge
        p4: tuple[float, float],  # y of rectangle edge
    ):
        o1 = orient(p1, p2, p3)
        o2 = orient(p1, p2, p4)
        o3 = orient(p3, p4, p1)
        o4 = orient(p3, p4, p2)
        if o1 == 0 and on_segment(p1, p2, p3):
            return True
        if o2 == 0 and on_segment(p1, p2, p4):
            return True
        if o3 == 0 and on_segment(p3, p4, p1):
            return True
        if o4 == 0 and on_segment(p3, p4, p2):
            return True
        return (o1 > 0) != (o2 > 0) and (o3 > 0) != (o4 > 0)

    left = ((xmin, ymin), (xmin, ymax))
    right = ((xmax, ymin), (xmax, ymax))
    bottom = ((xmin, ymin), (xmax, ymin))
    top = ((xmin, ymax), (xmax, ymax))
    for side in (left, right, bottom, top):
        if segments_intersect(a, b, side[0], side[1]):
            return True

    if (xmin <= a[0] <= xmax and ymin <= a[1] <= ymax) or (
        xmin <= b[0] <= xmax and ymin <= b[1] <= ymax
    ):
        return True
    return False


def aabb_intersects_aabb_3d(
    a_min: Tuple[float, float, float],
    a_max: Tuple[float, float, float],
    b_min: Tuple[float, float, float],
    b_max: Tuple[float, float, float],
    tol: float = 1e-12,
) -> bool:
    """
    Check whether two axis-aligned bounding boxes (AABBs) intersect.
    Works for proper boxes and degenerate ones (faces/edges/points),
    since faces in sub_losses are degenerate AABBs (one dimension collapsed).

    Intersection condition per axis: a_min[i] <= b_max[i] AND b_min[i] <= a_max[i].

    Parameters
    ----------
    a_min : Tuple[float, float, float]
        Minimum corner (xmin, ymin, zmin) of the first AABB
    a_max : Tuple[float, float, float]
        Maximum corner (xmax, ymax, zmax) of the first AABB
    b_min : Tuple[float, float, float]
        Minimum corner (xmin, ymin, zmin) of the second AABB
    b_max : Tuple[float, float, float]
        Maximum corner (xmax, ymax, zmax) of the second AABB
    tol : float
        Tolerance

    Returns
    -------
    bool
        True if the two AABBs intersect (including touching), False otherwise
    """
    return all(
        a_min[i] <= b_max[i] + tol and b_min[i] <= a_max[i] + tol for i in range(3)
    )


def polygon_intersects_rectangle(
    polygon: Optional[list[tuple[float, float]]], block: list[float]
) -> bool:
    """
    Checking the intersection of a segment and a rectangle

    Parameters
    ----------
    polygon: Optional[list[tuple[float, float]]]
        Polygon as list of points
    block: list[float]
        rectangle corners

    Returns
    -------
    is_intersection: bool
    """
    if polygon is None:
        return True

    unique_pts = set(polygon)

    if len(unique_pts) == 1 and len(block) == 2:
        return point_on_segment((block[0],), (block[1],), polygon)
    if len(unique_pts) == 2 and len(block) == 4:
        return segment_intersects_rect(polygon[0], polygon[1], block)
    if len(block) == 6:
        region_min, region_max = polygon[0], polygon[1]
        block_min = tuple(block[:3])
        block_max = tuple(block[3:])
        return aabb_intersects_aabb_3d(region_min, region_max, block_min, block_max)

    if polygon[0] != polygon[-1]:
        poly_for_clip = polygon + [polygon[0]]
    else:
        poly_for_clip = polygon
    clipped = sutherland_hodgman_clip(poly_for_clip, block)
    return len(clipped) > 0


if __name__ == "__main__":
    polygon = [(0, 0), (2, 2), (2, 4), (0, 6), (4, 6), (4, 0)]
    rect = (0, 0, 4, 4)

    c = sutherland_hodgman_clip(polygon, rect)
    print(c)
    poly_arr = np.asarray(polygon, dtype=float)
    test_points = [
        (2, 1),
        (0, 0),
        (2, 2),
        (3, 2),
        (2, 5),
        (2, 3),
        (1, 3),
        (1, 2),
        (1, 1),
    ]
    xs = np.array([p[0] for p in test_points], dtype=float)
    ys = np.array([p[1] for p in test_points], dtype=float)
    results = point_in_polygon(xs, ys, poly_arr)
    for (x, y), res in zip(test_points, results):
        print(f"point_in_poly({x}, {y}) = {res}")
