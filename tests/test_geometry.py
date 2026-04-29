import pytest
import math
from geofbpinn.geometry.geometry import *


@pytest.mark.parametrize(
    "poly, expected",
    [
        ([(0, 0)], 0),
        ([(0, 0), (1, 1)], 0),
        ([(0, 0), (2, 0), (0, 4)], 4),
        ([(0, 0), (0, 5), (5, 5), (5, 0)], 25),
    ],
)
def test_polygon_area(poly, expected):
    assert math.isclose(polygon_area(poly), expected)


@pytest.mark.parametrize(
    "polygon, rect, expected",
    [
        (
            [(0, 0), (2, 0), (2, 2), (0, 2)],
            (0.5, 0.5, 1.5, 1.5),
            [(1.5, 0.5), (1.5, 1.5), (0.5, 1.5), (0.5, 0.5)],
        ),
    ],
)
def test_sh_clip(polygon, rect, expected):
    actual = sutherland_hodgman_clip(polygon, rect)
    assert len(actual) == len(expected)
    for i in range(len(actual)):
        x_act, y_act = actual[i]
        x_exp, y_exp = expected[i]
        assert x_act == x_exp and y_act == y_exp


@pytest.mark.parametrize(
    "x_coords, y_coords, polygon, expected",
    [
        ([0], [0], [(0.0, 0.0), (1.5, 0.0), (1.5, 1.5), (0.0, 1.5)], [True]),
        ([0], [0], [(0.5, 0.5), (1.5, 0.5), (1.5, 1.5), (0.5, 1.5)], [False]),
        ([0], [1], [(0.5, 0.5), (1.5, 0.5), (1.5, 1.5), (0.5, 1.5)], [False]),
        ([1], [1], [(0.5, 0.5), (1.5, 0.5), (1.5, 1.5), (0.5, 1.5)], [True]),
        (
            [0, 0, 0, 1],
            [0, 0, 1, 1],
            [(0.5, 0.5), (1.5, 0.5), (1.5, 1.5), (0.5, 1.5)],
            [False, False, False, True],
        ),
    ],
)
def test_point_in_polygon(x_coords, y_coords, polygon, expected):
    xs = np.array(x_coords)
    ys = np.array(y_coords)
    poly_arr = np.array(polygon)
    actual = point_in_polygon(xs, ys, poly_arr).tolist()
    assert len(actual) == len(expected)
    for i in range(len(actual)):
        assert actual[i] == expected[i]


@pytest.mark.parametrize(
    "polygon, holes, n",
    [
        ([(0, 0), (3, 0), (3, 3), (0, 3)], [[(1, 1), (2, 1), (2, 2), (1, 2)]], 100),
    ],
)
def test_sample_point_in_polygon_with_holes(polygon, holes, n):
    points = sample_points_in_polygon_with_holes(polygon, holes, n)
    assert points.shape[0] == n and points.shape[1] == 2
    xs = points[:, 0]
    ys = points[:, 1]
    for hole in holes:
        assert not point_in_polygon(xs, ys, np.array(hole)).any()
    assert point_in_polygon(xs, ys, np.array(polygon)).all()


@pytest.mark.parametrize(
    "a, b, expected",
    [
        ([0], [0], 0),
        ([0], [1], 1),
        ([0], [10], 10),
        ([0, 0], [0, 10], 10),
        ([0, 0], [3, 4], 5),
        ([1, 2], [4, 6], 5),
        ([0, 0, 0], [1, 2, 2], 3),
        ([-1, 0, 1], [1, 3, 7], 7),
    ],
)
def test_dist(a, b, expected):
    assert dist(a, b) == expected


@pytest.mark.parametrize(
    "a, b, p, expected",
    [
        ([0], [0], [0], True),
        ([0], [1], [0.5], True),
        ([0, 0], [0, 10], [0, 10], True),
        ([0, 0], [0, 10], [0, 5], True),
        ([0, 0], [0, 10], [2, 5], False),
    ],
)
def test_point_on_segment(a, b, p, expected):
    assert point_on_segment(a, b, p) == expected


@pytest.mark.parametrize(
    "x_coords, y_coords, polygon, expected",
    [
        ([0], [0], [(0.0, 0.0), (1.5, 0.0), (1.5, 1.5), (0.0, 1.5)], [True]),
        ([0], [0], [(0.5, 0.5), (1.5, 0.5), (1.5, 1.5), (0.5, 1.5)], [False]),
        ([0], [1], [(0.5, 0.5), (1.5, 0.5), (1.5, 1.5), (0.5, 1.5)], [False]),
        ([1], [1], [(0.5, 0.5), (1.5, 0.5), (1.5, 1.5), (0.5, 1.5)], [False]),
        (
            [0, 0, 0, 1, 0.5],
            [0, 0, 1, 1, 1],
            [(0.5, 0.5), (1.5, 0.5), (1.5, 1.5), (0.5, 1.5)],
            [False, False, False, False, True],
        ),
    ],
)
def test_point_on_polygon_edge(x_coords, y_coords, polygon, expected):
    xs = np.array(x_coords)
    ys = np.array(y_coords)
    poly_arr = np.array(polygon)
    actual = point_on_polygon_edge(xs, ys, poly_arr).tolist()
    assert len(actual) == len(expected)
    for i in range(len(actual)):
        assert actual[i] == expected[i]


@pytest.mark.parametrize(
    "polygon, holes, n",
    [
        ([(0, 0), (4, 0), (4, 4), (0, 4)], [], 50),
        ([(0, 0), (3, 0), (3, 3), (0, 3)], [[(1, 1), (2, 1), (2, 2), (1, 2)]], 80),
    ],
)
def test_sample_points_on_boundary_shape_and_on_edge(polygon, holes, n):
    points = sample_points_on_boundary(polygon, holes, n)
    assert points.shape == (n, 2)
    xs = points[:, 0]
    ys = points[:, 1]
    poly_arr = np.array(polygon)
    assert point_on_polygon_edge(xs, ys, poly_arr).all()


@pytest.mark.parametrize(
    "polygon, holes, n",
    [
        ([(0, 0), (3, 0), (3, 3), (0, 3)], [[(1, 1), (2, 1), (2, 2), (1, 2)]], 80),
    ],
)
def test_sample_points_on_boundary_not_on_holes(polygon, holes, n):
    points = sample_points_on_boundary(polygon, holes, n, include_holes=False)
    xs = points[:, 0]
    ys = points[:, 1]
    for hole in holes:
        hole_arr = np.array(hole)
        assert not point_on_polygon_edge(xs, ys, hole_arr).any()


@pytest.mark.parametrize(
    "polygon, holes, n",
    [
        ([(0, 0), (3, 0), (3, 3), (0, 3)], [[(1, 1), (2, 1), (2, 2), (1, 2)]], 80),
    ],
)
def test_sample_points_on_boundary_include_holes(polygon, holes, n):
    points = sample_points_on_boundary(polygon, holes, n, include_holes=True)
    assert points.shape == (n, 2)
    xs = points[:, 0]
    ys = points[:, 1]
    poly_arr = np.array(polygon)
    hole_arrs = [np.array(h) for h in holes]
    on_outer = point_on_polygon_edge(xs, ys, poly_arr)
    on_holes = np.zeros(n, dtype=bool)
    for h_arr in hole_arrs:
        on_holes |= point_on_polygon_edge(xs, ys, h_arr)
    assert (on_outer | on_holes).all()


@pytest.mark.parametrize(
    "a, b, block, expected",
    [
        ((0.0, 0.5), (2.0, 0.5), [0.5, 0.0, 1.5, 1.0], True),
        ((0.6, 0.6), (0.9, 0.9), [0.5, 0.5, 1.5, 1.5], True),
        ((0.0, 0.0), (0.3, 0.3), [0.5, 0.5, 1.5, 1.5], False),
        ((0.0, 0.0), (0.5, 0.5), [0.5, 0.5, 1.5, 1.5], True),
        ((0.0, 2.0), (2.0, 2.0), [0.5, 0.5, 1.5, 1.5], False),
        ((0.0, 0.0), (2.0, 2.0), [0.5, 0.5, 1.5, 1.5], True),
        ((1.0, 1.0), (3.0, 3.0), [0.5, 0.5, 1.5, 1.5], True),
    ],
)
def test_segment_intersects_rect(a, b, block, expected):
    assert segment_intersects_rect(a, b, block) == expected


@pytest.mark.parametrize(
    "a_min, a_max, b_min, b_max, expected",
    [
        ((0, 0, 0), (2, 2, 2), (1, 1, 1), (3, 3, 3), True),
        ((0, 0, 0), (1, 1, 1), (1, 0, 0), (2, 1, 1), True),
        ((0, 0, 0), (1, 1, 1), (2, 0, 0), (3, 1, 1), False),
        ((0, 0, 0), (1, 1, 1), (0, 0, 2), (1, 1, 3), False),
        ((0, 0, 0), (4, 4, 4), (1, 1, 1), (2, 2, 2), True),
        ((0, 0, 1), (1, 1, 1), (0, 0, 1), (1, 1, 2), True),
        ((0, 0, 0), (1, 1, 0), (0, 0, 1), (1, 1, 2), False),
    ],
)
def test_aabb_intersects_aabb_3d(a_min, a_max, b_min, b_max, expected):
    assert aabb_intersects_aabb_3d(a_min, a_max, b_min, b_max) == expected


@pytest.mark.parametrize(
    "polygon, block, expected",
    [
        (None, [0, 0, 1, 1], True),
        ([(0, 0), (2, 0), (2, 2), (0, 2)], [0.5, 0.5, 1.5, 1.5], True),
        ([(3, 3), (4, 3), (4, 4), (3, 4)], [0.0, 0.0, 1.0, 1.0], False),
        ([(0, 0), (5, 0), (5, 5), (0, 5)], [1.0, 1.0, 2.0, 2.0], True),
        ([(0, 0), (10, 0), (10, 10), (0, 10)], [2.0, 2.0, 3.0, 3.0], True),
        ([(0, 0), (1, 0), (1, 1), (0, 1)], [1.0, 1.0, 2.0, 2.0], False),
        ([(0.0, 0.5), (2.0, 0.5)], [0.5, 0.0, 1.5, 1.0], True),
        ([(0.0, 0.0), (0.3, 0.3)], [0.5, 0.5, 1.5, 1.5], False),
    ],
)
def test_polygon_intersects_rectangle(polygon, block, expected):
    assert polygon_intersects_rectangle(polygon, block) == expected
