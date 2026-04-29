import numpy as np
import pytest
import torch

from geofbpinn.geometry.geometry import point_in_polygon, point_on_polygon_edge
from geofbpinn.geometry.polygon_decomposition import (
    BlockKind,
    PolygonBlock,
    Decomposition2DPolygon,
    polygon_points_sampler,
)


SQUARE = [(0.0, 0.0), (4.0, 0.0), (4.0, 4.0), (0.0, 4.0)]
SQUARE_BBOX_L = (0.0, 0.0)
SQUARE_BBOX_R = (4.0, 4.0)

TRIANGLE = [(0.0, 0.0), (4.0, 0.0), (0.0, 4.0)]
TRIANGLE_BBOX_L = (0.0, 0.0)
TRIANGLE_BBOX_R = (4.0, 4.0)

SQUARE_HOLE = [[(1.0, 1.0), (2.0, 1.0), (2.0, 2.0), (1.0, 2.0)]]


def make_decomp(
    polygon=None,
    bbox_left=None,
    bbox_right=None,
    block_size=(2.0, 2.0),
    overlap=(0.0, 0.0),
    block_scales=None,
    block_shift=None,
    points_per_block=20,
    holes=None,
    eps_full=1e-6,
    device="cpu",
):
    polygon = polygon or SQUARE
    bbox_left = bbox_left or SQUARE_BBOX_L
    bbox_right = bbox_right or SQUARE_BBOX_R
    return Decomposition2DPolygon(
        polygon_vertices=polygon,
        bbox_left=bbox_left,
        bbox_right=bbox_right,
        block_scales=block_scales or [1.0, 1.0],
        block_shift=block_shift or [0.0, 0.0],
        block_size=block_size,
        overlap=overlap,
        points_per_block=points_per_block,
        holes=holes or [],
        eps_full=eps_full,
        device=device,
    )


def test_block_kind_values():
    assert BlockKind.INTERIOR.value == "interior"
    assert BlockKind.BOUNDARY.value == "boundary"


@pytest.mark.parametrize(
    "polygon, holes, n",
    [
        (SQUARE, [], 50),
        (SQUARE, SQUARE_HOLE, 50),
        (TRIANGLE, [], 30),
    ],
)
def test_polygon_points_sampler_shape(polygon, holes, n):
    pts = polygon_points_sampler(polygon, holes)(n)
    assert pts.shape == (n, 2)


@pytest.mark.parametrize(
    "polygon, holes, n",
    [
        (SQUARE, [], 100),
        (SQUARE, SQUARE_HOLE, 100),
        (TRIANGLE, [], 100),
    ],
)
def test_polygon_points_sampler_points_inside_polygon(polygon, holes, n):
    pts = polygon_points_sampler(polygon, holes)(n)
    xs, ys = pts[:, 0], pts[:, 1]
    assert point_in_polygon(xs, ys, np.array(polygon)).all()


@pytest.mark.parametrize(
    "polygon, holes, n",
    [
        (SQUARE, SQUARE_HOLE, 100),
    ],
)
def test_polygon_points_sampler_points_not_in_holes(polygon, holes, n):
    pts = polygon_points_sampler(polygon, holes)(n)
    xs, ys = pts[:, 0], pts[:, 1]
    for hole in holes:
        assert not point_in_polygon(xs, ys, np.array(hole)).any()


def test_polygon_points_sampler_empty_polygon_returns_zeros():
    pts = polygon_points_sampler([], [])(10)
    assert pts.shape == (10, 2)
    assert (pts == 0).all()


@pytest.mark.parametrize(
    "polygon, bbox_left, bbox_right, block_size, overlap, expected_n_blocks",
    [
        (SQUARE, SQUARE_BBOX_L, SQUARE_BBOX_R, (2.0, 2.0), (0.0, 0.0), 4),
        (SQUARE, SQUARE_BBOX_L, SQUARE_BBOX_R, (2.0, 2.0), (0.5, 0.5), 9),
        (SQUARE, SQUARE_BBOX_L, SQUARE_BBOX_R, (4.0, 4.0), (0.0, 0.0), 1),
        (TRIANGLE, TRIANGLE_BBOX_L, TRIANGLE_BBOX_R, (2.0, 2.0), (0.0, 0.0), 3),
    ],
)
def test_block_count(
    polygon, bbox_left, bbox_right, block_size, overlap, expected_n_blocks
):
    d = make_decomp(
        polygon=polygon,
        bbox_left=bbox_left,
        bbox_right=bbox_right,
        block_size=block_size,
        overlap=overlap,
    )
    assert len(d.blocks) == expected_n_blocks


@pytest.mark.parametrize(
    "polygon, bbox_left, bbox_right, block_size, overlap, holes, expected_area_ratios",
    [
        (SQUARE, SQUARE_BBOX_L, SQUARE_BBOX_R, (4.0, 4.0), (0.0, 0.0), [], [1.0]),
        (
            SQUARE,
            SQUARE_BBOX_L,
            SQUARE_BBOX_R,
            (2.0, 2.0),
            (0.0, 0.0),
            [],
            [1.0, 1.0, 1.0, 1.0],
        ),
        (
            TRIANGLE,
            TRIANGLE_BBOX_L,
            TRIANGLE_BBOX_R,
            (2.0, 2.0),
            (0.0, 0.0),
            [],
            [1.0, 0.5, 0.5],
        ),
    ],
)
def test_block_area_ratio(
    polygon, bbox_left, bbox_right, block_size, overlap, holes, expected_area_ratios
):
    d = make_decomp(
        polygon=polygon,
        bbox_left=bbox_left,
        bbox_right=bbox_right,
        block_size=block_size,
        overlap=overlap,
        holes=holes,
    )
    assert len(d.blocks) == len(expected_area_ratios)
    for block, expected in zip(d.blocks, expected_area_ratios):
        assert abs(block.area_ratio - expected) < 1e-6


@pytest.mark.parametrize(
    "polygon, bbox_left, bbox_right, block_size, overlap, holes, expected_n_clipped_holes",
    [
        (
            SQUARE,
            SQUARE_BBOX_L,
            SQUARE_BBOX_R,
            (2.0, 2.0),
            (0.0, 0.0),
            [],
            [0, 0, 0, 0],
        ),
        (
            SQUARE,
            SQUARE_BBOX_L,
            SQUARE_BBOX_R,
            (4.0, 4.0),
            (0.0, 0.0),
            SQUARE_HOLE,
            [1],
        ),
    ],
)
def test_clipped_holes_count(
    polygon, bbox_left, bbox_right, block_size, overlap, holes, expected_n_clipped_holes
):
    d = make_decomp(
        polygon=polygon,
        bbox_left=bbox_left,
        bbox_right=bbox_right,
        block_size=block_size,
        overlap=overlap,
        holes=holes,
    )
    assert [len(b.clipped_holes) for b in d.blocks] == expected_n_clipped_holes


def test_clipped_polygon_vertices_inside_original():
    d = make_decomp(
        polygon=TRIANGLE,
        bbox_left=TRIANGLE_BBOX_L,
        bbox_right=TRIANGLE_BBOX_R,
        block_size=(2.0, 2.0),
        overlap=(0.0, 0.0),
    )
    poly_arr = np.array(TRIANGLE)
    for block in d.blocks:
        for vx, vy in block.clipped_polygon:
            xs, ys = np.array([vx]), np.array([vy])
            inside = point_in_polygon(xs, ys, poly_arr)
            on_edge = point_on_polygon_edge(xs, ys, poly_arr)
            assert inside[0] or on_edge[0]


@pytest.mark.parametrize("points_per_block", [10, 32, 64])
def test_get_data_shape(points_per_block):
    d = make_decomp(points_per_block=points_per_block)
    for block in d.blocks:
        data = block.get_data()
        assert data.shape == (points_per_block, 2)


@pytest.mark.parametrize("n_pts", [16, 64])
def test_window_function_shape_and_positive(n_pts):
    d = make_decomp()
    x = torch.rand(n_pts, 2) * 4.0
    for block in d.blocks:
        w = block.window_function(x)
        assert w.shape == (n_pts, 1)
        assert (w > 0).all()


def test_batched_window_shape():
    d = make_decomp()
    n_pts = 20
    x = torch.rand(n_pts, 2) * 4.0
    w = d.batched_window(x)
    assert w.shape == (len(d.blocks), n_pts, 1)
    assert (w > 0).all()


@pytest.mark.parametrize(
    "scales, shifts",
    [
        ([2.0, 3.0], [1.0, -1.0]),
        ([1.0, 1.0], [0.0, 0.0]),
    ],
)
def test_denorm_propagated_to_all_blocks(scales, shifts):
    d = make_decomp(block_scales=scales, block_shift=shifts)
    for block in d.blocks:
        assert block.out_denorm_scale == scales
        assert block.out_denorm_shift == shifts


@pytest.mark.parametrize(
    "block_size, overlap, expected_remaining",
    [
        ((2.0, 2.0), (0.0, 0.0), 4),
    ],
)
def test_remove_redundant_no_overlap_keeps_all(block_size, overlap, expected_remaining):
    d = make_decomp(block_size=block_size, overlap=overlap)
    d.remove_redundant_blocks(samples_per_block=100, tol=0.01, verbose=False)
    assert len(d.blocks) == expected_remaining
