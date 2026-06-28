import math
from functools import reduce

import pytest
import torch

from geofbpinn.geometry.base_decomposition import (
    RectangleDomain,
    Block,
    BaseDecomposition,
)
from geofbpinn.geometry.decomposition import DecompositionND


def make_decomp(
    bbox_left,
    bbox_right,
    block_size,
    overlap,
    domain_left=None,
    domain_right=None,
    block_scales=None,
    block_shift=None,
    points_per_block=10,
    omega=None,
    device="cpu",
):
    n = len(bbox_left)
    domain = RectangleDomain(
        left_corner=domain_left or bbox_left,
        right_corner=domain_right or bbox_right,
    )
    return DecompositionND(
        domain=domain,
        bbox_left=bbox_left,
        bbox_right=bbox_right,
        block_size=block_size,
        block_scales=block_scales or [1.0] * n,
        block_shift=block_shift or [0.0] * n,
        points_per_block=points_per_block,
        overlap=overlap,
        omega=omega,
        device=device,
    )


def make_block(lc, rc, n_pts=10, device="cpu"):
    n = len(lc)

    def sampler(n_samples):
        return torch.rand([n_samples, n], device=device) * (
            torch.tensor(rc) - torch.tensor(lc)
        ) + torch.tensor(lc)

    return Block(
        data_shape=[n_pts, n],
        left_corner=lc,
        right_corner=rc,
        window_function=lambda x: x,
        out_denorm_scale=[1.0] * n,
        out_denorm_shift=[0.0] * n,
        sampler=sampler,
        device=device,
    )


@pytest.mark.parametrize(
    "lc, rc, n_pts",
    [
        ([0.0], [1.0], 10),
        ([0.0, 0.0], [1.0, 2.0], 20),
        ([-1.0, 0.0, 0.0], [1.0, 1.0, 1.0], 5),
    ],
)
def test_block_get_data_shape(lc, rc, n_pts):
    block = make_block(lc, rc, n_pts)
    data = block.get_data()
    assert data.shape == (n_pts, len(lc))


@pytest.mark.parametrize(
    "lc, rc",
    [
        ([0.0], [1.0]),
        ([0.0, 0.0], [1.0, 2.0]),
        ([0.5, -1.0], [1.5, 0.0]),
    ],
)
def test_block_pool_values_in_range(lc, rc):
    block = make_block(lc, rc, n_pts=32)
    data = block.get_data()
    assert (data >= torch.tensor(lc) - 1e-5).all()
    assert (data <= torch.tensor(rc) + 1e-5).all()


@pytest.mark.parametrize(
    "lc, rc",
    [
        ([0.0], [1.0]),
        ([0.0, 0.0], [1.0, 2.0]),
    ],
)
def test_block_normalization_range(lc, rc):
    block = make_block(lc, rc, n_pts=32)
    data = block.get_data()
    normed = block.normalization(data)
    assert (normed >= -1.0 - 1e-5).all()
    assert (normed <= 1.0 + 1e-5).all()


@pytest.mark.parametrize(
    "lc, rc, scale, shift",
    [
        ([0.0], [1.0], [2.0], [1.0]),
        ([0.0, 0.0], [1.0, 1.0], [3.0, 0.5], [-1.0, 2.0]),
    ],
)
def test_block_unnormalization_zero_input(lc, rc, scale, shift):
    n = len(lc)
    block = Block(
        data_shape=[8, n],
        left_corner=lc,
        right_corner=rc,
        window_function=lambda x: x,
        out_denorm_scale=torch.tensor(scale, device="cpu"),
        out_denorm_shift=torch.tensor(shift, device="cpu"),
        sampler=None,
        device="cpu",
    )
    x = torch.zeros(8, n)
    result = block.unnormalization(x)
    expected = torch.tensor(shift, dtype=torch.float32).expand(8, -1)
    assert torch.allclose(result, expected, atol=1e-5)


@pytest.mark.parametrize(
    "lc, rc, loss_var_bounds, expected_count",
    [
        (
            [0.0, 0.0],
            [2.0, 2.0],
            [("L1", [(0.5, 1.5), None]), ("L2", [None, (0.5, 1.5)])],
            2,
        ),
        ([0.0, 0.0], [1.0, 1.0], [("L_out", [(2.0, 3.0), None])], 0),
        ([0.0], [1.0], [], 0),
    ],
)
def test_block_set_losses(lc, rc, loss_var_bounds, expected_count):
    block = make_block(lc, rc)
    block.set_losses(loss_var_bounds)
    assert len(block.get_losses()) == expected_count


@pytest.mark.parametrize(
    "bbox_left, bbox_right, block_size, overlap, expected",
    [
        ([0.0], [1.0], [0.5], [0.05], [3]),
        ([0.0], [1.0], [0.5], [0.1], [3]),
        ([0.0], [1.0], [1.0], [0.1], [1]),
        ([0.0], [1.0], [2.0], [0.1], [1]),
        ([0.0, 0.0], [1.0, 2.0], [0.5, 0.5], [0.05, 0.05], [3, 5]),
        ([0.0, 0.0], [0.9, 0.9], [0.6, 0.6], [0.2, 0.2], [2, 2]),
    ],
)
def test_num_blocks_per_axis(bbox_left, bbox_right, block_size, overlap, expected):
    d = make_decomp(bbox_left, bbox_right, block_size, overlap)
    assert d.num_blocks_per_axis == expected


@pytest.mark.parametrize(
    "bbox_left, bbox_right, block_size, overlap",
    [
        ([0.0], [1.0], [0.5], [0.05]),
        ([0.0], [1.0], [0.5], [0.1]),
        ([0.0], [1.0], [1.0], [0.1]),
        ([0.0, 0.0], [1.0, 2.0], [0.5, 0.5], [0.05, 0.05]),
        ([0.0, 0.0], [1.0, 1.0], [0.6, 0.6], [0.2, 0.2]),
    ],
)
def test_total_block_count(bbox_left, bbox_right, block_size, overlap):
    d = make_decomp(bbox_left, bbox_right, block_size, overlap)
    expected = reduce(lambda a, b: a * b, d.num_blocks_per_axis)
    assert len(d.blocks) == expected


@pytest.mark.parametrize(
    "bbox_left, bbox_right, block_size, overlap",
    [
        ([0.0], [1.0], [0.5], [0.05]),
        ([0.0, 0.0], [1.0, 2.0], [0.5, 0.5], [0.05, 0.05]),
        ([0.0, 0.0], [1.0, 1.0], [0.6, 0.6], [0.2, 0.2]),
    ],
)
def test_blocks_per_axis_structure(bbox_left, bbox_right, block_size, overlap):
    d = make_decomp(bbox_left, bbox_right, block_size, overlap)
    n = len(bbox_left)
    if n == 1:
        assert len(d.blocks_per_axis) == 1
        assert len(d.blocks_per_axis[0]) == d.num_blocks_per_axis[0]
    else:
        outer = reduce(lambda a, b: a * b, d.num_blocks_per_axis[:-1])
        assert len(d.blocks_per_axis) == outer
        for axis_blocks in d.blocks_per_axis:
            assert len(axis_blocks) == d.num_blocks_per_axis[-1]


@pytest.mark.parametrize(
    "bbox_left, bbox_right, block_size, overlap",
    [
        ([0.0], [1.0], [0.5], [0.05]),
        ([0.0], [1.0], [0.5], [0.1]),
        ([0.0, 0.0], [1.0, 2.0], [0.5, 0.5], [0.05, 0.05]),
    ],
)
def test_block_corners_in_range(bbox_left, bbox_right, block_size, overlap):
    d = make_decomp(bbox_left, bbox_right, block_size, overlap)
    for block in d.blocks:
        for j, (lc, rc) in enumerate(
            zip(block.left_down_corner, block.right_up_corner)
        ):
            assert lc >= bbox_left[j] - 1e-9
            assert rc <= bbox_right[j] + block_size[j] + 1e-9
            assert lc < rc


@pytest.mark.parametrize(
    "bbox_left, bbox_right, block_size, overlap, points_per_block",
    [
        ([0.0], [1.0], [0.5], [0.05], 16),
        ([0.0, 0.0], [1.0, 1.0], [0.5, 0.5], [0.05, 0.05], 8),
    ],
)
def test_decomp_get_data_shape(
    bbox_left, bbox_right, block_size, overlap, points_per_block
):
    d = make_decomp(
        bbox_left, bbox_right, block_size, overlap, points_per_block=points_per_block
    )
    for block in d.blocks:
        data = block.get_data()
        assert data.shape == (points_per_block, len(bbox_left))


@pytest.mark.parametrize(
    "bbox_left, bbox_right, block_size, overlap, n_pts",
    [
        ([0.0], [1.0], [0.5], [0.05], 20),
        ([0.0, 0.0], [1.0, 1.0], [0.5, 0.5], [0.05, 0.05], 30),
    ],
)
def test_window_function_shape_and_positive(
    bbox_left, bbox_right, block_size, overlap, n_pts
):
    d = make_decomp(bbox_left, bbox_right, block_size, overlap)
    n_dim = len(bbox_left)
    x = torch.rand(n_pts, n_dim)
    for block in d.blocks:
        w = block.window_function(x)
        assert w.shape == (n_pts, 1)
        assert (w > 0).all()


@pytest.mark.parametrize(
    "bbox_left, bbox_right, block_size, overlap, n_pts",
    [
        ([0.0], [1.0], [0.5], [0.05], 15),
        ([0.0, 0.0], [1.0, 1.0], [0.5, 0.5], [0.05, 0.05], 25),
    ],
)
def test_batched_window_shape(bbox_left, bbox_right, block_size, overlap, n_pts):
    d = make_decomp(bbox_left, bbox_right, block_size, overlap)
    x = torch.rand(n_pts, len(bbox_left))
    w = d.batched_window(x)
    assert w.shape == (len(d.blocks), n_pts, 1)
    assert (w > 0).all()


@pytest.mark.parametrize(
    "bbox_left, bbox_right, block_size, overlap",
    [
        ([0.0], [1.0], [1.0], [0.1]),
        ([0.0, 0.0], [1.0, 1.0], [1.0, 1.0], [0.1, 0.1]),
    ],
)
def test_window_center_larger_than_edge(bbox_left, bbox_right, block_size, overlap):
    d = make_decomp(bbox_left, bbox_right, block_size, overlap)
    assert len(d.blocks) == 1
    block = d.blocks[0]
    lc = torch.tensor(block.left_down_corner)
    rc = torch.tensor(block.right_up_corner)
    center = ((lc + rc) / 2).unsqueeze(0)
    edge = lc.unsqueeze(0)
    w_center = block.window_function(center)
    w_edge = block.window_function(edge)
    assert (w_center >= w_edge).all()


@pytest.mark.parametrize(
    "bbox_left, bbox_right, block_size, overlap, scales, shifts",
    [
        ([0.0], [1.0], [0.5], [0.05], [2.0], [1.0]),
        ([0.0, 0.0], [1.0, 1.0], [0.5, 0.5], [0.05, 0.05], [3.0, 0.5], [-1.0, 2.0]),
    ],
)
def test_block_denorm_scale_shift_stored(
    bbox_left, bbox_right, block_size, overlap, scales, shifts
):
    d = make_decomp(
        bbox_left,
        bbox_right,
        block_size,
        overlap,
        block_scales=scales,
        block_shift=shifts,
    )
    for block in d.blocks:
        assert block.out_denorm_scale == scales
        assert block.out_denorm_shift == shifts


@pytest.mark.parametrize(
    "x_coord, expected_b0, expected_b1",
    [
        (0.5, 0.8176, 0.1824),
        (0.25, 0.9525, 0.0001),
        (0.75, 0.0025, 0.9951),
        (-1.0, 0.0000, 0.0000),
    ],
)
def test_batched_window_specific_values_1d(x_coord, expected_b0, expected_b1):
    d = make_decomp(
        bbox_left=[0.0],
        bbox_right=[1.0],
        block_size=[0.7],
        overlap=[0.3],
        omega=30.0,
    )
    d.prepare_batched()
    d.blocks = sorted(d.blocks, key=lambda b: b.left_down_corner[0])
    assert len(d.blocks) == 2

    # (N_pts, d) -> (1, 1)
    x = torch.tensor([[x_coord]], dtype=torch.float32, device=d.device)

    w = d.batched_window(x)  # (N_blocks, N_pts, 1)

    actual_b0 = w[0, 0, 0].item()
    actual_b1 = w[1, 0, 0].item()

    assert actual_b0 == pytest.approx(expected_b0, abs=1e-4)
    assert actual_b1 == pytest.approx(expected_b1, abs=1e-4)


@pytest.mark.parametrize(
    "bbox_left, bbox_right, block_size, overlap, margin",
    [
        ([0.0], [1.0], [0.5], [0.1], 0.15),
        ([0.0], [1.0], [0.3], [0.08], 0.1),
        ([0.0, 0.0], [1.0, 1.0], [0.5, 0.5], [0.1, 0.1], 0.15),
    ],
)
def test_batched_window_partition_of_unity(
    bbox_left, bbox_right, block_size, overlap, margin
):
    d = make_decomp(bbox_left, bbox_right, block_size, overlap)
    n = len(bbox_left)
    lo = torch.tensor([l + margin for l in bbox_left])
    hi = torch.tensor([r - margin for r in bbox_right])
    x = torch.rand(500, n) * (hi - lo) + lo
    w = d.batched_window(x)  # (N_blocks, N_pts, 1)
    w_sum = w.sum(dim=0).squeeze(-1)
    assert torch.allclose(w_sum, torch.ones_like(w_sum), atol=1e-3)
