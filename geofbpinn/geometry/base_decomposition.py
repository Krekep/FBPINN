import math
from typing import Callable, Optional, Tuple

import numpy as np
import numpy.typing as npt
import torch

from geofbpinn.geometry.auto_params import resolve_params
from geofbpinn.geometry.window_functions import get_window_fn, prod_last_dim


class RectangleDomain:
    left_down_corner: list[float]
    right_up_corner: list[float]

    def __init__(self, left_corner: list[float], right_corner: list[float]) -> None:
        self.left_down_corner = left_corner
        self.right_up_corner = right_corner


class Block:
    """
    Single block of decomposition
    """

    __slots__ = [
        "window_function",
        "left_down_corner",
        "right_up_corner",
        "vmax",
        "vmin",
        "losses",
        "data_shape",
        "data_size",
        "out_denorm_scale",
        "out_denorm_shift",
        "sampler",
        "_pool",
        "_pool_size",
    ]

    def get_data(self) -> torch.Tensor:
        """
        Generate data for block with points from `vmin` to `vmax`.
        If `self.sampler` is not None, then generate data with sampler

        Returns
        -------
        data: torch.Tensor
        """
        pool_size = self._pool.shape[0]
        n = self.data_shape[0]
        # idx = torch.randperm(pool_size, device=self._pool.device)[:n]
        idx = torch.randint(0, pool_size, (n,), device=self._pool.device)
        return self._pool[idx]
        # if self.sampler is not None:
        #     pts = self.sampler(self.data_shape[0])
        #     data = torch.tensor(pts, dtype=torch.float32, device=self.vmin.device)
        # else:
        #     data = (torch.rand(self.data_shape, device=self.vmin.device) * (self.vmax - self.vmin) + self.vmin)
        # return data

    def normalization(self, data: torch.Tensor) -> torch.Tensor:
        data_norm = (
            2.0 * ((data - self.vmin) / (self.vmax - self.vmin)) - 1.0
        )  # subdomain normalisation --- R -> [-1; 1]
        return data_norm

    def unnormalization(self, data: torch.Tensor) -> torch.Tensor:
        data_unnorm = (
            data * self.out_denorm_scale + self.out_denorm_shift
        )  # output unnormalisation
        return data_unnorm

    def set_losses(
        self, losses: list[tuple[Callable, list[tuple[float, float]]]]
    ) -> None:
        """
        Set physic losses, that acts on area of this block

        Parameters
        ----------
        losses: list[tuple[Callable, list[tuple[float, float]]]]
            List of physical loss functions with their corresponding regions, specified as list of vertices
        """
        res = []
        left_down_corner = self.left_down_corner
        right_up_corner = self.right_up_corner
        for loss, var_bound in losses:
            fl = True
            for lc, var_b, rc in zip(left_down_corner, var_bound, right_up_corner):
                if var_b is not None:
                    for b in var_b:
                        if b < lc or rc < b:
                            fl = False
                            break
            if fl:
                res.append(loss)
        self.losses = res

    def get_losses(self):
        return self.losses

    def refresh_pool(self):
        if self.sampler is not None:
            pts = self.sampler(self._pool_size)
            self._pool = torch.tensor(
                pts, dtype=torch.float32, device=self._pool.device
            )
        else:
            self._pool = (
                torch.rand(self.data_shape, device=self.vmin.device)
                * (self.vmax - self.vmin)
                + self.vmin
            )

    def __init__(
        self,
        data_shape,
        left_corner,
        right_corner,
        window_function,
        out_denorm_scale,
        out_denorm_shift,
        sampler,
        device,
        pool_size: int = 10000,
    ) -> None:
        self.left_down_corner: list = left_corner
        self.right_up_corner: list = right_corner
        self.window_function: Callable[
            [npt.NDArray[np.float64]], npt.NDArray[np.float64]
        ] = window_function
        self.data_shape = data_shape

        self.vmax: torch.Tensor = torch.tensor(
            right_corner, dtype=torch.float32, device=device
        )
        self.vmin: torch.Tensor = torch.tensor(
            left_corner, dtype=torch.float32, device=device
        )
        self.out_denorm_scale = out_denorm_scale
        self.out_denorm_shift = out_denorm_shift
        self.sampler = sampler
        self._pool_size = pool_size
        if self.sampler is not None:
            pts = self.sampler(pool_size)
            self._pool = torch.tensor(pts, dtype=torch.float32, device=device)
        else:
            self._pool = (
                torch.rand(self.data_shape, device=self.vmin.device)
                * (self.vmax - self.vmin)
                + self.vmin
            )


class BaseDecomposition:
    """
    Base class for domain decomposition
    """

    def __init__(
        self,
        block_size: list[float] | tuple[float, ...],
        block_scales: list[float],
        block_shift: list[float],
        window_fn_type: str = "sigmoid",
        points_per_block: int = 100,
        overlap: Optional[list[float] | tuple[float, ...]] = None,
        kappa: float = 0.2,
        omega: Optional[float] = None,
        eps: float = 1e-4,
        device: str = "",
    ):
        """
        Parameters
        ----------
        block_size: list[float]
            size of blocks per dimension
        block_scales: list[float]
            Unnormalization multiplier for blocks per dimension
        block_shift: list[float]
            Unnormalization term for blocks per dimension
        window_fn_type: str
            Name of window function. Look at `networks.utils.get_classes.get_window_fn`
        points_per_block: int
        overlap: Optional[list[float]]
            overlaps per dimension. Mutually exclusive with `kappa`.
        kappa: float
            Overlap ratio (delta / B). Mutually exclusive with `overlap`.
        omega: Optional[float]
            Sharpness of sigmoid transition. If None, computed like
            omega = 2 * ln(1 / eps) / (kappa * B).
        eps: float
            Numerical threshold for omega calculation
        device: str
            `cpu`/`cuda`/etc.
        """
        if device == "":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device

        overlap, kappa, omega = resolve_params(
            block_size=tuple(block_size),
            eps=eps,
            overlap=overlap,
            kappa=kappa,
            omega=omega,
        )

        self.overlap = torch.tensor(overlap, dtype=torch.float32, device=device)
        self._kappa = kappa
        self.omega = omega
        self.eps = eps
        self.window_function_type = window_fn_type
        self.window_function = get_window_fn(window_fn_type)(self.overlap, eps, omega)
        self.block_size = block_size
        self.points_per_block = points_per_block
        self.blocks = []
        self.num_blocks_per_axis: list[int] = []
        self.blocks_per_axis: list[list[Block]] = []

        if isinstance(block_scales, list):
            block_scales = torch.tensor(
                block_scales, dtype=torch.float32, device=device
            )
        if isinstance(block_shift, list):
            block_shift = torch.tensor(block_shift, dtype=torch.float32, device=device)
        self.block_scales = block_scales
        self.block_shift = block_shift

    def get_window_function(
        self, left_corner, right_corner
    ) -> Callable[[torch.Tensor], torch.Tensor]:
        left_corner_tf = torch.tensor(
            left_corner, dtype=torch.float32, device=self.device, requires_grad=False
        )
        right_corner_tf = torch.tensor(
            right_corner, dtype=torch.float32, device=self.device, requires_grad=False
        )

        def window_function(x_in: torch.Tensor) -> torch.Tensor:
            """$ w_i(x) = \prod_j^d (\text{edge}(x^j, a_i^j) \cdot \text{edge}(b_i^j, x^j)) $
            Edge shape is delegated to `self.window_fn`"""
            x = x_in  # x (n, d)
            a = left_corner_tf + self.overlap / 2.0
            b = right_corner_tf - self.overlap / 2.0
            per_dim = self.window_function(x, a, b)  # (n, d)

            result = torch.prod(per_dim, dim=1, keepdims=True)  # (n, 1)
            # result = prod_last_dim(per_dim)  # (n, 1)
            return result

        return window_function

    def build_decomposition(self) -> None:
        """
        Method for decompose domain area.
        Must be implemented in inheritors
        """
        raise NotImplementedError

    def remove_redundant_blocks(self, **kwargs) -> None:
        """
        Method for remove unnecessary blocks.
        Must be implemented in inheritors
        """
        raise NotImplementedError

    def prepare_batched(self) -> None:
        """
        Precompute stacked geometry tensors for batched window evaluation.
        Call this once after decomposition is built and device is known.
        """
        lcs = [
            torch.tensor(
                b.left_down_corner,
                dtype=torch.float32,
                device=self.device,
                requires_grad=False,
            )
            for b in self.blocks
        ]
        rcs = [
            torch.tensor(
                b.right_up_corner,
                dtype=torch.float32,
                device=self.device,
                requires_grad=False,
            )
            for b in self.blocks
        ]

        self._all_lc = torch.stack(lcs).unsqueeze(1)  # (N_blocks, 1, d)
        self._all_rc = torch.stack(rcs).unsqueeze(1)  # (N_blocks, 1, d)
        ov = torch.tensor(
            self.overlap, dtype=torch.float32, device=self.device, requires_grad=False
        )
        self._all_ov = (
            ov.unsqueeze(0).expand(len(self.blocks), -1).unsqueeze(1)
        )  # (N_blocks, 1, d)
        self._all_a = self._all_lc + self._all_ov / 2.0
        self._all_b = self._all_rc - self._all_ov / 2.0

    def batched_window(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute window functions for all blocks simultaneously.

        Parameters
        ----------
        x : torch.Tensor
            Input points, shape (N_pts, d)

        Returns
        -------
        torch.Tensor
            Window values, shape (N_blocks, N_pts, 1)
        """
        x_exp = x.unsqueeze(0)  # (1, N_pts, d)

        per_dim = self.window_function(
            x_exp, self._all_a, self._all_b
        )  # (N_blocks, N_pts, d)
        # return prod_last_dim(per_dim)  # (N_blocks, N_pts, 1)
        return torch.prod(per_dim, dim=-1, keepdims=True)  # (N_blocks, N_pts, 1)

    def get_config(self) -> dict:
        return dict(
            overlap=self.overlap,
            kappa=self._kappa,
            block_size=self.block_size,
            block_scales=self.block_scales,
            block_shift=self.block_shift,
            points_per_block=self.points_per_block,
            omega=self.omega,
            eps=self.eps,
            window_fn_type=self.window_function_type,
        )
