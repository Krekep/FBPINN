from typing import Callable

import numpy as np
import numpy.typing as npt
import torch


class RectangleDomain:
    left_down_corner: list[float]
    right_up_corner: list[float]

    def __init__(
            self, left_corner: list[float], right_corner: list[float]
    ) -> None:
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
        idx = torch.randperm(pool_size, device=self._pool.device)[:n]
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
        data_unnorm = data * self.out_denorm_scale + self.out_denorm_shift  # output unnormalisation
        return data_unnorm

    def set_losses(
            self,
            losses: list[tuple[Callable, list[tuple[float, float]]]]
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

    def refresh_pool(self, size: int = 10000):
        if self.sampler is not None:
            pts = self.sampler(size)
            self._pool = torch.tensor(
                pts, dtype=torch.float32, device=self._pool.device
            )
        else:
            self._pool = (torch.rand(self.data_shape, device=self.vmin.device) * (self.vmax - self.vmin) + self.vmin)

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
            pool_size: int = 10000
    ) -> None:
        self.left_down_corner: list = left_corner
        self.right_up_corner: list = right_corner
        self.window_function: Callable[
            [npt.NDArray[np.float64]],
            npt.NDArray[np.float64]
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
        if self.sampler is not None:
            pts = self.sampler(pool_size)
            self._pool = torch.tensor(
                pts, dtype=torch.float32, device=device
            )
        else:
            self._pool = (torch.rand(self.data_shape, device=self.vmin.device) * (self.vmax - self.vmin) + self.vmin)


class BaseDecomposition:
    """
    Base class for domain decomposition
    """

    def __init__(
            self,
            overlap: list[float] | tuple[float, ...],
            block_size: list[float] | tuple[float, ...],
            block_scales: list[float],
            block_shift: list[float],
            points_per_block: int = 100,
            device: str = ""
    ):
        """
        Parameters
        ----------
        overlap: list[float]
            overlaps per dimension
        block_size: list[float]
            size of blocks per dimension
        block_scales: list[float]
            Unnormalization multiplier for blocks per dimension
        block_shift: list[float]
            Unnormalization term for blocks per dimension
        points_per_block: int
        device: str
            `cpu`/`cuda`/etc.
        """
        if device == "":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device

        self.overlap = torch.tensor(overlap, dtype=torch.float32)
        self.block_size = block_size
        self.block_scales = block_scales
        self.block_shift = block_shift
        self.points_per_block = points_per_block
        self.blocks = []
        self.num_blocks_per_axis: list[int] = []
        self.blocks_per_axis: list[list[Block]] = []

    def get_window_function(
            self, left_corner, right_corner, omega: float = 30
    ) -> Callable[[torch.Tensor], torch.Tensor]:
        def sigmoid(x: torch.Tensor) -> torch.Tensor:
            x_clipped = torch.clip(x, -50.0, 50.0)
            return torch.maximum(1 / (1 + torch.exp(-x_clipped)), torch.tensor(1e-10))

        left_corner_tf = torch.tensor(
            left_corner, dtype=torch.float32, device=self.device, requires_grad=False
        )
        right_corner_tf = torch.tensor(
            right_corner, dtype=torch.float32, device=self.device, requires_grad=False
        )
        omega = torch.tensor(omega, dtype=torch.float32)

        def window_function(x_in: torch.Tensor) -> torch.Tensor:
            """$ w_i(x) = \prod_j^d (\phi((x^j - a_i^j) / \sigma_i^j) \phi((b_i^j - x^j) / \siqma_i^j) ) $"""
            x = x_in  # x (n, d)
            a = left_corner_tf + self.overlap / 2.0
            b = right_corner_tf - self.overlap / 2.0
            left = sigmoid((x - a) * omega)  # (n, d)
            right = sigmoid((b - x) * omega)  # (n, d)

            result = torch.prod(left * right, dim=1, keepdims=True)  # (n, 1)
            return result

        return window_function

    def build_decomposition(self) -> None:
        """
        Method for decompose domain area.
        Must be implemented in inheritors
        """
        raise NotImplementedError

    def prepare_batched(self) -> None:
        """
        Precompute stacked geometry tensors for batched window evaluation.
        Call this once after decomposition is built and device is known.
        """
        lcs = [torch.tensor(b.left_down_corner, dtype=torch.float32, device=self.device, requires_grad=False)
               for b in self.blocks]
        rcs = [torch.tensor(b.right_up_corner, dtype=torch.float32, device=self.device, requires_grad=False)
               for b in self.blocks]

        self._all_lc = torch.stack(lcs)  # (N_blocks, d)
        self._all_rc = torch.stack(rcs)  # (N_blocks, d)
        ov = torch.tensor(self.overlap, dtype=torch.float32, device=self.device, requires_grad=False)
        self._all_ov = ov.unsqueeze(0).expand(len(self.blocks), -1)  # (N_blocks, d)

    def batched_window(self, x: torch.Tensor, omega: float = 30.0) -> torch.Tensor:
        """
        Compute window functions for all blocks simultaneously.

        Parameters
        ----------
        x : torch.Tensor
            Input points, shape (N_pts, d)
        omega : float
            Sharpness of sigmoid transition

        Returns
        -------
        torch.Tensor
            Window values, shape (N_blocks, N_pts, 1)
        """
        def sigmoid(x: torch.Tensor) -> torch.Tensor:
            return torch.clamp(torch.sigmoid(x.clamp(-50.0, 50.0)), min=1e-10)
        x_exp = x.unsqueeze(0)  # (1, N_pts, d)
        lc = self._all_lc.unsqueeze(1)  # (N_blocks, 1, d)
        rc = self._all_rc.unsqueeze(1)  # (N_blocks, 1, d)
        ov = self._all_ov.unsqueeze(1)  # (N_blocks, 1, d)

        a = lc + ov / 2.0
        b = rc - ov / 2.0
        left = sigmoid((x_exp - a) * omega)  # (N_blocks, N_pts, d)
        right = sigmoid((b - x_exp) * omega)

        return (left * right).prod(dim=-1, keepdim=True)  # (N_blocks, N_pts, 1)
        # log_left = -torch.nn.functional.softplus(-(x_exp - a) * omega)  # (N_blocks, N_pts, d)
        # log_right = -torch.nn.functional.softplus(-(b - x_exp) * omega)
        #
        # log_w = (log_left + log_right).sum(dim=-1, keepdim=True)  # (N_blocks, N_pts, 1)
        # return torch.exp(log_w)
