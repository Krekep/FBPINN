from typing import Optional

import torch

from geofbpinn.geometry.base_decomposition import (
    RectangleDomain,
    Block,
    BaseDecomposition,
)


class DecompositionND(BaseDecomposition):
    """
    Class for N-dimensional domain decomposition
    """

    def __init__(
        self,
        domain: RectangleDomain,
        bbox_left: list[float],
        bbox_right: list[float],
        block_size: list[float],
        block_scales: list[float],
        block_shift: list[float],
        points_per_block: int = 100,
        overlap: Optional[list[float]] = None,
        kappa: float = 0.3,
        omega: Optional[float] = None,
        eps: float = 1e-4,
        device: str = "",
    ) -> None:
        """
        Parameters
        ----------
        domain: RectangleDomain
        bbox_left: list[float]
            Left lower corner of bounding n-d rectangle
        bbox_right: list[float]
            Right upper corner of bounding n-d rectangle
        block_size: list[float]
            size of blocks per dimension
        block_scales: list[float]
            Unnormalization multiplier for blocks per dimension
        block_shift: list[float]
            Unnormalization term for blocks per dimension
        points_per_block: int
        overlap: Optional[list[float]]
            overlaps per dimension. Mutually exclusive with `kappa`.
        kappa: float
            Overlap ratio (delta / B). Mutually exclusive with `overlap`.
        omega: Optional[float]
            Sharpness of sigmoid transition. If None, auto-calculated.
        eps: float
            Numerical threshold for omega calculation.
        device: str
            `cpu`/`cuda`/etc.
        """
        super().__init__(
            block_size=block_size,
            block_scales=block_scales,
            block_shift=block_shift,
            points_per_block=points_per_block,
            overlap=overlap,
            kappa=kappa,
            omega=omega,
            eps=eps,
            device=device,
        )
        self.domain = domain
        self.bbox_left = bbox_left
        self.bbox_right = bbox_right
        self.points_per_block = points_per_block
        for i in range(len(bbox_left)):
            number_of_blocks = 1
            last = bbox_left[i] + block_size[i]
            while last < bbox_right[i]:
                number_of_blocks += 1
                last = last - self.overlap[i].item() + block_size[i]
            self.num_blocks_per_axis.append(number_of_blocks)

        self.build_decomposition()
        self.prepare_batched()

    def build_decomposition(self) -> None:
        n = len(self.domain.left_down_corner)
        self._build_decomposition(0, [0] * n, n)

    def _build_decomposition(
        self,
        current_ax: int,
        current_idx: list[int],
        n: int,
    ) -> None:
        """
        Decompose N-dimensional domain to list of sub-blocks

        Parameters
        ----------
        current_ax: int
            Dimension to process (from 0 to N-1)
        current_idx: list[int]
            How much we already process in respective axis
        n: int
            Number of domain dimensions
        """
        if current_ax == n - 1:
            if len(self.blocks_per_axis) == 0:  # 1D case
                self.blocks_per_axis.append([])

            left_corner = []
            right_corner = []
            for j in range(n):
                curr_id = current_idx[j]  # offset in axis
                lc = (
                    self.bbox_left[j]
                    + (self.block_size[j] - self.overlap[j].item()) * curr_id
                )
                rc = (
                    self.bbox_left[j]
                    + self.block_size[j]
                    + (self.block_size[j] - self.overlap[j].item()) * curr_id
                )

                left_corner.append(lc)
                right_corner.append(rc)
            for i in range(self.num_blocks_per_axis[current_ax]):
                left_corner[
                    -1
                ] = (  # in current axis we move only last boundary of block
                    self.bbox_left[-1]
                    + (self.block_size[-1] - self.overlap[-1].item()) * i
                )
                right_corner[-1] = (
                    self.bbox_left[-1]
                    + self.block_size[-1]
                    + (self.block_size[-1] - self.overlap[-1].item()) * i
                )
                lc_list = left_corner.copy()
                rc_list = right_corner.copy()
                window_function = self.get_window_function(lc_list, rc_list)
                data_lc = [
                    max(a, b) for a, b in zip(lc_list, self.domain.left_down_corner)
                ]
                data_rc = [
                    min(a, b) for a, b in zip(rc_list, self.domain.right_up_corner)
                ]
                size = [self.points_per_block] + [len(lc_list)]
                sampler = lambda n, lc=tuple(lc_list), rc=tuple(data_rc), lc_val=tuple(
                    data_lc
                ): torch.rand([n] + [len(lc)], device=self.device) * (
                    torch.tensor(rc, device=self.device)
                    - torch.tensor(lc_val, device=self.device)
                ) + torch.tensor(
                    lc_val, device=self.device
                )
                block = Block(
                    data_shape=size,
                    left_corner=lc_list,
                    right_corner=rc_list,
                    window_function=window_function,
                    out_denorm_scale=self.block_scales,
                    out_denorm_shift=self.block_shift,
                    sampler=sampler,
                    device=self.device,
                    pool_size=max(10_000, self.points_per_block),
                )
                self.blocks.append(block)
                self.blocks_per_axis[-1].append(block)
        else:
            while current_idx[current_ax] < self.num_blocks_per_axis[current_ax]:
                self.blocks_per_axis.append([])
                self._build_decomposition(current_ax + 1, current_idx, n)
                current_idx[current_ax] += 1
            current_idx[current_ax] = 0

    def remove_redundant_blocks(self, **kwargs) -> None:
        pass

    def get_config(self) -> dict:
        cfg = super().get_config()
        cfg.update(
            blocks_per_axis=self.blocks_per_axis,
            domain=self.domain,
            bbox_left=self.bbox_left,
            bbox_right=self.bbox_right,
            num_blocks_per_axis=self.num_blocks_per_axis,
        )
        return cfg
