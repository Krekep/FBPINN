import torch

from src.geometry.base_decomposition import RectangleDomain, Block, BaseDecomposition


class DecompositionND(BaseDecomposition):
    """
    Class for N-dimensional domain decomposition
    """

    def __init__(
        self,
        domain: RectangleDomain,
        bbox_left: list[float],
        bbox_right: list[float],
        overlap: list[float],
        block_size: list[float],
        block_scales: list[float],
        block_shift: list[float],
        points_per_block: int = 100,
        device: str = ""
    ) -> None:
        """
        Parameters
        ----------
        domain: RectangleDomain
        overlap: list[float]
            overlaps per dimension
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
        device: str
            `cpu`/`cuda`/etc.
        """
        super().__init__(
            overlap,
            block_size,
            block_scales,
            block_shift,
            points_per_block,
            device
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
                last = last - overlap[i] + block_size[i]
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
                    + (self.block_size[j] - self.overlap[j]) * curr_id
                )
                rc = (
                    self.bbox_left[j]
                    + self.block_size[j]
                    + (self.block_size[j] - self.overlap[j]) * curr_id
                )

                left_corner.append(lc)
                right_corner.append(rc)
            for i in range(self.num_blocks_per_axis[current_ax]):
                left_corner[-1] = (  # in current axis we move only last boundary of block
                    self.bbox_left[-1]
                    + (self.block_size[-1] - self.overlap[-1]) * i
                )
                right_corner[-1] = (
                    self.bbox_left[-1]
                    + self.block_size[-1]
                    + (self.block_size[-1] - self.overlap[-1]) * i
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
                sampler = (
                    lambda n: torch.rand([n] + [len(lc_list)], device=self.device)
                              * (torch.tensor(data_rc, device=self.device)
                                 - torch.tensor(data_lc, device=self.device))
                              + torch.tensor(data_lc, device=self.device)
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
                )
                self.blocks.append(block)
                self.blocks_per_axis[-1].append(block)
        else:
            while current_idx[current_ax] < self.num_blocks_per_axis[current_ax]:
                self.blocks_per_axis.append([])
                self._build_decomposition(
                    current_ax + 1, current_idx, n
                )
                current_idx[current_ax] += 1
            current_idx[current_ax] = 0


if __name__ == "__main__":
    domain = RectangleDomain([0, 0], [4, 5])
    dec = DecompositionND(
        domain=domain,
        overlap=[1, 1],
        block_size=[2, 2],
        block_scales=[1, 2],
        block_shift=[0, 0.5],
        points_per_block=100
    )
    print(dec.blocks_per_axis)
    print(dec.num_blocks_per_axis)
    print(dec.blocks)

    domain = RectangleDomain([0], [4])
    dec = DecompositionND(
        domain=domain,
        overlap=[1],
        block_size=[2],
        block_scales=[1],
        block_shift=[1],
        points_per_block=100
    )
    print(dec.blocks_per_axis)
    print(dec.num_blocks_per_axis)
    print(dec.blocks)
