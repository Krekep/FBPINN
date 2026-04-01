from typing import Tuple, List, Callable, Optional
import numpy as np

from src.geometry import Block
from src.geometry.base_decomposition import BaseDecomposition
from src.geometry.geometry import (
    sutherland_hodgman_clip,
    polygon_area,
    point_in_poly,
    sample_points_in_polygon_with_holes
)
from enum import Enum


class BlockKind(Enum):
    INTERIOR: str = "interior"
    BOUNDARY: str = "boundary"


class PolygonBlock(Block):
    """
    Block with some addition information for 2D case with holes
    """
    __slots__ = Block.__slots__ + [
        "kind",
        "area_ratio",
        "clipped_polygon",
        "clipped_holes",
        "boundary_points",
    ]

    def __init__(
            self,
            block_args: tuple,
            kind: BlockKind,
            area_ratio: float,
            clipped_polygon: list[tuple[float, float]],
            clipped_holes: list[list[tuple[float, float]]],
    ):
        """
        Parameters
        ----------
        block_args: tuple
            Args for `Block` class
        kind: BlockKind
            Label for visualization
        area_ratio: float
            Ratio of block area to the area of intersection block with polygon
        clipped_polygon: list[tuple[float, float]]
            Intersection block with polygon
        clipped_holes: list[list[tuple[float, float]]]
            Intersection block with holes
        """
        super().__init__(*block_args)
        self.kind = kind
        self.area_ratio = area_ratio
        self.clipped_polygon = clipped_polygon
        self.clipped_holes = clipped_holes


def polygon_points_sampler(
        clipped_polygon: list[tuple[float, float]],
        clipped_holes: list[list[tuple[float, float]]]
) -> Callable[[int], np.ndarray[tuple[int, int], np.dtype[np.float32]]]:
    """
    Sampler function for block data. Generate points in polygon, but outside the holes

    Parameters
    ----------
    clipped_polygon: list[tuple[float, float]]
        Polygon of block
    clipped_holes: list[list[tuple[float, float]]]
        Holes inside of polygon

    Returns
    -------
    sampler: Callable[[int], np.ndarray[tuple[int, int], np.dtype[np.float32]]]
        Function that takes `n` and returns `n` points
    """
    def sampler(n: int) -> np.ndarray[tuple[int, int], np.dtype[np.float32]]:
        pts = sample_points_in_polygon_with_holes(clipped_polygon, clipped_holes, n)
        if pts is None or len(pts) == 0:
            pts = np.zeros((n, 2), dtype=np.float32)
        return pts
    return sampler


class Decomposition2DPolygon(BaseDecomposition):
    def __init__(
            self,
            polygon_vertices: List[Tuple[float, float]],
            bbox_left: Tuple[float, float],
            bbox_right: Tuple[float, float],
            block_scales: list[float],
            block_shift: list[float],
            block_size: Tuple[float, float],
            overlap: Tuple[float, float],
            points_per_block: int = 200,
            holes: Optional[List[List[Tuple[float, float]]]] = None,
            eps_full: float = 1e-6,
            device: str = ""
    ) -> None:
        """
        Parameters
        ----------
        polygon_vertices: List[Tuple[float, float]]
            List of (x,y) boundary points of domain in order (closed not required)
        bbox_left: Tuple[float, float]
            Left lower corner of bounding rectangle
        bbox_right: Tuple[float, float]
            Right upper corner of bounding rectangle
        overlap: list[float]
            overlaps per dimension
        block_size: list[float]
            size of blocks per dimension
        block_scales: list[float]
            Unnormalization multiplier for blocks per dimension
        block_shift: list[float]
            Unnormalization term for blocks per dimension
        points_per_block: int
        holes: Optional[List[List[Tuple[float, float]]]]
            List of holes in domain
        eps_full: float
            Accuracy of determining whether block is boundary
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
        self.polygon = polygon_vertices
        self.holes = holes if holes is not None else []
        self.bbox_left = bbox_left
        self.bbox_right = bbox_right
        self.eps_full = eps_full
        self.num_blocks_per_axis: List[int] = [0, 0]

        self.build_decomposition()
        self.prepare_batched()

    def build_decomposition(self):
        """
        Decompose 2d area with holes.
        """
        xmin, ymin = self.bbox_left
        xmax, ymax = self.bbox_right
        step_x = self.block_size[0] - self.overlap[0].item()
        step_y = self.block_size[1] - self.overlap[1].item()
        wrappers: List[PolygonBlock] = []

        x = xmin
        while x < xmax - 1e-12:
            self.num_blocks_per_axis[0] += 1
            self.blocks_per_axis.append([])
            blocks_per_y = 0
            y = ymin
            while y < ymax - 1e-12:
                rect = (x, y, x + self.block_size[0], y + self.block_size[1])

                clipped_outer = sutherland_hodgman_clip(self.polygon, rect)
                if clipped_outer and len(clipped_outer) >= 3:
                    clipped_holes = []
                    for hole in self.holes:
                        ch = sutherland_hodgman_clip(hole, rect)
                        if ch and len(ch) >= 3:
                            clipped_holes.append(ch)

                    rect_area = self.block_size[0] * self.block_size[1]
                    inter_area = polygon_area(clipped_outer)
                    area_ratio = inter_area / rect_area if rect_area > 0 else 0.0
                    kind = BlockKind.INTERIOR if area_ratio >= 1.0 - self.eps_full and not clipped_holes else BlockKind.BOUNDARY

                    pts = sample_points_in_polygon_with_holes(clipped_outer, clipped_holes, self.points_per_block)

                    window_fn = self.get_window_function((rect[0], rect[1]), (rect[2], rect[3]))
                    if pts is None:
                        pts = np.zeros((self.points_per_block, 2), dtype=float)
                    sampler = polygon_points_sampler(clipped_outer,
                                                     clipped_holes) if kind == BlockKind.BOUNDARY else None
                    block_args = (
                        pts.shape,
                        [rect[0], rect[1]],
                        [rect[2], rect[3]],
                        window_fn,
                        self.block_scales,
                        self.block_shift,
                        sampler,
                        self.device,
                    )
                    wrapper = PolygonBlock(
                        block_args=block_args,
                        kind=kind,
                        area_ratio=area_ratio,
                        clipped_polygon=clipped_outer,
                        clipped_holes=clipped_holes,
                    )
                    wrappers.append(wrapper)
                    blocks_per_y += 1
                    self.blocks_per_axis[-1].append(wrapper)
                y += step_y
            x += step_x
            self.num_blocks_per_axis[1] = max(blocks_per_y, self.num_blocks_per_axis[1])
            if len(self.blocks_per_axis[-1]) == 0:
                self.blocks_per_axis.pop()

        self.blocks = wrappers

    def remove_redundant_blocks(
            self,
            samples_per_block: int = 400,
            tol: float = 0.01,
            verbose: bool = True,
    ) -> None:
        """
        Remove blocks that do not have their own "unique" area in the domain.

        A block is considered redundant if a sufficiently large fraction of its
        interior (sampled points) is covered by the union of other blocks that
        overlap its bounding box. Redundant blocks are removed from the
        decomposition.

        Parameters
        ----------
        samples_per_block : int
            Number of points to sample from each block's polygon (with holes) to
            estimate coverage.
        tol : float
            Tolerance for the uncovered fraction. Blocks with uncovered fraction
            (1 - covered_ratio) <= tol are removed.
        verbose : bool
            If True, print removal information for each block and a summary.
        """
        blocks = self.blocks
        n = len(blocks)

        bboxes = np.array([
            (b.left_down_corner[0], b.left_down_corner[1],
             b.right_up_corner[0], b.right_up_corner[1])
            for b in blocks
        ], dtype=float)  # (n, 4)

        areas = (
                np.array([b.area_ratio for b in blocks]) *
                (bboxes[:, 2] - bboxes[:, 0]) *
                (bboxes[:, 3] - bboxes[:, 1])
        )  # (n,)

        order = np.argsort(areas).tolist()
        to_keep = [True] * n

        for idx in order:
            if not to_keep[idx]:
                continue
            b = blocks[idx]
            if not b.clipped_polygon or len(b.clipped_polygon) < 3:
                to_keep[idx] = False
                continue

            xi0, yi0, xi1, yi1 = bboxes[idx]
            keep_mask = np.array(to_keep)
            keep_mask[idx] = False
            overlap_mask = (  # overlap with keeping blocks
                    keep_mask &
                    (bboxes[:, 2] > xi0) & (bboxes[:, 0] < xi1) &
                    (bboxes[:, 3] > yi0) & (bboxes[:, 1] < yi1)
            )
            potential = np.where(overlap_mask)[0].tolist()
            if not potential:
                continue

            pts = sample_points_in_polygon_with_holes(
                b.clipped_polygon, b.clipped_holes, samples_per_block
            )
            if pts is None or pts.shape[0] == 0:
                to_keep[idx] = False
                continue

            px, py = pts[:, 0], pts[:, 1]

            covered = np.zeros(len(pts), dtype=bool)
            for j in potential:
                other = blocks[j]
                poly_arr = np.asarray(other.clipped_polygon, dtype=float)
                in_poly = point_in_poly(px, py, poly_arr)  # (P,)

                # Since holes are global for all block and
                # `sample_points_in_polygon_with_holes` can not generate points in hole
                # this check has no sense
                in_hole = np.zeros(len(pts), dtype=bool)
                # for ch in other.clipped_holes:
                #     in_hole |= point_in_poly(px, py, np.asarray(ch, dtype=float))

                covered |= (in_poly & ~in_hole)

                if covered.all():
                    break

            frac_uncovered = 1.0 - covered.mean()
            if frac_uncovered <= tol:
                to_keep[idx] = False
                if verbose:
                    print(f"Removing block {idx}: uncovered fraction {frac_uncovered:.4f} <= tol {tol}")
            else:
                if verbose:
                    print(f"Keeping block {idx}: uncovered fraction {frac_uncovered:.4f} > tol {tol}")

        new_blocks = [b for k, b in zip(to_keep, blocks) if k]
        removed = len(blocks) - len(new_blocks)
        self.blocks = new_blocks
        self.blocks_per_axis = [
            [w for w in col if w in set(new_blocks)]
            for col in self.blocks_per_axis
            if any(w in set(new_blocks) for w in col)
        ]

        if verbose:
            print(f"Removed {removed} redundant blocks, remaining {len(self.blocks)}")
