from typing import Tuple, List, Optional

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.patches import Rectangle, Polygon as MplPolygon
from matplotlib.lines import Line2D

from geofbpinn.geometry import Block
from geofbpinn.geometry.polygon_decomposition import PolygonBlock


def _extract_data_from_block_obj(block_obj: Block) -> np.ndarray:
    return np.array(block_obj.get_data().detach().cpu().numpy(), dtype=np.float32)


def plot_decomposition2d(
    decomposition: list[PolygonBlock],
    polygon_vertices: list[tuple[float, float]],
    holes: Optional[List[List[Tuple[float, float]]]] = None,
    figsize: tuple[int, int] = (11, 4),
    savepath: Optional[str] = None,
    title: str = "Decomposition with holes",
) -> None:
    """
    Plot `Decomposition2DPolygon`

    Parameters
    ----------
    decomposition: list[PolygonBlock]
        field `Decomposition2DPolygon.blocks`
    polygon_vertices: list[tuple[float, float]]
        List of polygon vertices
    holes: Optional[List[List[Tuple[float, float]]]]
        List of holes in polygon
    figsize: tuple[int, int]
        Figsize for matplotlib
    savepath: Optional[str]
        Path to save picture
    title: str
        Title of plot
    """
    fig, ax = plt.subplots(figsize=figsize)
    # ax.set_aspect("equal", adjustable="box")

    poly_np = np.array(polygon_vertices)
    if poly_np.ndim == 2 and poly_np.shape[0] > 0:
        ax.plot(
            poly_np[:, 0], poly_np[:, 1], "-k", linewidth=1.2, label="outer contour"
        )

    if holes:
        for h in holes:
            h_np = np.array(h)
            if h_np.ndim == 2 and h_np.shape[0] > 0:
                ax.plot(h_np[:, 0], h_np[:, 1], "-r", linewidth=1.1, label="_nolegend_")

    interior_face = (0.2, 0.6, 0.2, 0.14)
    boundary_face = (0.8, 0.2, 0.2, 0.18)
    rect_edge_color = (0.2, 0.2, 0.2, 0.8)

    xs = []
    ys = []
    plotted_collocation_legend = False

    for block in decomposition:
        left = block.left_down_corner
        right = block.right_up_corner
        kind = block.kind
        clipped = block.clipped_polygon
        clipped_holes = block.clipped_holes
        data = _extract_data_from_block_obj(block)
        x0, y0 = left
        x1, y1 = right
        width = x1 - x0
        height = y1 - y0
        xs += [x0, x1]
        ys += [y0, y1]

        face = interior_face if kind == "interior" else boundary_face
        rect = Rectangle(
            (x0, y0),
            width,
            height,
            linewidth=0.6,
            edgecolor=rect_edge_color,
            facecolor=face,
        )
        ax.add_patch(rect)

        clipped_np = np.array(clipped)
        if clipped_np.shape[0] >= 3:
            ax.add_patch(
                MplPolygon(
                    clipped_np,
                    closed=True,
                    fill=False,
                    edgecolor="black",
                    linewidth=0.8,
                )
            )

        if clipped_holes:
            for ch in clipped_holes:
                ch_np = np.array(ch)
                if ch_np.shape[0] >= 3:
                    ax.add_patch(
                        MplPolygon(
                            ch_np,
                            closed=True,
                            fill=False,
                            edgecolor="red",
                            linewidth=0.8,
                            linestyle="--",
                        )
                    )

        if data is not None:
            darr = np.array(data, dtype=float)
            if darr.ndim == 2 and darr.shape[1] >= 2 and darr.shape[0] > 0:
                ax.scatter(
                    darr[:, 0],
                    darr[:, 1],
                    s=4,
                    c="tab:blue",
                    alpha=0.6,
                    marker=".",
                    label="_nolegend_",
                )
                if not plotted_collocation_legend:
                    plotted_collocation_legend = True

    if xs and ys:
        xmin, xmax = min(xs), max(xs)
        ymin, ymax = min(ys), max(ys)
        dx = (xmax - xmin) * 0.14 + 1e-6
        dy = (ymax - ymin) * 0.14 + 1e-6
        ax.set_xlim(xmin - dx, xmax + dx)
        ax.set_ylim(ymin - dy, ymax + dy)

    legend_elems = [
        Line2D([0], [0], color="k", lw=1.2, label="outer contour"),
        Line2D([0], [0], color="r", lw=1.2, label="hole (wing)", linestyle="-"),
        # Rectangle(
        #     (0, 0),
        #     1,
        #     1,
        #     facecolor=interior_face,
        #     edgecolor=rect_edge_color,
        #     label="interior block",
        # ),
        Rectangle(
            (0, 0),
            1,
            1,
            facecolor=boundary_face,
            edgecolor=rect_edge_color,
            label="boundary block",
        ),
        Line2D(
            [0],
            [0],
            marker=".",
            color="w",
            label="collocation pts",
            markerfacecolor="tab:blue",
            markersize=6,
        ),
    ]
    ax.legend(handles=legend_elems, loc="upper right", fontsize="small")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title(title)
    ax.grid(alpha=0.25)
    plt.tight_layout()
    if savepath:
        plt.savefig(savepath, dpi=450)
        print(f"Saved figure to {savepath}")
    plt.show()
