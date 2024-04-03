from typing import Union, Iterable, Optional, Tuple, Sequence, Literal, List

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pyvista as pv
from skimage.measure import marching_cubes

from .plotter import plot_savefig
from ..schema import Density, RealNum, PathLike, DensityParseResult
from ..toolkits import parse_density

__all__ = ["iso2D", "iso3D"]

def iso3D(
    density: Union[Density, DensityParseResult],
    target: Optional[Union[int, Iterable[int]]] = 0,
    permute: Optional[Iterable[int]] = None,
    level: Union[float, Iterable[float]] = 0.5,
    backend: Literal["vista", "mpl"] = "vista",
    interactive: bool = True,
    frame: bool = True,
    save: Optional[PathLike] = None,
    **kwargs,
):

    if not isinstance(density, DensityParseResult):
        parsed = parse_density(density, target, permute, **kwargs)
    else:
        parsed = density
    lxlylz = parsed.lxlylz
    NxNyNz = parsed.NxNyNz
    mats = parsed.mat
    if all(lxlylz / parsed.expand == 1):
        lxlylz *= NxNyNz

    x, y, z = np.mgrid[
        0 : lxlylz[0] : complex(0, NxNyNz[0]),
        0 : lxlylz[1] : complex(0, NxNyNz[1]),
        0 : lxlylz[2] : complex(0, NxNyNz[2]),
    ]

    colors = kwargs.get("colors", [])
    for c in ["blue", "red", "green"]:
        if c not in colors:
            colors.append(c)
    style = kwargs.get("style", "surface")
    opacity = kwargs.get("opacity", 0.8)

    if backend == "vista":
        grid = pv.StructuredGrid(x, y, z)
        level = level if isinstance(level, Sequence) else [level]
        pv.set_plot_theme(kwargs.get("theme", "document"))  # type: ignore
        p = pv.Plotter(
            line_smoothing=True, polygon_smoothing=True, lighting="light kit"
        )

        for idx, target in enumerate(parsed.target):
            contours = grid.contour(
                level,
                scalars=mats[idx].flatten(),
                method="contour",
            )

            if frame and idx == 0:
                p.add_mesh(grid.outline(), color="k")

            p.add_mesh(
                contours,
                color=colors[idx],
                show_edges=False,
                style=style,
                smooth_shading=True,
                opacity=opacity,
                pbr=True,
                roughness=0,
            )
        p.view_isometric()
        p.background_color = kwargs.get("bk", "white")
        if save is not None:
            p.save_graphic(filename=save)
        if interactive:
            p.show()
        return p
    elif backend == "mpl":
        if isinstance(level, list) and len(level) == 1:
            level = level[0]
        assert isinstance(level, (float, int)), "等值面应为数值"
        vert, faces, _, _ = marching_cubes(
            parsed.mat.squeeze(), level, spacing=(1, 1, 1)
        )
        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")
        ax.plot_trisurf(vert[:, 0], vert[:, 1], faces, vert[:, 2], lw=1)
        plt.show()
        return fig, ax


def iso2D(
    density: Union[Density, DensityParseResult],
    target: Optional[Union[int, Iterable[int], str]] = 0,
    permute: Optional[Iterable[int]] = None,
    slices: Optional[tuple[int, int]] = None,
    titles: Optional[Sequence[str]] = None,
    grid: Optional[Sequence[int]] = None,
    figsize: Optional[Sequence[RealNum]] = None,
    aspect: Optional[RealNum] = None,
    norm: Union[Tuple[RealNum, RealNum], List[Tuple[RealNum, RealNum]]] = None,
    scale: Union[Tuple[RealNum, RealNum], List[Tuple[RealNum, RealNum]]] = None,
    cmap: str = "jet",
    label: bool = True,
    colorbar: bool = False,
    interactive: bool = True,
    save: Union[PathLike, bool] = False,
    fontsize: int = 25,
    verbose: bool = False,
    stack: bool = False,
    **kwargs,
):
    """

    :param density: Density | DensityParseResult
        密度信息或者二次处理后的密度信息。

    :param target:
    :param permute:
    :param slices:
    :param titles:
    :param grid:
    :param figsize:
    :param aspect:
    :param norm:
    :param scale:
    :param cmap:
    :param label:
    :param colorbar:
    :param interactive:
    :param save:
    :param fontsize:
    :param verbose:
    :param stack:
    :param kwargs:
    :return:
    """

    if not isinstance(density, DensityParseResult):
        parsed = parse_density(density, target, permute, slices, **kwargs)
    else:
        parsed = density
    if stack:
        length = 1
    else:
        length = int(parsed.mat.shape[0])

    if titles is not None:
        assert (
            len(titles) == length
        ), f"请提供与子图数量相等的子图标题，需要{length}个，现仅提供了{len(titles)}个。"

    if grid is None:
        grid = (1, length)
    if verbose: print(grid)

    if figsize is None:
        figsize = (grid[1] * 4, grid[0] * 3.5)

    fig, axes = plt.subplots(nrows=grid[0], ncols=grid[1], figsize=figsize)
    if length == 1:
        axes = [axes]
    else:
        axes = axes.flatten()

    aspect = aspect if aspect else parsed.lxlylz[1] / parsed.lxlylz[0]
    aspect = aspect * parsed.NxNyNz[0] / parsed.NxNyNz[1]
    rotation = kwargs.get("rotation", 1)

    if norm is not None:
        if isinstance(norm[0], (int, float)):
            norms = length * [norm]
        else:
            norms = norm

    if scale is not None:
        if isinstance(scale[0], (int, float)):
            scales = length * [scale]
        else:
            scales = scale

    mats = [np.stack(parsed.mat, axis=-1)] if stack else parsed.mat
    for i, (ax, phi) in enumerate(zip(axes, mats)):
        phi = np.rot90(phi, rotation)
        vmin, vmax = phi.min(), phi.max()

        if norm is not None:
            normalizer = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
            phi = normalizer(phi) * (norms[i][1] - norms[i][0]) + norms[i][0]
            if verbose: print(norms[i])

        if scale is not None:
            vmin, vmax = scales[i]
        else:
            vmin, vmax = phi.min(), phi.max()

        if verbose: print(vmin, vmax)
        im = ax.imshow(
            phi,
            cmap=cmap,
            interpolation="spline36",
            aspect=aspect,
            vmin=vmin,
            vmax=vmax,
        )
        if titles is not None:
            ax.set_title(titles[i])

        for _dir in ["bottom", "top", "right", "left"]:
            ax.spines[_dir].set_color(None)
        ax.yaxis.set_ticks([])
        ax.xaxis.set_ticks([])
        if label:
            ax.set_xlabel(
                r"{} $R_g$".format(round(parsed.lxlylz[0], 3)), fontsize=fontsize
            )
            ax.set_ylabel(
                r"{} $R_g$".format(round(parsed.lxlylz[1], 3)), fontsize=fontsize
            )
        if colorbar:
            clb = plt.colorbar(im, ax=ax, fraction=kwargs.get("fraction", 0.05))
            clb.ax.tick_params(labelsize=kwargs.get("clb_labelsize", fontsize - 10))

    fig.tight_layout()
    plot_savefig(
        density,
        prefix=kwargs.get("prefix", "iso2d"),
        suffix=kwargs.get("suffix", "_".join(str(i) for i in parsed.target)),
        save=save,
        **kwargs,
    )
    if interactive:
        plt.show()
    return fig, axes
