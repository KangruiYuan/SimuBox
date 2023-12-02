from typing import List, Union, Iterable, Optional, Tuple, Sequence

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pyvista as pv

from skimage.measure import marching_cubes

from ..Schema import Density, NumericType
from ..Toolkits import parse_density


def iso3D(
    density: Density,
    target: int = 0,
    permute: Optional[Iterable[int]] = None,
    level: Union[float, List[float]] = 0.5,
    backend: str = "vista",
    interactive: bool = True,
    frame: bool = True,
    **kwargs,
):
    parsed = parse_density(density, target, permute, **kwargs)
    lxlylz = parsed.lxlylz
    NxNyNz = parsed.NxNyNz
    if all(lxlylz / parsed.expand == 1):
        lxlylz *= NxNyNz

    x, y, z = np.mgrid[
        0 : lxlylz[0] : complex(0, NxNyNz[0]),
        0 : lxlylz[1] : complex(0, NxNyNz[1]),
        0 : lxlylz[2] : complex(0, NxNyNz[2]),
    ]

    if backend == "vista":
        grid = pv.StructuredGrid(x, y, z)
        level = level if isinstance(level, Sequence) else [level]
        contours = grid.contour(
            level,
            scalars=parsed.mat.flatten(),
            # rng=[1,2],
            method="contour",
        )
        pv.set_plot_theme(kwargs.get("theme", "document"))  # type: ignore
        p = pv.Plotter(
            line_smoothing=True, polygon_smoothing=True, lighting="light kit"
        )
        if frame:
            p.add_mesh(grid.outline(), color="k")

        p.add_mesh(
            contours,
            color=kwargs.get("color", "blue"),
            show_edges=False,
            style=kwargs.get("style", "surface"),
            smooth_shading=True,
            opacity=kwargs.get("opacity", 0.8),
            pbr=True,
            roughness=0,
        )
        p.view_isometric()
        p.background_color = kwargs.get("bk", "white")
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
    density: Density,
    target: Optional[Union[int, Iterable[int]]] = 0,
    permute: Optional[Iterable[int]] = None,
    slices: Optional[tuple[int, int]] = None,
    titles: Optional[Sequence[str]] = None,
    norm: Optional[Tuple[NumericType, NumericType]] = None,
    label: bool = True,
    colorbar: bool = False,
    interactive: bool = True,
    **kwargs,
):

    parsed = parse_density(density, target, permute, slices, **kwargs)
    length = int(parsed.mat.shape[0])

    if titles is not None:
        assert len(titles) == length, "如果要指定titles，请与需要绘制的图幅相等"

    fig, axes = plt.subplots(1, length, figsize=(4 * length, 3.5))
    if length == 1:
        axes = [axes]

    asp = kwargs.get("asp", parsed.lxlylz[1] / parsed.lxlylz[0])
    rotation = kwargs.get("rotation", 3)

    if norm is not None:
        normalizer = mpl.colors.Normalize(vmin=norm[0], vmax=norm[1])
    else:
        normalizer = lambda x: x

    for i, (ax, phi) in enumerate(zip(axes, parsed.mat)):
        phi = np.rot90(phi, rotation)
        phi = normalizer(phi)
        im = ax.imshow(phi, cmap="jet", interpolation="spline36", aspect=asp)
        if titles is not None:
            ax.set_title(titles[i])

        for _dir in ["bottom", "top", "right", "left"]:
            ax.spines[_dir].set_color(None)
        ax.yaxis.set_ticks([])  # type: ignore
        ax.xaxis.set_ticks([])
        if label:
            ax.set_xlabel(r"{} $R_g$".format(round(parsed.lxlylz[0], 3)), fontsize=20)
            ax.set_ylabel(r"{} $R_g$".format(round(parsed.lxlylz[1], 3)), fontsize=20)
        if colorbar:
            plt.colorbar(im, ax=ax)

    fig.tight_layout()
    if save := kwargs.get("save"):
        plt.savefig(save, dpi=150)
    if interactive:
        plt.show()
    return fig, axes