from typing import List, Union, Iterable, Optional, Tuple, Sequence

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pyvista as pv
from mayavi import mlab
from skimage.measure import marching_cubes

from ..Schema import Density, NumericType
from ..Toolkits import parse_density


def iso3D(
    density: Density,
    target: int = 0,
    permute: Optional[Iterable[int]] = None,
    level: Union[float, List[float]] = 0.5,
    backend: str = "vista",
    **kwargs,
):
    phi, NxNyNz, lxlylz = parse_density(density, target, permute)

    x, y, z = np.mgrid[
        0 : lxlylz[0] : complex(0, NxNyNz[0]),
        0 : lxlylz[1] : complex(0, NxNyNz[1]),
        0 : lxlylz[2] : complex(0, NxNyNz[2]),
    ]

    if backend == "vista":
        grid = pv.StructuredGrid(x, y, z)
        grid["vol"] = phi.flatten()
        level = level if isinstance(level, Sequence) else [level]
        contours = grid.contour(level)  # type: ignore
        pv.set_plot_theme(kwargs.get("theme", "document"))  # type: ignore
        p = pv.Plotter(
            line_smoothing=True, polygon_smoothing=True, lighting="light kit"
        )
        p.add_mesh(
            contours,
            color=kwargs.get("color", "white"),
            show_edges=False,
            style="surface",
            smooth_shading=True,
            # Enable physics based rendering(PBR)
            pbr=True,
            roughness=0,
        )
        p.show()
    elif backend == "mpl":
        if isinstance(level, list) and len(level) == 1:
            level = level[0]
        assert isinstance(level, (float, int)), "等值面应为数值"
        vert, faces, _, _ = marching_cubes(phi, level, spacing=(1, 1, 1))
        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")
        ax.plot_trisurf(vert[:, 0], vert[:, 1], faces, vert[:, 2], lw=1)
        plt.show()
    elif backend == "mayavi":
        phi = phi.T
        mayavi_contour3d(x, y, z, phi, level, **kwargs)

@mlab.show
def mayavi_contour3d(x, y, z, vol, level, **kwargs):
    mlab.contour3d(
        x, y, z, vol, contours=level, transparent=kwargs.get("transparent", True)
    )


def iso2D(
    density: Density,
    target: Optional[Union[int, Iterable[int]]] = 0,
    permute: Optional[Iterable[int]] = None,
    titles: Optional[Sequence[str]] = None,
    norm: Optional[Tuple[NumericType, NumericType]] = None,
    label: bool = True,
    **kwargs,
):

    phis, _, lxlylz = parse_density(density, target, permute)
    if isinstance(target, int):
        phis = [phis]
    length = len(phis)

    if titles is not None:
        assert len(titles) == length, "如果要指定titles，请与需要绘制的图幅相等"

    fig, axes = plt.subplots(1, length, figsize=(4 * length, 3.5))
    if length == 1:
        axes = [axes]

    asp = kwargs.get("asp", lxlylz[1] / lxlylz[0])
    rotation = kwargs.get("rotation", 3)

    if norm is not None:
        normalizer = mpl.colors.Normalize(vmin=norm[0], vmax=norm[1])
    else:
        normalizer = lambda x: x

    for i, (ax, phi) in enumerate(zip(axes, phis)):
        phi = np.rot90(phi, rotation)
        phi = normalizer(phi)
        ax.imshow(phi, cmap="jet", interpolation="spline36", aspect=asp)
        if titles is not None:
            ax.set_title(titles[i])

        for _dir in ["bottom", "top", "right", "left"]:
            ax.spines[_dir].set_color(None)
        ax.yaxis.set_ticks([])  # type: ignore
        ax.xaxis.set_ticks([])
        if label:
            ax.set_xlabel(r"{} $R_g$".format(round(density.lxlylz[1], 3)), fontsize=20)
            ax.set_ylabel(r"{} $R_g$".format(round(density.lxlylz[0], 3)), fontsize=20)  # type: ignore

    fig.tight_layout()
    if save := kwargs.get("save"):
        plt.savefig(save, dpi=150)
    plt.show()
