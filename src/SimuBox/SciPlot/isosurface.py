from typing import List, Union

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pyvista as pv
from mayavi import mlab
from skimage.measure import marching_cubes


class IsoSurf():

    @classmethod
    def iso3D(
        cls,
        phi: Union[str, np.ndarray],
        level: Union[float, List[float]] = 0.5,
        backend: str = "vista",
        **kwargs,
    ):
        assert cls.dim == 3, f"数据应该是3维，目前为{cls.dim}维"

        vol: np.ndarray = getattr(cls, phi) if isinstance(phi, str) else phi
        lxlylz = getattr(cls, "lxlylz")
        NxNyNz = getattr(cls, "NxNyNz")

        x, y, z = np.mgrid[
            0 : lxlylz[0] : complex(0, NxNyNz[0]),
            0 : lxlylz[1] : complex(0, NxNyNz[1]),
            0 : lxlylz[2] : complex(0, NxNyNz[2]),
        ]

        # print(x.shape, y.shape, z.shape, vol.shape)

        if backend == "vista":
            grid = pv.StructuredGrid(x, y, z)
            grid["vol"] = vol.flatten()
            _level = level if isinstance(level, list) else [level]
            contours = grid.contour(_level)  # type: ignore
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
            verts, faces, _, _ = marching_cubes(vol, level, spacing=(1, 1, 1))
            fig = plt.figure()
            ax = fig.add_subplot(111, projection="3d")
            ax.plot_trisurf(verts[:, 0], verts[:, 1], faces, verts[:, 2], lw=1)
            plt.show()
        elif backend == "mayavi":
            vol = vol.T
            cls.mayavi_contour3d(x, y, z, vol, level, **kwargs)

    @mlab.show
    def mayavi_contour3d(self, x, y, z, vol, level, **kwargs):
        mlab.contour3d(
            x, y, z, vol, contours=level, transparent=kwargs.get("transparent", True)
        )

    def iso2D(self, phi: Union[str, np.ndarray, List[str]], **kwargs):
        if isinstance(phi, str):
            vol = [getattr(self, phi)]
        elif isinstance(phi, list):
            if isinstance(phi[0], str):
                vol = [getattr(self, _phi) for _phi in phi]
        else:
            raise ValueError

        fig, axes = plt.subplots(1, len(vol), figsize=(4 * len(vol), 3.5))
        if len(vol) == 1:
            axes = [axes]

        rot = kwargs.get("rot", 3)
        names = kwargs.get("names", False)
        alpha = kwargs.get("alpha", 1)
        assert self.lxlylz is not None
        asp = kwargs.get("asp", self.lxlylz[2] / self.lxlylz[1])
        if norm := kwargs.get("norm", False):
            normer = mpl.colors.Normalize(vmin=0, vmax=1)

        for i, (ax, _vol) in enumerate(zip(axes, vol)):
            if rot:
                _vol = np.rot90(_vol, rot)
            if norm:
                _vol = normer(_vol)  # type: ignore
            ax.imshow(  # type: ignore
                _vol, cmap="jet", alpha=alpha, interpolation="spline36", aspect=asp
            )
            if isinstance(names, list):
                ax.set_title(names[i])  # type: ignore
            elif names == "auto":
                ax.set_title(phi[i])
            ax.spines["bottom"].set_color(None)
            ax.spines["top"].set_color(None)
            ax.spines["right"].set_color(None)
            ax.spines["left"].set_color(None)  # type: ignore
            ax.yaxis.set_ticks([])  # type: ignore
            ax.xaxis.set_ticks([])
            ax.set_xlabel(str(round(self.lxlylz[2], 3)) + " Rg", fontsize=20)
            ax.set_ylabel(str(round(self.lxlylz[1], 3)) + " Rg", fontsize=20)  # type: ignore

        fig.tight_layout()
        if save := kwargs.get("save", False):
            plt.savefig(save, dpi=300)
        plt.show()
