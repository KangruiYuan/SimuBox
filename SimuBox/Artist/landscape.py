from decimal import Decimal
from functools import cached_property
from pathlib import Path
from typing import Optional, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.interpolate import griddata
from shapely.geometry import Polygon

from collections import ChainMap
from .PlotUtils import plot_locators, plot_savefig
from ..Schema import LandscapeResult, PathType, CommonLabels
from ..Toolkits import read_csv

LAND_PLOT_CONFIG = {
    "font.family": "Times New Roman",
    "font.size": 30,
    "mathtext.fontset": "stix",
    "font.serif": ["SimSun"],
    "axes.unicode_minus": False,
    "xtick.direction": "out",
    "ytick.direction": "out",
    "xtick.major.width": 4,
    "xtick.minor.width": 4,
    "xtick.major.size": 12,
    "xtick.minor.size": 6,
    "ytick.major.width": 4,
    "ytick.minor.width": 4,
    "ytick.major.size": 12,
    "ytick.minor.size": 6,
    "axes.linewidth": 2,
    "legend.frameon": False,
    "legend.fontsize": "small",
}


class Landscaper:
    def __init__(self, path: PathType, labels: Optional[dict[str, str]] = None):
        self.path = Path(path)
        self.labels = (
            ChainMap(labels, CommonLabels) if labels is not None else CommonLabels
        )

    @cached_property
    def data(self):
        return read_csv(self.path, subset=["ly", "lz", "freeE", "phase"])

    @staticmethod
    def get_w_s(num: np.ndarray, Res: int):
        xs = Decimal(str(num.min())).as_tuple()
        return 10 ** (-abs(xs[2]) - Res)  # type: ignore

    @staticmethod
    def levels_IQ(contour_set, levels: list[float] = None):
        # 获取等高面的信息
        if levels is None:
            levels = [0.001, 0.01]
        for i in range(len(contour_set.collections)):
            # for collection in contour_set.collections:
            level = contour_set.levels[i]
            if level not in levels:
                continue
            collection = contour_set.collections[i]
            path_list = collection.get_paths()
            for path in path_list:
                vertices = path.vertices
                polygon = Polygon(vertices)
                area = polygon.area
                length = polygon.length
                IQ = 4 * area * np.pi / length**2
                # centroid = polygon.centroid
                print("level: ", level, "面积：", area, "周长：", length, "IQ: ", IQ)

    def prospect(
        self,
        AxisX: str = "ly",
        AxisY: str = "lz",
        Vals: str = "freeE",
        precision: int = 3,
        save: Optional[Union[Path, str, bool]] = True,
        tick_num: int = 11,
        asp: Union[str, float, int] = 1,
        **kwargs,
    ):
        df: pd.DataFrame = self.data
        # df: pd.DataFrame = deepcopy(df)
        ly = np.sort(df[AxisX].unique())
        lz = np.sort(df[AxisY].unique())
        min_data = df[df[Vals] == df[Vals].min()]

        min_data = min_data.drop_duplicates(subset=["lz", "ly"])

        for idx in min_data.index:
            print(
                "极小值点: ly: {}, lz:{}, freeE: {}".format(
                    min_data.loc[idx, AxisX],
                    min_data.loc[idx, AxisY],
                    min_data.loc[idx, Vals],
                )
            )

        y_all = df[AxisX].values.reshape(-1, 1)  # type: ignore
        x_all = df[AxisY].values.reshape(-1, 1)  # type: ignore
        yx_all = np.hstack([y_all, x_all])
        step_ly = self.get_w_s(ly, precision)
        step_lz = self.get_w_s(lz, precision)
        grid_ly, grid_lz = np.mgrid[
            ly.min() : ly.max() : step_ly, lz.min() : lz.max() : step_lz
        ]
        # print(grid_ly.shape)
        grid_freeE = griddata(
            yx_all, df[Vals].values, (grid_ly, grid_lz), method="cubic"
        )

        freeEMat = grid_freeE.copy()
        print(f"Free energy mat shape: {grid_freeE.shape}")
        ly = np.unique(grid_ly)
        lz = np.unique(grid_lz)

        if kwargs.get("relative", True):
            freeEMat = freeEMat - freeEMat.min()

        if levels := kwargs.get("levels", []):
            # levels = kwargs.get("levels", False)
            ticks = levels

        else:
            levels = np.linspace(np.min(freeEMat), np.max(freeEMat), tick_num)
            ticks = levels

        fig = plt.figure(figsize=kwargs.get("figsize", (16, 12)))
        ax = plt.gca()
        reverse = kwargs.get("reverse", 3)
        contourf_fig = plt.contourf(lz, ly, freeEMat, levels=levels, cmap="viridis")
        contour_fig = plt.contour(contourf_fig, colors="w", linewidths=2.5)
        self.levels_IQ(contour_fig)

        if manual := kwargs.get("manual", []):
            plt.clabel(
                contour_fig,
                fontsize=30,
                colors=["w"] * (len(ticks) - reverse) + ["k"] * reverse,
                fmt="%g",
                manual=manual,
                zorder=7,
            )
        else:
            plt.clabel(
                contour_fig,
                fontsize=30,
                colors=["w"] * (len(ticks) - reverse) + ["k"] * reverse,
                fmt="%g",
            )

        shrink = kwargs.get("shrink", 1.0)
        clb = plt.colorbar(contourf_fig, ticks=ticks, shrink=shrink, pad=-0.15)
        clb.set_ticklabels(np.around(ticks, kwargs.get("clb_acc", 6)), fontsize=35)
        # clb.set_ylabel(r'$\Delta F/k_{\rm{B}}T$', fontsize=20)
        clb.ax.set_title(r"$\Delta F/nk_{B}T$", fontsize=40, pad=18)
        clb.ax.tick_params(which="major", width=2)

        plt.xlabel(self.labels[AxisY], fontdict={"size": 45})
        plt.ylabel(self.labels[AxisX], fontdict={"size": 45})

        if asp is not None:
            if asp == "auto":
                ax.set_aspect(1.0 / ax.get_data_ratio())
            elif asp == "square" or asp == "equal":
                plt.axis(asp)
            else:
                ax.set_aspect(asp)

        if point_list := kwargs.get("point_list", []):
            # p = [x, y, marker, c, size]
            for p in point_list:
                plt.scatter(
                    p[0], p[1], s=p[-1], c=p[-2], marker=p[2], alpha=1, zorder=6
                )

        plot_locators(**kwargs)

        plt.tight_layout()
        plot_savefig(self)
        plt.show()

        return LandscapeResult(
            freeEMat=freeEMat,
            ly=ly,
            lz=lz,
            levels=levels,
            ticks=ticks,
            fig=fig,
            ax=ax,
            contourf_fig=contourf_fig,
            contour_fig=contour_fig,
            clb=clb,
        )
