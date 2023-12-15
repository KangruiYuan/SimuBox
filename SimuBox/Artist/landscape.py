from collections import ChainMap
from decimal import Decimal
from pathlib import Path
from typing import Optional, Union, Sequence, Literal

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.interpolate import griddata
from shapely.geometry import Polygon

from .plotter import plot_locators, plot_savefig
from ..Schema import LandscapeResult, PathLike, CommonLabels, Numeric, Vector, IQResult
from ..Toolkits import read_csv, find_nearest_1d

LAND_PLOT_CONFIG = {
    "font.family": "Times New Roman",
    "font.size": 16,
    "mathtext.fontset": "stix",
    "font.serif": ["SimSun"],
    "axes.unicode_minus": False,
    "xtick.direction": "out",
    "ytick.direction": "out",
    "xtick.major.width": 3,
    "xtick.minor.width": 3,
    "xtick.major.size": 10,
    "xtick.minor.size": 5,
    "ytick.major.width": 3,
    "ytick.minor.width": 3,
    "ytick.major.size": 10,
    "ytick.minor.size": 5,
    "axes.linewidth": 2,
    "legend.frameon": False,
    "legend.fontsize": "small",
}


class Landscaper:
    def __init__(self, path: PathLike, labels: Optional[dict[str, str]] = None):
        self.path = Path(path)
        self.labels = (
            ChainMap(labels, CommonLabels) if labels is not None else CommonLabels
        )

    def data(self, **kwargs):
        if "subset" not in kwargs:
            kwargs["subset"] = ["ly", "lz", "freeE", "phase"]
        return read_csv(self.path, **kwargs)

    @staticmethod
    def get_step_for_discrete(num: np.ndarray, precision: int):
        xs = Decimal(str(num.min())).as_tuple()
        return 10 ** (-abs(xs[2]) - precision)

    @staticmethod
    def levels_IQ(contour_set, levels: Sequence[float] = None, **kwargs):
        # 获取等高面的信息
        if levels is None:
            levels = [0.001, 0.01]
        IQs = []
        for i in range(len(contour_set.collections)):
            # for collection in contour_set.collections:
            level = contour_set.levels[i]
            if level not in levels:
                continue
            collection = contour_set.collections[i]
            contour_paths = collection.get_paths()

            for contour_path in contour_paths:
                vertices = contour_path.vertices
                polygon = Polygon(vertices)
                area = polygon.area
                length = polygon.length
                IQ = 4 * area * np.pi / length**2
                # centroid = polygon.centroid
                print("level: ", level, "面积：", area, "周长：", length, "IQ: ", IQ)
                IQs.append(IQResult(level=level, area=area, length=length, IQ=IQ))
        return IQs

    def prospect(
        self,
        x_axis: str = "ly",
        y_axis: str = "lz",
        target: str = "freeE",
        levels: Optional[Vector] = None,
        precision: int = 3,
        save: Optional[bool] = True,
        tick_num: Optional[int] = None,
        aspect: Union[Literal["auto", "equal", "square"], Numeric] = 1,
        relative: bool = True,
        IQ: bool = True,
        interactive: bool = True,
        data: Optional[pd.DataFrame] = None,
        **kwargs,
    ):
        df: pd.DataFrame = self.data(**kwargs) if data is None else data.copy()
        x_ticks = np.sort(df[x_axis].unique())
        y_ticks = np.sort(df[y_axis].unique())
        min_data = df[df[target] == df[target].min()]

        min_data = min_data.drop_duplicates(subset=[x_axis, y_axis])

        for idx in min_data.index:
            print(
                f"极小值点: {x_axis}: {min_data.loc[idx, x_axis]}, {y_axis}:{min_data.loc[idx, y_axis]}, {target}: {min_data.loc[idx, target]}"
            )

        x_all_point = df[x_axis].values.reshape(-1, 1)
        y_all_point = df[y_axis].values.reshape(-1, 1)
        xy_all_point = np.hstack([x_all_point, y_all_point])
        step_x = self.get_step_for_discrete(x_ticks, precision)
        step_y = self.get_step_for_discrete(y_ticks, precision)
        grid_x, grid_y = np.mgrid[
            x_ticks.min() : x_ticks.max() : step_x,
            y_ticks.min() : y_ticks.max() : step_y,
        ]
        # print(grid_ly.shape)
        grid_target = griddata(
            xy_all_point, df[target].values, (grid_x, grid_y), method="cubic"
        )

        target_mat = grid_target.copy()
        print(f"{target} mat shape: {grid_target.shape}")
        x_ticks = np.sort(grid_x[:, 0])
        y_ticks = np.sort(grid_y[0, :])

        if relative:
            target_mat = target_mat - target_mat.min()

        if levels is not None:
            levels = np.array(levels)

        else:
            assert tick_num is not None
            levels = np.linspace(np.min(target_mat), np.max(target_mat), tick_num)
        ticks = levels

        fig = plt.figure(figsize=kwargs.get("figsize", (8, 6)))
        ax = plt.gca()

        cmap_kind = kwargs.get("cmap", "viridis")
        cmap = plt.colormaps.get_cmap(cmap_kind)
        colors = [cmap(i / len(levels)) for i in range(len(levels))]
        contourf_fig = plt.contourf(
            y_ticks, x_ticks, target_mat, levels=levels, colors=colors
        )
        contour_fig = plt.contour(contourf_fig, colors="w", linewidths=2.5)

        if IQ:
            IQs = self.levels_IQ(contour_fig, levels=kwargs.get("levels_for_IQ"))
        else:
            IQs = []

        inverse_color = kwargs.get("inverse_color", 0)
        clabel_fontsize = kwargs.get("clabel_fontsize", 15)
        manual = kwargs.get("manual", ())

        if manual == "auto":
            diag = np.diagonal(target_mat)
            half_length = len(diag) // 2
            front_half = diag[:half_length]
            back_half = diag[half_length:]
            mark_coords = []
            exclude = kwargs.get("exclude", (min(levels), max(levels)))
            for _level in levels:
                if _level in exclude:
                    continue
                front_idx = find_nearest_1d(front_half, _level)
                mark_coords.append((x_ticks[front_idx], y_ticks[front_idx]))
                back_idx = find_nearest_1d(back_half, _level) + half_length
                mark_coords.append((x_ticks[back_idx], y_ticks[back_idx]))
            manual = mark_coords

        if isinstance(manual, Sequence) and manual:
            # print(manual)
            plt.clabel(
                contour_fig,
                fontsize=clabel_fontsize,
                colors=["w"] * (len(ticks) - inverse_color) + ["k"] * inverse_color,
                fmt="%g",
                manual=manual,
                zorder=7,
            )
        else:
            plt.clabel(
                contour_fig,
                fontsize=clabel_fontsize,
                colors=["w"] * (len(ticks) - inverse_color) + ["k"] * inverse_color,
                fmt="%g",
                zorder=7,
            )

        colorbar_fontsize = kwargs.get("colorbar_fontsize", 15)
        colorbar_accuracy = kwargs.get("colorbar_accuracy", 8)
        colorbar_pad = kwargs.get("colorbar_pad", 0.1)
        shrink = kwargs.get("shrink", 1.0)
        clb = plt.colorbar(contourf_fig, ticks=ticks, shrink=shrink, pad=colorbar_pad)
        clb.set_ticklabels(
            np.around(ticks, colorbar_accuracy), fontsize=colorbar_fontsize
        )
        # clb.set_ylabel(r'$\Delta F/k_{\rm{B}}T$', fontsize=20)
        if colorbar_title := kwargs.get("colorbar_title", r"$\Delta F/nk_{\rm B}T$"):
            clb.ax.set_title(colorbar_title, fontsize=colorbar_fontsize, pad=18)
        clb.ax.tick_params(which="major", width=2)

        label_fontsize = kwargs.get("label_fontsize", 20)
        plt.xlabel(self.labels.get(x_axis, x_axis), fontsize=label_fontsize)
        plt.ylabel(self.labels.get(y_axis, y_axis), fontsize=label_fontsize)

        if aspect == "auto":
            ax.set_aspect(1.0 / ax.get_data_ratio())
        elif aspect in ("square", "equal"):
            plt.axis(aspect)
        elif aspect is not None:
            ax.set_aspect(aspect)

        if point_list := kwargs.get("point_list", []):
            for p in point_list:
                plt.scatter(
                    p[0], p[1], s=p[-1], c=p[-2], marker=p[2], alpha=1, zorder=6
                )

        # print(kwargs["xmajor"])
        plot_locators(**kwargs)

        plt.tight_layout()
        plot_savefig(self, save=save, **kwargs)
        if interactive:
            plt.show()

        return LandscapeResult(
            mat=target_mat,
            x_ticks=x_ticks,
            y_ticks=y_ticks,
            levels=levels,
            ticks=ticks,
            fig=fig,
            ax=ax,
            contourf_fig=contourf_fig,
            contour_fig=contour_fig,
            clb=clb,
            IQs=IQs,
        )
