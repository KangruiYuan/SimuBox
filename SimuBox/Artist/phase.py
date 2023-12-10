from collections import ChainMap
from pathlib import Path
from typing import Any, Optional, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from numpy.polynomial import Chebyshev
from scipy.io import loadmat

from .PlotUtils import plot_locators, plot_savefig, generate_colors
from ..Schema import DetectionMode, CompareResult, PhasePointData, CommonLabels, PathLike
from ..Toolkits import read_csv

PHASE_PLOT_CONFIG = {
    "font.family": "Times New Roman",
    "font.serif": ["SimSun"],
    "font.size": 16,
    "mathtext.fontset": "stix",
    "axes.unicode_minus": False,
    "xtick.direction": "in",
    "ytick.direction": "in",
    "xtick.top": True,
    "ytick.right": True,
    "xtick.major.width": 3,
    "xtick.minor.width": 3,
    "xtick.major.size": 10,
    "xtick.minor.size": 5,
    "ytick.major.width": 3,
    "ytick.minor.width": 3,
    "ytick.major.size": 10,
    "ytick.minor.size": 5,
    "xtick.labelsize": "medium",
    "ytick.labelsize": "medium",
    "axes.linewidth": 3,
    "legend.frameon": False,
    "legend.fontsize": "medium",
}


class PhaseDiagram:
    def __init__(
        self,
        path: PathLike,
        xlabel: str,
        ylabel: str,
        colors: Optional[dict] = None,
        labels: Optional[dict] = None,
    ) -> None:
        self.path = Path(path)
        self.colors = colors if colors is not None else {}
        self.labels = (
            ChainMap(labels, CommonLabels) if labels is not None else CommonLabels
        )
        self.xlabel = xlabel
        self.ylabel = ylabel

    def query_point(
        self,
        data: pd.DataFrame,
        xval: Optional[Union[float, str, int]] = None,
        yval: Optional[Union[float, str, int]] = None,
        **kwargs,
    ):
        res = data.copy()
        columns = res.columns
        if xval:
            res = res[res[self.xlabel] == xval]
        if yval:
            res = res[res[self.ylabel] == yval]
        for key, val in kwargs.items():
            if key in columns:
                res = res[res[key] == val]
        return res

    @staticmethod
    def checkin(phase: list, candidates: list):
        if len(phase) == 0:
            return False
        for i in phase:
            if i in candidates:
                return True
        return False

    def compare(
        self,
        name: Optional[str] = None,
        path: Optional[Union[str, Path]] = None,
        plot: bool = True,
        acc: int = 3,
        **kwargs,
    ):
        if path is None:
            assert name is not None
            path = (self.path / name).with_suffix(".csv")
        else:
            if not isinstance(path, Path):
                path = Path(path)

        df = read_csv(path=path, acc=acc)
        print(f"Include phase: {set(df['phase'].values)}")

        plot_dict = dict()
        y_set = np.sort(df[self.ylabel].unique())
        x_set = np.sort(df[self.xlabel].unique())

        exclude = kwargs.get("exclude", [])

        mat = np.zeros((len(y_set), len(x_set)), dtype=PhasePointData)
        for i, y in enumerate(y_set):
            for j, x in enumerate(x_set):
                phase = "unknown"
                temp_data = df[(df[self.ylabel] == y) & (df[self.xlabel] == x)]
                min_data = temp_data[temp_data["freeE"] == temp_data["freeE"].min()]
                if len(min_data) == 0:
                    continue
                freeE = min_data["freeE"].min()
                min_label = min_data["phase"].unique()
                lxly = min_data["lxly"].unique()
                lylx = np.around(1 / lxly, acc)
                lylz = min_data["lylz"].unique()
                lzly = np.around(1 / lylz, acc)

                if self.checkin(exclude, min_label):
                    phase = min_label[0]
                elif "L" in min_label:
                    phase = "L"
                elif self.checkin(["C4", "Crect"], min_label):
                    if any(lylz == 1):
                        phase = "C4"
                    else:
                        phase = "Crect"
                elif len(min_label) == 1:
                    if self.checkin(["C6", "C3"], min_label):
                        if lylz == np.around(np.sqrt(3), acc) or lzly == np.around(
                            1 / np.sqrt(3), acc
                        ):
                            phase = min_label[0]
                    elif self.checkin(["iHPa"], min_label):
                        if lxly == np.around(np.sqrt(3), acc) or lylx == np.around(
                            1 / np.sqrt(3), acc
                        ):
                            phase = "iHPa"
                        elif lylz == np.around(np.sqrt(2), acc) or lxly == 1:
                            phase = "SC"
                    elif "PL" in min_label:
                        if lxly == np.around(np.sqrt(3), acc) or lylx == np.around(
                            1 / np.sqrt(3), acc
                        ):
                            phase = "PL"
                    elif self.checkin(["L", "Disorder", "O70"], min_label):
                        phase = min_label[0]
                    elif self.checkin(
                        ["SC", "SG", "DG", "BCC", "FCC", "sdgn"], min_label
                    ):
                        if lxly == 1 and lylz == 1:
                            phase = min_label[0]
                    else:
                        phase = "_".join([phase, min_label[0]])
                mat[i][j] = PhasePointData(phase=phase, x=x, y=y, value=freeE)
                if phase in plot_dict:
                    for attr, val in zip(
                        [self.xlabel, self.ylabel, "freeE", "lylz", "lxly"],
                        [x, y, freeE, lylz, lxly],
                    ):
                        plot_dict[phase][attr].append(val)
                else:
                    plot_dict[phase] = {
                        self.xlabel: [x],
                        self.ylabel: [y],
                        "freeE": [freeE],
                        "lylz": [lylz],
                        "lxly": [lxly],
                    }

        comp_res = CompareResult(df=df, plot_dict=plot_dict, mat=mat)

        if plot:
            fig, ax = plt.subplots(1, 1, figsize=(8, 6))
            for key, value in plot_dict.items():
                if key not in self.colors:
                    self.colors[key] = generate_colors(mode="HEX")[0]
                ax.scatter(
                    value[self.xlabel],
                    value[self.ylabel],
                    c=self.colors[key],
                    label=key,
                )
            ax.tick_params(top="on", right="on", which="both")
            ax.tick_params(which="both", width=2, length=4, direction="in")
            plot_locators(**kwargs)
            ax.set_xlabel(
                self.labels.get(self.xlabel, self.xlabel), fontdict={"size": 20}
            )
            ax.set_ylabel(
                self.labels.get(self.ylabel, self.ylabel), fontdict={"size": 20}
            )
            ax.legend(frameon=False, loc="upper left", bbox_to_anchor=(0.98, 0.8))
            fig.tight_layout()
            comp_res.fig = fig
            comp_res.ax = ax

        return comp_res

    @staticmethod
    def cross_point(x1, y1, x2, y2, x3, y3, x4, y4):
        b1 = (y2 - y1) * x1 + (x1 - x2) * y1
        b2 = (y4 - y3) * x3 + (x3 - x4) * y3
        D = (x2 - x1) * (y4 - y3) - (x4 - x3) * (y2 - y1)
        D1 = b2 * (x2 - x1) - b1 * (x4 - x3)
        D2 = b2 * (y2 - y1) - b1 * (y4 - y3)
        return D1 / D, D2 / D

    @staticmethod
    def de_unknown(comp_res: CompareResult):
        mat = comp_res.mat
        for i in range(mat.shape[0]):
            for j in range(mat.shape[1]):
                try:
                    mat[i][j].phase = mat[i][j].phase.lstrip("unknown_")
                except AttributeError:
                    continue
        comp_res.mat = mat
        return comp_res

    def scan(self, folder: Optional[Union[str, Path]] = None, **kwargs):
        if folder is not None:
            if isinstance(folder, str):
                folder = Path(folder)
        else:
            folder = self.path

        filelist = list(folder.glob("*.csv"))
        fig, ax = plt.subplots(figsize=kwargs.get("figsize", (8, 6)))
        skip: list[str] = kwargs.get("skip", [])
        extract: list[str] = kwargs.get("extract", [])
        inverse_lst: list[str] = kwargs.get("inverse", [])
        deunknown: list[str] = kwargs.get("deunknown", [])
        ann_dict: dict[str, Any] = kwargs.get("annotation", {})

        for filename in filelist:
            tmp_name = filename.stem
            tmp_dict: dict = ann_dict.get(tmp_name, {})
            print(tmp_name, end="\t")
            if tmp_name in skip:
                print("Skip...")
                continue
            else:
                comp_res = self.compare(
                    path=filename, plot=False, acc=tmp_dict.get("acc", 2)
                )

            if tmp_name in deunknown or deunknown == "All":
                comp_res = self.de_unknown(comp_res)

            if tmp_name in extract:
                tmp_xys = self.extract_edge(comp_res.mat, **tmp_dict)
            else:
                # tmp_xys = self.boundary(comp_res.df, comp_res.mat, mode=["hori", "ver"])
                tmp_xys = self.boundary_detect(comp_res, **tmp_dict)

            if tmp_name in inverse_lst:
                tmp_xys = tmp_xys[np.argsort(tmp_xys[:, 1])]
                self.draw_line(tmp_xys, ax=ax, annotation=tmp_dict, inverse=True)
            else:
                tmp_xys = tmp_xys[np.argsort(tmp_xys[:, 0])]
                self.draw_line(tmp_xys, ax=ax, annotation=tmp_dict)

        if phase_name := ann_dict.get("phase_name", ()):
            for name, _x, _y in phase_name:
                ax.text(_x, _y, name, fontsize=20, color="k")

        if phase_name_arrow := ann_dict.get("phase_name_arrow", ()):
            for name, _x, _y, _text_x, _text_y in phase_name_arrow:
                ax.annotate(
                    name,
                    xy=(_x, _y),
                    xycoords="data",
                    xytext=(_text_x, _text_y),
                    textcoords="data",
                    # weight="bold",
                    color="k",
                    fontsize=20,
                    arrowprops=dict(
                        arrowstyle="->", connectionstyle="arc3", color="k", lw=2
                    ),
                )

        if mat_path := kwargs.get("mat_path", None):
            wrongline = loadmat(mat_path)["origin_wrong"]
            self.draw_line(
                wrongline,
                ax=ax,
                annotation=kwargs.get(
                    "mat", {"adde": 0.001, "ls": ":", "shrinks": 7, "alpha": 0.5}
                ),
            )

        plt.margins(0)
        plot_locators(**kwargs)
        if xlim := kwargs.get("xlim", []):
            plt.xlim(xlim)
        if ylim := kwargs.get("ylim", []):
            plt.ylim(ylim)
        ax.set_xlabel(self.labels[self.xlabel], fontdict={"size": 30})
        ax.set_ylabel(self.labels[self.ylabel], fontdict={"size": 30})
        plt.tick_params(labelsize=20, pad=8)
        fig.tight_layout()
        plot_savefig(self)
        plt.show()

    @classmethod
    def check_phase(
        cls,
        point: PhasePointData,
        point_other: Optional[PhasePointData] = None,
        ignore: Optional[Union[str, list[str]]] = None,
    ):
        if ignore is None:
            ignore = []
        if point_other is None:
            return (
                not isinstance(point, PhasePointData)
                or "unknown" in point.phase
                or point.phase in ignore
            )
        else:
            return (
                cls.check_phase(point=point, ignore=ignore)
                or cls.check_phase(point=point_other, ignore=ignore)
                or point.phase == point_other.phase
            )

    def boundary_detect(
        self,
        comp_res: CompareResult,
        mode: DetectionMode = DetectionMode.BOTH,
        ignore: Optional[Union[str, list[str]]] = None,
        **kwargs,
    ):
        if isinstance(ignore, str):
            ignore = [ignore]
        xys = []
        mat = comp_res.mat.copy()
        df = comp_res.df.copy()
        if mode == DetectionMode.BOTH or mode == DetectionMode.HORIZONTAL:
            for idx in range(mat.shape[0]):
                line = mat[idx]
                line = line[line != 0]
                for ele_idx in range(len(line) - 1):
                    ele_left_1: PhasePointData = line[ele_idx]
                    ele_right_1: PhasePointData = line[ele_idx + 1]
                    if self.check_phase(ele_left_1, ele_right_1, ignore):
                        continue
                    ele_left_2: pd.DataFrame = self.query_point(
                        data=df,
                        xval=ele_left_1.x,
                        yval=ele_left_1.y,
                        phase=ele_right_1.phase,
                    )
                    ele_right_2: pd.DataFrame = self.query_point(
                        data=df,
                        xval=ele_right_1.x,
                        yval=ele_right_1.y,
                        phase=ele_left_1.phase,
                    )

                    if len(ele_left_2) != 1 or len(ele_right_2) != 1:
                        continue
                    x0, _ = self.cross_point(
                        ele_left_1.x,
                        ele_left_1.value,
                        ele_right_2[self.xlabel].values[0],
                        ele_right_2.freeE.values[0],
                        ele_right_1.x,
                        ele_right_1.value,
                        ele_left_2[self.xlabel].values[0],
                        ele_left_2.freeE.values[0],
                    )
                    if [x0, ele_left_1.y] not in xys:
                        xys.append([x0, ele_left_1.y])
        if mode == DetectionMode.BOTH or mode == DetectionMode.VERTICAL:
            for idx in range(mat.shape[1]):
                line = mat[:, idx]
                line = line[line != 0]
                for ele_idx in range(len(line) - 1):
                    ele_left_1: PhasePointData = line[ele_idx]
                    ele_right_1: PhasePointData = line[ele_idx + 1]
                    if self.check_phase(ele_left_1, ele_right_1, ignore):
                        continue

                    ele_left_2: pd.DataFrame = self.query_point(
                        data=df,
                        xval=ele_left_1.x,
                        yval=ele_left_1.y,
                        phase=ele_right_1.phase,
                    )
                    ele_right_2: pd.DataFrame = self.query_point(
                        data=df,
                        xval=ele_right_1.x,
                        yval=ele_right_1.y,
                        phase=ele_left_1.phase,
                    )

                    if len(ele_left_2) != 1 or len(ele_right_2) != 1:
                        continue

                    y0, _ = self.cross_point(
                        ele_left_1.y,
                        ele_left_1.value,
                        float(ele_right_2[self.ylabel].values),
                        float(ele_right_2.freeE.values),
                        ele_right_1.y,
                        ele_right_1.value,
                        float(ele_left_2[self.ylabel].values),
                        float(ele_left_2.freeE.values),
                    )
                    if [ele_left_1.x, y0] not in xys:
                        xys.append([ele_left_1.x, y0])

        return np.array(xys)

    @staticmethod
    def draw_line(
        xys: np.ndarray,
        ax: Optional[plt.Axes] = None,
        annotation: Optional[dict] = None,
        inverse: bool = False,
    ):
        if annotation is None:
            annotation = {}
        if cuts := annotation.get("cuts", None):
            xys = xys[cuts:]
        if cute := annotation.get("cute", None):
            xys = xys[:cute]

        if inverse:
            xs = xys[:, 1].copy()
            ys = xys[:, 0].copy()
        else:
            xs = xys[:, 0].copy()
            ys = xys[:, 1].copy()

        coefs = Chebyshev.fit(xs, ys, annotation.get("order", 3))

        new_x = np.linspace(
            xs.min() - annotation.get("adds", 0),
            xs.max() + annotation.get("adde", 0),
            300,
        )
        new_y = coefs(new_x)  # type: ignore

        if shrinks := annotation.get("shrinks", None):
            new_x = new_x[shrinks:]
            new_y = new_y[shrinks:]
        if shrinke := annotation.get("shrinke", None):
            new_x = new_x[:shrinke]
            new_y = new_y[:shrinke]

        if inverse:
            new_x, new_y = new_y, new_x

        if ax:
            ax.plot(
                new_x,
                new_y,
                c="k",
                lw=3,
                ls=annotation.get("ls", "-"),
                alpha=annotation.get("alpha", 1),
            )
        else:
            plt.plot(new_x, new_y, c="k", lw=3)

    def extract_edge(
        self,
        mat: np.ndarray,
        mode: DetectionMode = DetectionMode.INTERP,
        factor: float = 0.5,
        axis: int = 1,
        ignore: Optional[Union[str, list[str]]] = None,
        **kwargs,
    ):
        if isinstance(ignore, str):
            ignore = [ignore]
        edge_data = []
        assert axis <= 1
        # var_label = 'y' if axis else 'x'
        # const_label = 'x' if axis else 'y'
        for col_idx in range(mat.shape[axis]):
            col = mat[:, col_idx] if axis == 1 else mat[col_idx, :]
            col = col[col != 0]
            for ele_idx in range(len(col) - 1):
                ele_1: PhasePointData = col[ele_idx]
                ele_2: PhasePointData = col[ele_idx + 1]
                if self.check_phase(ele_1, ele_2, ignore):
                    continue

                def calculate_intersection_point_xy_plane(point1, point2):
                    # 提取点的坐标
                    x1, y1, z1 = point1
                    x2, y2, z2 = point2

                    # 如果两点的z坐标相等，则直线与x-y平面平行，没有交点
                    if z1 == z2:
                        return [x1, y1, 0]

                    # 如果两点的x坐标相等，则直线与y轴平行，交点在y轴上
                    if x1 == x2:
                        x_intersection = x1
                        y_intersection = y1 + (0 - z1) / (z2 - z1) * (y2 - y1)
                        return x_intersection, y_intersection, 0

                    # 如果两点的y坐标相等，则直线与x轴平行，交点在x轴上
                    if y1 == y2:
                        x_intersection = x1 + (0 - z1) / (z2 - z1) * (x2 - x1)
                        y_intersection = y1
                        return x_intersection, y_intersection, 0

                    # 计算直线的斜率
                    slope_x = (x2 - x1) / (z2 - z1)
                    slope_y = (y2 - y1) / (z2 - z1)

                    # 计算直线与x-y平面的交点坐标
                    x_intersection = x1 + (0 - z1) / slope_x
                    y_intersection = y1 + (0 - z1) / slope_y

                    # 返回交点的坐标
                    return x_intersection, y_intersection, 0

                if mode == DetectionMode.INTERP:
                    x, y, _ = calculate_intersection_point_xy_plane(
                        point1=(ele_1.x, ele_1.y, 0),
                        point2=(ele_2.x, ele_2.y, ele_2.value - ele_1.value),
                    )
                    edge_data.append([x, y])
                elif mode == DetectionMode.MIX:
                    edge_data.append(
                        [
                            ele_1.x * factor + ele_2.x * (1 - factor),
                            ele_1.y * factor + ele_2.y * (1 - factor),
                        ]
                    )
                else:
                    raise NotImplementedError(mode)

        return np.array(edge_data)
