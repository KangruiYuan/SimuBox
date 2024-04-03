from collections import ChainMap
from pathlib import Path
from typing import Any, Optional, Union, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from numpy.polynomial import Chebyshev
from scipy.io import loadmat

from .plotter import plot_locators, plot_savefig, plot_legend, generate_colors
from ..schema import (
    DetectionMode,
    PhaseCompareResult,
    Point,
    CommonLabels,
    PathLike,
    Operation,
    RealNum,
    ColorMode,
)
from ..toolkits import read_csv

__all__ = ["PHASE_PLOT_CONFIG", "PhaseDiagram"]

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
        """
        初始化对象。

        :param path: 存放一系列相图文件(.csv)的文件夹路径。
        :param xlabel: 相图横轴数值对应的列名。
        :param ylabel: 相图纵轴数值对应的列名。
        :param colors: 各相结构在绘制时的颜色。
        :param labels: 各标签的姓名映射，如{"tau": r"$\tau$"}，即在渲染时将tau渲染为τ
        """
        self.path = Path(path)
        self.colors = colors if colors is not None else {}
        self.labels = (
            ChainMap(labels, CommonLabels) if labels is not None else CommonLabels
        )
        self.xlabel = xlabel
        self.ylabel = ylabel

    def query(
        self,
        data: Union[pd.DataFrame, PathLike],
        x: Optional[Union[float, str, int]] = None,
        y: Optional[Union[float, str, int]] = None,
        **kwargs,
    ):
        """
        对表格数据进行快速筛选

        :param data:
        :param x:
        :param y:
        :param kwargs:
        :return:
        """
        if isinstance(data, pd.DataFrame):
            res = data.copy()
        else:
            res = read_csv(data)
        if x:
            res = res[res[self.xlabel] == x]
        if y:
            res = res[res[self.ylabel] == y]

        columns = res.columns
        for key, val in kwargs.items():
            if key in columns:
                res = res[res[key] == val]
        return res

    @staticmethod
    def checkin(phases: Sequence, candidates: Sequence):
        """
        检查相结构是否属于备选结构。

        :param phases:
        :param candidates:
        :return:
        """
        if len(phases) == 0:
            return False
        for i in phases:
            if i in candidates:
                return True
        return False

    def compare(
        self,
        path: PathLike,
        plot: bool = True,
        accuracy: int = 3,
        xlabel: Optional[str] = None,
        ylabel: Optional[str] = None,
        verbose: bool = True,
        operation: Optional[Union[Operation, Sequence[Operation]]] = None,
        subset: Optional[Union[str, Sequence[str]]] = ("phase", "freeE"),
        figsize: Union[Sequence[int]] = (8, 6),
        interactive: bool = True,
        **kwargs,
    ):
        if not isinstance(path, Path):
            path = Path(path)
        if not path.is_file():
            path = self.path / path
            if not path.is_file():
                if path.with_suffix(".csv").is_file():
                    path = path.with_suffix(".csv")
                elif path.with_suffix(".xlsx").is_file():
                    path = path.with_suffix(".xlsx")
                else:
                    raise ValueError(f"请输入正确的文件名或者路径信息。{str(path)}不存在")

        df = read_csv(
            path=path, accuracy=accuracy, operation=operation, subset=subset, **kwargs
        )
        if verbose:
            print(f"包括相结构: {set(df['phase'].values)}")

        phase_map = dict()
        xlabel = xlabel if xlabel is not None else self.xlabel
        ylabel = ylabel if ylabel is not None else self.ylabel
        y_set = np.sort(df[ylabel].unique())
        x_set = np.sort(df[xlabel].unique())

        exclude = kwargs.get("exclude", ())
        sqrt3 = np.around(np.sqrt(3), decimals=accuracy)
        sqrt2 = np.around(np.sqrt(2), decimals=accuracy)

        mat = np.zeros((len(y_set), len(x_set)), dtype=Point)
        for i, y in enumerate(y_set):
            for j, x in enumerate(x_set):
                phase = "unknown"
                temp_data = df[(df[ylabel] == y) & (df[xlabel] == x)]
                min_data = temp_data[temp_data["freeE"] == temp_data["freeE"].min()]
                if len(min_data) == 0:
                    continue
                freeE = min_data["freeE"].min()
                min_label = min_data["phase"].unique()
                lxly = min_data["lxly"].unique()
                lylx = np.around(1 / lxly, accuracy)
                lylz = min_data["lylz"].unique()
                lzly = np.around(1 / lylz, accuracy)

                if self.checkin(min_label, exclude):
                    phase = min_label[0]
                elif "L" in min_label:
                    phase = "L"
                elif self.checkin(min_label, ["C4", "Crect"]):
                    if any(lylz == 1) or any(lylx == 1):
                        phase = "C4"
                    else:
                        phase = "Crect"
                elif len(min_label) == 1:
                    if self.checkin(min_label, ["C6", "C3"]):
                        if sqrt3 in [lylz, lzly, lxly, lylx]:
                            phase = min_label[0]
                    elif self.checkin(min_label, ["iHPa"]):
                        if sqrt3 in [lylz, lzly, lxly, lylx]:
                            phase = "iHPa"
                        elif sqrt2 in [lylz, lzly, lxly, lylx]:
                            phase = "SC"
                    elif "PL" in min_label:
                        if sqrt3 in [lylz, lzly, lxly, lylx]:
                            phase = "PL"
                    elif self.checkin(min_label, ["L", "Disorder", "O70"]):
                        phase = min_label[0]
                    elif self.checkin(
                        min_label, ["SC", "SG", "DG", "BCC", "FCC", "sdgn"]
                    ):
                        if lxly == 1 and lylz == 1:
                            phase = min_label[0]
                    else:
                        phase = "_".join([phase, min_label[0]])
                mat[i][j] = Point(label=phase, x=x, y=y, value=freeE)
                if phase in phase_map:
                    for attr, val in zip(
                        [xlabel, ylabel, "freeE", "lylz", "lxly"],
                        [x, y, freeE, lylz, lxly],
                    ):
                        phase_map[phase][attr].append(val)
                else:
                    phase_map[phase] = {
                        xlabel: [x],
                        ylabel: [y],
                        "freeE": [freeE],
                        "lylz": [lylz],
                        "lxly": [lxly],
                    }

        comp_res = PhaseCompareResult(data=df, phase_map=phase_map, mat=mat)

        if plot:
            fig, ax = plt.subplots(1, 1, figsize=figsize)
            for key, value in phase_map.items():
                if key not in self.colors:
                    self.colors[key] = generate_colors(mode=ColorMode.HEX)[0]
                ax.scatter(
                    value[xlabel],
                    value[ylabel],
                    c=self.colors[key],
                    label=key,
                )
            ax.tick_params(top="on", right="on", which="both")
            ax.tick_params(which="both", width=2, length=4, direction="in")
            plot_locators(**kwargs)
            ax.set_xlabel(self.labels.get(xlabel, xlabel), fontdict={"size": 20})
            ax.set_ylabel(self.labels.get(ylabel, ylabel), fontdict={"size": 20})
            plot_legend(
                legend=kwargs.get(
                    "legend",
                    {
                        "mode": "outside",
                        "loc": "upper left",
                        "bbox_to_anchor": (0.98, 0.8),
                    },
                )
            )
            fig.tight_layout()
            comp_res.fig = fig
            comp_res.ax = ax
            if not interactive:
                plt.show()

        return comp_res

    @staticmethod
    def cross_point(
        x1: RealNum,
        y1: RealNum,
        x2: RealNum,
        y2: RealNum,
        x3: RealNum,
        y3: RealNum,
        x4: RealNum,
        y4: RealNum,
    ):
        """
        两点确定直线，计算两条直线之间的交点。
        (x1, y1), (x2, y2)为第一条直线的两点。
        (x3, y3), (x4, y4)为第二条直线的两点。
        y一般对应自由能的数值。

        :param x1:
        :param y1:
        :param x2:
        :param y2:
        :param x3:
        :param y3:
        :param x4:
        :param y4:
        :return:
        """
        b1 = (y2 - y1) * x1 + (x1 - x2) * y1
        b2 = (y4 - y3) * x3 + (x3 - x4) * y3
        D = (x2 - x1) * (y4 - y3) - (x4 - x3) * (y2 - y1)
        D1 = b2 * (x2 - x1) - b1 * (x4 - x3)
        D2 = b2 * (y2 - y1) - b1 * (y4 - y3)
        return D1 / D, D2 / D

    @staticmethod
    def de_unknown(comp_res: PhaseCompareResult):
        """
        将相结构名称前缀中的unknown去掉。

        :param comp_res:
        :return:
        """
        mat = comp_res.mat
        for i in range(mat.shape[0]):
            for j in range(mat.shape[1]):
                try:
                    new_label = mat[i][j].label.lstrip("unknown_")
                    if new_label == "":
                        mat[i][j] = 0
                    else:
                        mat[i][j].label = new_label
                except AttributeError:
                    continue
        comp_res.mat = mat
        return comp_res

    def scan(
        self,
        path: Optional[Union[str, Path]] = None,
        filetype: Sequence[str] = (".csv"),
        figsize: Union[Sequence[int]] = (8, 8),
        skip: Sequence[str] = (),
        extract: Sequence[str] = (),
        inverse: Sequence[str] = (),
        deunknown: Sequence[str] = (),
        annotation: Optional[dict] = None,
        verbose: bool = True,
        **kwargs,
    ):
        if annotation is None:
            annotation = {}

        if path is not None:
            if isinstance(path, str):
                path = Path(path)
                assert path.is_dir()
        else:
            path = self.path

        filelist = []
        for ft in filetype:
            filelist += list(path.glob("*" + ft))

        fig, ax = plt.subplots(figsize=figsize)
        for filename in filelist:
            tmp_name = filename.stem
            tmp_dict: dict = annotation.get(tmp_name, {})
            if verbose:
                print(tmp_name, end="\t")
            if tmp_name in skip:
                if verbose:
                    print("Skip...")
                continue
            else:
                comp_res = self.compare(
                    path=filename, plot=False, accuracy=tmp_dict.get("acc", 2)
                )

            if tmp_name in deunknown or deunknown == "All":
                comp_res = self.de_unknown(comp_res)

            if tmp_name in extract:
                # print("Extracting.")
                tmp_xys = self.extract_edge(comp_res.mat, **tmp_dict)
            else:
                tmp_xys = self.boundary_detect(comp_res, **tmp_dict)

            try:
                if tmp_name in inverse:
                    tmp_xys = tmp_xys[np.argsort(tmp_xys[:, 1])]
                    self.draw_line(tmp_xys, ax=ax, annotation=tmp_dict, inverse=True)
                else:
                    tmp_xys = tmp_xys[np.argsort(tmp_xys[:, 0])]
                    self.draw_line(tmp_xys, ax=ax, annotation=tmp_dict)
            except IndexError as ie:
                print(tmp_xys)
                raise ie

        if phase_name := annotation.get("phase_name", ()):
            for name, _x, _y in phase_name:
                ax.text(_x, _y, name, fontsize=20, color="k")

        if phase_name_arrow := annotation.get("phase_name_arrow", ()):
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
        plot_savefig(obj=self, **kwargs)
        plt.show()

    @classmethod
    def check_phase(
        cls,
        point: Point,
        point_other: Optional[Point] = None,
        ignore: Optional[Union[str, list[str]]] = None,
    ):
        if ignore is None:
            ignore = []
        if point_other is None:
            return (
                not isinstance(point, Point)
                or "unknown" in point.label
                or point.label in ignore
            )
        else:
            return (
                cls.check_phase(point=point, ignore=ignore)
                or cls.check_phase(point=point_other, ignore=ignore)
                or point.label == point_other.label
            )

    def boundary_detect(
        self,
        comp_res: PhaseCompareResult,
        mode: DetectionMode = DetectionMode.BOTH,
        ignore: Optional[Union[str, list[str]]] = None,
        **kwargs,
    ):
        if isinstance(ignore, str):
            ignore = [ignore]
        xys = []
        mat = comp_res.mat.copy()
        df = comp_res.data.copy()
        if mode == DetectionMode.BOTH or mode == DetectionMode.HORIZONTAL:
            for idx in range(mat.shape[0]):
                line = mat[idx]
                line = line[line != 0]
                for ele_idx in range(len(line) - 1):
                    ele_left_1: Point = line[ele_idx]
                    ele_right_1: Point = line[ele_idx + 1]
                    if self.check_phase(ele_left_1, ele_right_1, ignore):
                        continue
                    ele_left_2: pd.DataFrame = self.query(
                        data=df,
                        x=ele_left_1.x,
                        y=ele_left_1.y,
                        phase=ele_right_1.label,
                    )
                    ele_right_2: pd.DataFrame = self.query(
                        data=df,
                        x=ele_right_1.x,
                        y=ele_right_1.y,
                        phase=ele_left_1.label,
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
                    ele_left_1: Point = line[ele_idx]
                    ele_right_1: Point = line[ele_idx + 1]
                    if self.check_phase(ele_left_1, ele_right_1, ignore):
                        continue

                    ele_left_2: pd.DataFrame = self.query(
                        data=df,
                        x=ele_left_1.x,
                        y=ele_left_1.y,
                        phase=ele_right_1.label,
                    )
                    ele_right_2: pd.DataFrame = self.query(
                        data=df,
                        x=ele_right_1.x,
                        y=ele_right_1.y,
                        phase=ele_left_1.label,
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
                ele_1: Point = col[ele_idx]
                ele_2: Point = col[ele_idx + 1]
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
