import os
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Optional, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.figure import Figure
from matplotlib.ticker import AutoMinorLocator, MultipleLocator
from numpy.polynomial import Chebyshev
from scipy.io import loadmat


@dataclass()
class CompareResult:
    df: pd.DataFrame
    plot_dict: dict
    mat: np.ndarray
    fig: Optional[Figure] = None
    ax: Optional[plt.Axes] = None


class DetectionMode(str, Enum):
    HORIZONTAL = "hori"
    VERTICAL = "vert"
    BOTH = "both"


@dataclass()
class PointInfo:
    phase: str
    x: float
    y: float
    freeEnergy: float


class PhaseDiagram:
    def __init__(
        self, xlabel: str, ylabel: str, color_dict: Optional[dict] = None, label_dict: Optional[dict] = None, **kwargs
    ) -> None:
        if label_dict is None:
            label_dict = {
                "tau": r"$\tau$",
                "ksi": r"$\xi$",
                "volH": r"$\phi_{\rm{H}}$",
                "fA": r"$f_{\rm{A}}$",
                "chiN": r"$\chi \rm{N}$",
            }
        if color_dict is None:
            color_dict = {
                "C4": "r",
                "Crect": "dodgerblue",
                "L": "deeppink",
                "DG": "orange",
                "iHPa": "blue",
                "SG": "magenta",
                "SC": "goldenrod",
                "C6": "tan",
                "C3": "m",
                "HCP": "crimson",
                "FCC": "yellowgreen",
                "PL": "darkorchid",
                "BCC": "limegreen",
                "sdgn": "teal",
                "O70": "crimson",
                "unknown": "k",
                "Disorder": "y",
            }
        self.color_dict = color_dict
        self.color_dict.update(kwargs.get("color", {}))
        self.label_dict = label_dict
        self.label_dict.update(kwargs.get("label", {}))
        self.xlabel = xlabel
        self.ylabel = ylabel
        pass

    def query_point(
        self,
        df: pd.DataFrame,
        xval: Optional[Union[float, str, int]] = None,
        yval: Optional[Union[float, str, int]] = None,
        **kwargs,
    ):
        res = df.copy()
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
    def checkin(phase, candidates):
        if len(phase) == 0:
            return False
        for i in phase:
            if i in candidates:
                return True
        return False

    def data(self, path, **kwargs):
        dropset = kwargs.get("dropset", ["lx", "ly", "lz", "phase", "freeE"])
        div = kwargs.get("div", 1)
        div_var = kwargs.get("div_var", "chiN")
        acc = kwargs.get("acc", 3)

        df = pd.read_csv(path)
        if dropset:
            df = df.drop_duplicates(subset=dropset)
        df["lylz"] = np.around(df["ly"] / df["lz"], acc)
        df["lxly"] = np.around(df["lx"] / df["ly"], acc)
        try:
            df[div_var] = df[div_var] / div
        except KeyError:
            pass
        return df

    def compare(self, path: Union[str, Path], plot: bool = True, acc: int = 3, **kwargs):
        df = self.data(path=path, acc=acc)
        print(f"Include phase: {set(df['phase'].values)}")

        plot_dict = dict()
        y_set = np.sort(df[self.ylabel].unique())
        x_set = np.sort(df[self.xlabel].unique())

        exclude = kwargs.get("exclude", [])

        mat = np.zeros((len(y_set), len(x_set)), dtype=PointInfo)
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
                        if lylz == np.around(np.sqrt(3), acc) or lzly == np.around(1 / np.sqrt(3), acc):
                            phase = min_label[0]
                    elif self.checkin(["iHPa"], min_label):
                        if lxly == np.around(np.sqrt(3), acc) or lylx == np.around(1 / np.sqrt(3), acc):
                            phase = "iHPa"
                        elif lylz == np.around(np.sqrt(2), acc) or lxly == 1:
                            phase = "SC"
                    elif "PL" in min_label:
                        if lxly == np.around(np.sqrt(3), acc) or lylx == np.around(1 / np.sqrt(3), acc):
                            phase = "PL"
                    elif self.checkin(["L", "Disorder", "O70"], min_label):
                        phase = min_label[0]
                    elif self.checkin(["SC", "SG", "DG", "BCC", "FCC", "sdgn"], min_label):
                        if lxly == 1 and lylz == 1:
                            phase = min_label[0]
                    else:
                        phase = "_".join([phase, min_label[0]])
                # mat[i][j] = [phase, x, y, freeE]
                mat[i][j] = PointInfo(phase=phase, x=x, y=y, freeEnergy=freeE)
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
                ax.scatter(
                    value[self.xlabel],
                    value[self.ylabel],
                    c=self.color_dict.get(key, "k"),
                    label=key,
                )
            ax.tick_params(top="on", right="on", which="both")
            ax.tick_params(which="both", width=2, length=4, direction="in")
            ax.xaxis.set_minor_locator(AutoMinorLocator(5))
            ax.yaxis.set_minor_locator(AutoMinorLocator(5))
            ax.set_xlabel(self.label_dict.get(self.xlabel, self.xlabel), fontdict={"size": 20})
            ax.set_ylabel(self.label_dict.get(self.ylabel, self.ylabel), fontdict={"size": 20})
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
        return (D1 / D, D2 / D)

    @staticmethod
    def de_unknown(comp_res: CompareResult):
        mat = comp_res.mat
        for i in range(mat.shape[0]):
            for j in range(mat.shape[1]):
                mat[i][j].phase = mat[i][j].phase.lstrip("unknown_")
        comp_res.mat = mat
        return comp_res

    def scan(self, folder: Union[str, Path], ann_dict: dict, **kwargs):
        if isinstance(folder, str):
            folder = Path(folder)

        # filelist = os.listdir(folder)
        filelist = list(folder.glob("*.csv"))
        fig, ax = plt.subplots(figsize=kwargs.get("figsize", (8, 6)))
        skip = kwargs.get("skip", [])
        extract = kwargs.get("extract", [])
        inverse_lst = kwargs.get("inverse", [])
        deunknown = kwargs.get("deunknown", [])

        for filename in filelist:
            # if not filename.endswith('.csv'):
            #     continue
            # tmp_name = os.path.splitext(filename)[0]
            tmp_name = filename.stem
            tmp_dict = ann_dict.get(tmp_name, {})
            print(tmp_name, end="\t")
            if tmp_name in skip:
                print("Skip...")
                continue
            else:
                comp_res = self.compare(path=filename, plot=False, acc=tmp_dict.get("acc", 2))

            if tmp_name in deunknown or deunknown == "All":
                comp_res = self.de_unknown(comp_res)

            if tmp_name in extract:
                tmp_xys = self.extract_edge(comp_res.mat, **tmp_dict)
            else:
                # tmp_xys = self.boundary(comp_res.df, comp_res.mat, mode=["hori", "ver"])
                tmp_xys = self.boundary_detect(comp_res)

            if tmp_name in inverse_lst:
                tmp_xys = tmp_xys[np.argsort(tmp_xys[:, 1])]
                self.draw_line(tmp_xys, ax=ax, tmp_ann_dict=tmp_dict, inverse=True)
            else:
                tmp_xys = tmp_xys[np.argsort(tmp_xys[:, 0])]
                self.draw_line(tmp_xys, ax=ax, tmp_ann_dict=tmp_dict)

        if phase_name := ann_dict.get("phase_name", False):
            for key, value in phase_name.items():
                ax.text(value[0], value[1], key, fontsize=20, color="k")

        if phase_name_arrow := ann_dict.get("phase_name_with_arrow", {}):
            for key, value in phase_name_arrow.items():
                ax.annotate(
                    key,
                    xy=value[:2],
                    xycoords="data",
                    xytext=value[2:4],
                    textcoords="data",
                    weight="bold",
                    color="k",
                    fontsize=20,
                    arrowprops=dict(arrowstyle="->", connectionstyle="arc3", color="k", lw=2),
                )

        save_path = kwargs.get("path", "")
        if mat_path := kwargs.get("mat_path", None):
            wrongline = loadmat(mat_path)["origin_wrong"]
            self.draw_line(
                wrongline,
                ax=ax,
                tmp_ann_dict=kwargs.get("mat", {"adde": 0.001, "ls": ":", "shrinks": 7, "alpha": 0.5}),
            )

        plt.margins(0)
        ax.xaxis.set_minor_locator(AutoMinorLocator(5))
        ax.yaxis.set_minor_locator(AutoMinorLocator(5))
        if ymajor := kwargs.get("ymajor", 0):
            ax.yaxis.set_major_locator(MultipleLocator(ymajor))
        ax.set_xlabel(self.label_dict[self.xlabel], fontdict={"size": 30})
        ax.set_ylabel(self.label_dict[self.ylabel], fontdict={"size": 30})
        plt.tick_params(labelsize=20, pad=8)
        fig.tight_layout()
        plt.show()
        if save_path:
            plt.savefig(save_path, dpi=200)

    def boundary_detect(self, comp_res: CompareResult, mode: DetectionMode = DetectionMode.BOTH):
        xys = []
        mat = comp_res.mat.copy()
        df = comp_res.df.copy()
        if mode == DetectionMode.BOTH or mode == DetectionMode.HORIZONTAL:
            for idx in range(mat.shape[0]):
                line = mat[idx]
                line = line[line != 0]
                for ele_idx in range(len(line) - 1):
                    ele_left_1: PointInfo = line[ele_idx]
                    ele_right_1: PointInfo = line[ele_idx + 1]
                    if (
                        ele_left_1.phase == ele_right_1.phase
                        or ("unknown" in ele_left_1.phase)
                        or ("unknown" in ele_right_1.phase)
                    ):
                        continue

                    ele_left_2: pd.DataFrame = self.query_point(
                        df=df, xval=ele_left_1.x, yval=ele_left_1.y, phase=ele_right_1.phase
                    )
                    ele_right_2: pd.DataFrame = self.query_point(
                        df=df, xval=ele_right_1.x, yval=ele_right_1.y, phase=ele_left_1.phase
                    )

                    if len(ele_left_2) != 1 or len(ele_right_2) != 1:
                        continue
                    x0, _ = self.cross_point(
                        ele_left_1.x,
                        ele_left_1.freeEnergy,
                        ele_right_2[self.xlabel].values[0],
                        ele_right_2.freeE.values[0],
                        ele_right_1.x,
                        ele_right_1.freeEnergy,
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
                    ele_left_1: PointInfo = line[ele_idx]
                    ele_right_1: PointInfo = line[ele_idx + 1]
                    if (
                        ele_left_1.phase == ele_right_1.phase
                        or ("unknown" in ele_left_1.phase)
                        or ("unknown" in ele_right_1.phase)
                    ):
                        continue

                    ele_left_2: pd.DataFrame = self.query_point(
                        df=df, xval=ele_left_1.x, yval=ele_left_1.y, phase=ele_right_1.phase
                    )
                    ele_right_2: pd.DataFrame = self.query_point(
                        df=df, xval=ele_right_1.x, yval=ele_right_1.y, phase=ele_left_1.phase
                    )

                    if len(ele_left_2) != 1 or len(ele_right_2) != 1:
                        continue

                    y0, _ = self.cross_point(
                        ele_left_1.y,
                        ele_left_1.freeEnergy,
                        ele_right_2[self.ylabel].values[0],
                        ele_right_2.freeE.values[0],
                        ele_right_1.y,
                        ele_right_1.freeEnergy,
                        ele_left_2[self.ylabel].values[0],
                        ele_left_2.freeE.values[0],
                    )
                    if [ele_left_1.x, y0] not in xys:
                        xys.append([ele_left_1.x, y0])

        return np.array(xys)

    @staticmethod
    def draw_line(xys, ax=None, tmp_ann_dict=None, inverse: bool = False):
        if tmp_ann_dict is None:
            tmp_ann_dict = {}
        if cuts := tmp_ann_dict.get("cuts", None):
            xys = xys[cuts:]
        if cute := tmp_ann_dict.get("cute", None):
            xys = xys[:cute]

        if inverse:
            xs = xys[:, 1].copy()
            ys = xys[:, 0].copy()
        else:
            xs = xys[:, 0].copy()
            ys = xys[:, 1].copy()

        try:
            coefs = Chebyshev.fit(xs, ys, tmp_ann_dict.get("order", 3))
        except BaseException:
            print(xs, ys)
            return

        new_x = np.linspace(
            xs.min() - tmp_ann_dict.get("adds", 0),
            xs.max() + tmp_ann_dict.get("adde", 0),
            300,
        )
        new_y = coefs(new_x)

        if shrinks := tmp_ann_dict.get("shrinks", None):
            new_x = new_x[shrinks:]
            new_y = new_y[shrinks:]
        if shrinke := tmp_ann_dict.get("shrinke", None):
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
                ls=tmp_ann_dict.get("ls", "-"),
                alpha=tmp_ann_dict.get("alpha", 1),
            )
        else:
            plt.plot(new_x, new_y, c="k", lw=3)

    @staticmethod
    def extract_edge(
        mat: np.ndarray,
        factor: float = 0.5,
        axis: int = 1,
        **kwargs,
    ):
        edge_data = []
        assert axis <= 1
        # var_label = 'y' if axis else 'x'
        # const_label = 'x' if axis else 'y'
        for col_idx in range(mat.shape[axis]):
            col = mat[:, col_idx] if axis else mat[col_idx, :]
            col = col[col != 0]
            for ele_idx in range(len(col) - 1):
                ele_1 = col[ele_idx]
                ele_2 = col[ele_idx + 1]
                if ele_1.phase == ele_2.phase:
                    continue
                edge_data.append([ele_1.x * factor + ele_2.x * (1 - factor), ele_1.y * factor + ele_2.y * (1 - factor)])

        edge_data = np.array(edge_data)
        return edge_data
