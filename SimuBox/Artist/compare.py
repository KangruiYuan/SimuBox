from copy import deepcopy
from pathlib import Path
from typing import Optional, Union, Literal

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from cycler import cycler

from collections import ChainMap

from ..toolkits import read_csv
from ..schema import AbsCommonLabels, DiffCommonLabels, LineCompareResult, Line, PathLike
from .plotter import plot_trans, plot_legend, plot_locators, plot_savefig

COMPARE_PLOT_CONFIG = {
    "font.family": "Times New Roman",
    # "font.family":'serif',
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


_STYLE_REF = cycler(
    color=list("rbm")
    + [
        "indianred",
        "tomato",
        "chocolate",
        "olivedrab",
        "teal",
        "deepskyblue",
        "darkviolet",
    ]
) + cycler(marker=list("sXvoP*D><p"))

_STYLE_ABS = cycler(
    color=list("krbm")
    + ["indianred", "tomato", "chocolate", "olivedrab", "teal", "deepskyblue"]
) + cycler(marker=list("osXvP*D><p"))


class CompareJudger:
    def __init__(
        self,
        path: Union[str, Path],
        div: Union[int, float] = 1,
        diff_labels: Optional[dict[str, str]] = None,
        abs_labels: Optional[dict[str, str]] = None,
        **kwargs,
    ):
        self.path = path if isinstance(path, Path) else Path(path)
        self.div = div
        self.diff_labels = (
            ChainMap(diff_labels, DiffCommonLabels) if diff_labels is not None else DiffCommonLabels
        )
        self.abs_labels = (
            ChainMap(abs_labels, AbsCommonLabels) if abs_labels is not None else AbsCommonLabels
        )

    def data(self, **kwargs):
        return read_csv(self.path, **kwargs)

    def diff_comparison(
        self,
        base: str,
        others: Union[str, list[str]],
        xlabel: str,
        ylabel: Union[str, list[str]],
        horiline: Literal["all", "mask", None] = "all",
        save: Optional[Union[bool, PathLike]] = True,
        data: Optional[pd.DataFrame] = None,
        interactive: bool = True,
        fontsize:int = 20,
        **kwargs,
    ):

        lines = []
        data = self.data(**kwargs) if data is None else data
        data = data.sort_values(by=xlabel)

        if isinstance(ylabel, list):
            y_name = kwargs.get("y_name", "") or "TMP"
            data[y_name] = np.sum(data[ylabel].values, axis=0)
            print(f'标签{"+".join(ylabel)}总名称指定为{y_name}')
            ylabel = y_name

        fig = plt.figure(figsize=kwargs.get("figsize", (8, 6)))
        ax = plt.gca()
        ax.set_prop_cycle(_STYLE_REF)

        base_data = data[data["phase"] == base]

        base_xticks = base_data[xlabel].values
        base_yticks = base_data[ylabel].values

        if isinstance(others, str):
            if others != "others":
                others = [others]
            else:
                others = set(data["phase"])
                others.remove(base)

        for o in others:
            o_data = data[data["phase"] == o]
            o_xticks = o_data[xlabel].values
            o_yticks = o_data[ylabel].values
            mask = np.in1d(o_xticks, base_xticks)
            o_xticks = o_xticks[mask]
            o_yticks = o_yticks[mask]

            inverse_mask = np.in1d(base_xticks, o_xticks)
            base_xticks_mask = base_xticks[inverse_mask]
            base_yticks_mask = base_yticks[inverse_mask]
            try:
                o_yticks = o_yticks - base_yticks_mask
            except ValueError as ve:
                print(f"In phase {o}:")
                print("o_xticks：", o_xticks)
                print("o_yticks：", o_yticks)
                print("base_yticks_mask：", base_yticks_mask)
                raise ve
            ax.plot(
                o_xticks,
                o_yticks,
                label=self.diff_labels.get(o, o),
                lw=2.5,
                markersize=8,
                alpha=1.0,
            )
            lines.append(
                Line(x=o_xticks, y=o_yticks, label=self.diff_labels.get(o, o))
            )

        if horiline:
            if horiline == "all":
                horiline_xticks = base_xticks
                horiline_yticks = np.zeros_like(horiline_xticks)
            elif horiline == "mask":
                rest = data[(data["phase"] != base) & data["phase"].isin(others)]
                rest_xticks = rest[xlabel].unique()
                mask = np.in1d(base_xticks, rest_xticks)
                horiline_xticks = base_xticks[mask]
                horiline_yticks = np.zeros_like(horiline_xticks)
            else:
                raise NotImplementedError(
                    f"Not implemented method for horiline: {horiline}"
                )

            ax.plot(
                horiline_xticks,
                horiline_yticks,
                label=self.diff_labels.get(base, base),
                lw=2.5,
                c="k",
                marker="o",
                markersize=8,
                alpha=0.8,
            )
            lines.append(
                Line(
                    x=horiline_xticks,
                    y=horiline_yticks,
                    label=self.diff_labels.get(base, base),
                )
            )

        plot_trans(**kwargs)
        plot_locators(ax=ax, **kwargs)
        plot_legend(**kwargs)

        plt.tick_params(axis="both", labelsize=kwargs.get("labelsize", fontsize), pad=8)
        plt.ylabel(self.diff_labels.get(ylabel, ylabel), fontsize=fontsize)
        plt.xlabel(self.diff_labels.get(xlabel, xlabel), fontsize=fontsize)

        if margin := kwargs.get("margin", (0.15, 0.15)):
            plt.margins(*margin)
        plt.tight_layout()
        suffix = kwargs.get("suffix", ylabel)
        # print(kwargs)
        if "suffix" in kwargs:
            del kwargs["suffix"]
        plot_savefig(self, prefix="diff", suffix=suffix, save=save, **kwargs)
        if interactive:
            plt.show()
        return LineCompareResult(
            fig=fig,
            ax=ax,
            lines=lines,
            xlabel=self.diff_labels.get(xlabel, xlabel),
            ylabel=self.diff_labels.get(ylabel, ylabel),
        )

    def abs_comparison(
        self,
        xlabel: str,
        ylabel: Union[str, list[str]],
        phases: Optional[Union[list[str], str]] = None,
        save: Optional[Union[PathLike, bool]] = True,
        data: Optional[pd.DataFrame] = None,
        interactive: bool = True,
        **kwargs,
    ):
        lines = []
        data = self.data(**kwargs) if data is None else data
        data = data.sort_values(by=xlabel)

        phases = (
            phases
            if phases is not None
            else data[kwargs.get("phase_col", "phase")].unique()
        )
        phases = [phases] if isinstance(phases, str) else phases
        print(f"当前考虑的phase为：{phases}")

        if isinstance(ylabel, list):
            y_name = kwargs.get("y_name", "") or "TMP"
            data[y_name] = np.sum(data[ylabel].values, axis=0)
            print(f'标签{"+".join(ylabel)}总名称指定为{y_name}')
            ylabel = y_name

        fig = plt.figure(figsize=kwargs.get("figsize", (8, 6)))
        ax = plt.gca()
        ax.set_prop_cycle(_STYLE_ABS)

        for p in phases:
            tmp = data[data.phase == p]
            x = tmp[xlabel].values
            y = tmp[ylabel].values
            ax.plot(
                x,
                y,
                label=self.abs_labels.get(p, p),
                lw=2.5,
                markersize=8,
            )
            lines.append(Line(x=x, y=y, label=self.abs_labels.get(p, p)))

        plt.xlabel(self.abs_labels.get(xlabel, xlabel), fontsize=30)
        plt.ylabel(self.abs_labels.get(ylabel, ylabel), fontsize=30)
        plt.tick_params(axis="both", labelsize=25, pad=8)

        plot_trans(**kwargs)
        plot_locators(ax=ax, **kwargs)
        plot_legend(**kwargs)

        if margin := kwargs.get("margin", (0.15, 0.15)):
            plt.margins(*margin)

        plt.tight_layout()
        plot_savefig(self, prefix="abs", suffix=ylabel, save=save)
        if interactive:
            plt.show()
        return LineCompareResult(
            fig=fig,
            ax=ax,
            lines=lines,
            xlabel=self.abs_labels.get(xlabel, xlabel),
            ylabel=self.abs_labels.get(ylabel, ylabel),
        )

    def multi_target_ref(
        self,
        base: str,
        other: Union[str, list[str]],
        xlabel: str,
        ylabels: Union[str, list[str]],
        ylabel_name: str,
        save: Optional[Union[PathLike, bool]] = True,
        data: Optional[pd.DataFrame] = None,
        interactive: bool = True,
        **kwargs,
    ):
        lines = []
        data = self.data(**kwargs) if data is None else data
        data = data.sort_values(by=xlabel)

        fig = plt.figure(figsize=kwargs.get("figsize", (8, 6)))
        ax = plt.gca()
        ax.set_prop_cycle(_STYLE_REF)

        ylabels = ylabels if isinstance(ylabels, list) else [ylabels]
        base_data = data[data["phase"] == base]
        base_xticks = base_data[xlabel].values

        for yl in ylabels:
            # self.ref_compare(base, other, xlabel, yl, ax=ax, horiline=False)
            base_yticks = base_data[yl].values
            o_data = data[data["phase"] == other]
            o_xticks = o_data[xlabel].values
            o_yticks = o_data[yl].values

            mask = np.in1d(o_xticks, base_xticks)
            o_xticks = o_xticks[mask]
            o_yticks = o_yticks[mask]

            inverse_mask = np.in1d(base_xticks, o_xticks)
            # base_xticks_mask = base_xticks[inverse_mask]
            base_yticks_mask = base_yticks[inverse_mask]
            o_yticks = o_yticks - base_yticks_mask
            ax.plot(
                o_xticks,
                o_yticks,
                label=self.diff_labels.get(yl, yl),
                lw=2.5,
                markersize=8,
                alpha=1.0,
            )
            lines.append(
                Line(x=o_xticks, y=o_yticks, label=self.diff_labels.get(yl, yl))
            )

        plot_trans(**kwargs)
        plot_locators(ax=ax, **kwargs)
        plot_legend(**kwargs)

        plt.tick_params(axis="both", labelsize=25, pad=8)
        plt.tick_params(axis="both", labelsize=25)
        plt.ylabel(self.diff_labels.get(ylabel_name, ylabel_name), fontsize=30)
        plt.xlabel(self.diff_labels.get(xlabel, xlabel), fontsize=30)

        if margin := kwargs.get("margin", (0.15, 0.15)):
            plt.margins(*margin)

        plt.tight_layout()
        plot_savefig(self, prefix="multi", suffix=ylabel_name, save=save, **kwargs)
        if interactive:
            plt.show()

        return LineCompareResult(
            fig=fig,
            ax=ax,
            lines=lines,
            xlabel=self.diff_labels.get(xlabel, xlabel),
            ylabel=self.diff_labels.get(ylabel_name, ylabel_name),
        )
