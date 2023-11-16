from copy import deepcopy
from pathlib import Path
from typing import Optional, Union

import matplotlib.pyplot as plt
import numpy as np
from cycler import cycler
from matplotlib.ticker import AutoMinorLocator, MultipleLocator

from ..SciTools import read_csv
from ..Schema import AbsCommon, DiffCommon

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
    def __init__(self, path: Union[str, Path], div: Union[int, float] = 1, **kwargs):
        self.path = path if isinstance(path, Path) else Path(path)
        self.div = div
        self.ref_labels = deepcopy(DiffCommon)
        self.ref_labels.update(kwargs.get("ref_labels", {}))
        self.abs_labels = deepcopy(AbsCommon)
        self.abs_labels.update(kwargs.get("abs_labels", {}))

    def ref_compare(
        self,
        base: str,
        others: Union[str, list[str]],
        xlabel: str,
        ylabel: Union[str, list[str]],
        horiline: bool = True,
        save: Optional[Union[Path, str, bool]] = True,
        **kwargs,
    ):

        data = read_csv(self.path, **kwargs)
        data = data.sort_values(by=xlabel)

        if isinstance(ylabel, list):
            y_name = kwargs.get("y_name", "") or "TMP"
            data[y_name] = np.sum(data[ylabel].values, axis=0)
            print(f'标签{"+".join(ylabel)}总名称指定为{y_name}')
            ylabel = y_name

        plt.figure(figsize=kwargs.get("figsize", (9, 6.5)))
        ax = plt.gca()
        ax.set_prop_cycle(_STYLE_REF)

        base_data = data[data["phase"] == base]

        base_xticks = base_data[xlabel].values
        base_yticks = base_data[ylabel].values

        if isinstance(others, str):
            if others != "others":
                others = [others]
            else:
                others = set(data['phase'])
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
                ax.plot(
                    o_xticks,
                    o_yticks - base_yticks_mask,
                    label=self.ref_labels.get(o, o),
                    lw=2.5,
                    markersize=8,
                    alpha=1.0,
                )
            except ValueError as ve:
                print("Phase:", o)
                print(o_xticks)
                print(base_xticks_mask)
                raise ve

        if horiline:
            rest = data[data["phase"] != base]
            rest_xticks = rest[xlabel].unique()
            mask = np.in1d(base_xticks, rest_xticks)
            base_xticks_mask = base_xticks[mask]
            ax.plot(
                base_xticks_mask,
                np.zeros_like(base_xticks_mask),
                label=self.ref_labels.get(base, base),
                lw=2.5,
                c="k",
                marker="o",
                markersize=8,
                alpha=0.8,
            )

        if trans := kwargs.get("trans"):
            for t in trans:
                ax.axvline(x=t, c="k", alpha=0.5, ls="--", lw=2.5)
        if xminor := kwargs.get("xminor", 5):
            ax.xaxis.set_minor_locator(AutoMinorLocator(xminor))
        if yminor := kwargs.get("yminor", 5):
            ax.yaxis.set_minor_locator(AutoMinorLocator(yminor))
        if xmain := kwargs.get("xmain", False):
            ax.yaxis.set_major_locator(MultipleLocator(xmain))
        if ymain := kwargs.get("ymain", False):
            ax.yaxis.set_major_locator(MultipleLocator(ymain))

        plt.tick_params(axis="both", labelsize=25, pad=8)
        plt.ylabel(self.ref_labels.get(ylabel, ylabel), fontsize=30)
        plt.xlabel(self.ref_labels.get(xlabel, xlabel), fontsize=30)
        if loc := kwargs.get("legend", "in"):
            if loc == "in":
                plt.legend(fontsize=25, loc="best")
            elif loc == "out":
                plt.legend(fontsize=25, loc="upper left", bbox_to_anchor=(1, 1))
        plt.margins(*kwargs.get("margin", (0.15, 0.15)))
        plt.tight_layout()
        plt.show()
        if save:
            if isinstance(save, bool):
                plt.savefig(str(self.path)[:-4] + ".png", dpi=300)
            else:
                plt.savefig(save, dpi=300)

    def abs_compare(
        self,
        phases: Union[list[str], str],
        xlabel: str,
        ylabel: Union[str, list[str]],
        save: Optional[Union[Path, str, bool]] = True,
        **kwargs,
    ):

        data = read_csv(self.path, **kwargs)
        data = data.sort_values(by=xlabel)

        if isinstance(ylabel, list):
            y_name = kwargs.get("y_name", "") or "TMP"
            data[y_name] = np.sum(data[ylabel].values, axis=0)
            print(f'标签{"+".join(ylabel)}总名称指定为{y_name}')
            ylabel = y_name

        plt.figure(figsize=kwargs.get("figsize", (9, 6.5)))
        ax = plt.gca()
        ax.set_prop_cycle(_STYLE_ABS)

        phases = [phases] if isinstance(phases, str) else phases

        for p in phases:
            tmp = data[data.phase == p]
            ax.plot(
                tmp[xlabel],
                tmp[ylabel],
                label=self.abs_labels.get(p, p),
                lw=2.5,
                markersize=8,
            )

        plt.xlabel(self.abs_labels.get(xlabel, xlabel), fontsize=30)
        plt.ylabel(self.abs_labels.get(ylabel, ylabel), fontsize=30)
        plt.tick_params(axis="both", labelsize=25, pad=8)
        if xminor := kwargs.get("xminor", 5):
            ax.xaxis.set_minor_locator(AutoMinorLocator(xminor))
        if yminor := kwargs.get("yminor", 5):
            ax.yaxis.set_minor_locator(AutoMinorLocator(yminor))
        if xmain := kwargs.get("xmain", False):
            ax.yaxis.set_major_locator(MultipleLocator(xmain))
        if ymain := kwargs.get("ymain", False):
            ax.yaxis.set_major_locator(MultipleLocator(ymain))
        if trans := kwargs.get("trans"):
            for i in trans:
                plt.axvline(x=i, c="k", alpha=0.5, ls="--", lw=2.5)

        if kwargs.get("legend", True):
            plt.legend(fontsize=25, loc="upper left", bbox_to_anchor=(0.98, 0.8))

        plt.margins(*kwargs.get("margin", (0.15, 0.15)))
        plt.tight_layout()
        if save:
            if isinstance(save, bool):
                plt.savefig(str(self.path)[:-4] + ".png", dpi=300)
            else:
                plt.savefig(save, dpi=300)

    def multi_target_ref(
        self,
        base: str,
        other: Union[str, list[str]],
        xlabel: str,
        ylabels: Union[str, list[str]],
        ylabel_name: str,
        save: Optional[Union[Path, str, bool]] = True,
        **kwargs,
    ):
        data = self.read_csv(self.path, **kwargs)
        data = data.sort_values(by=xlabel)

        plt.figure(figsize=kwargs.get("figsize", (9, 6.5)))
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
            ax.plot(
                o_xticks,
                o_yticks - base_yticks_mask,
                label=DiffCommon[yl],
                lw=2.5,
                markersize=8,
                alpha=1.0,
            )

        ax.xaxis.set_minor_locator(AutoMinorLocator(5))
        ax.yaxis.set_minor_locator(AutoMinorLocator(5))
        plt.axhline(y=0, c="k", ls=":", lw=4, alpha=0.5)
        ax.xaxis.set_major_locator(MultipleLocator(kwargs.get("xmain", 5)))
        plt.tick_params(axis="both", labelsize=25, pad=8)
        plt.tick_params(axis="both", labelsize=25)
        plt.ylabel(self.ref_labels.get(ylabel_name, ylabel_name), fontsize=30)
        plt.xlabel(self.ref_labels.get(xlabel, xlabel), fontsize=30)

        if kwargs.get("legend", True):
            plt.legend(fontsize=25)
        plt.margins(*kwargs.get("margin", (0.15, 0.15)))
        plt.tight_layout()
        if save:
            if isinstance(save, bool):
                plt.savefig(str(self.path)[:-4] + ".png", dpi=300)
            else:
                plt.savefig(save, dpi=300)
