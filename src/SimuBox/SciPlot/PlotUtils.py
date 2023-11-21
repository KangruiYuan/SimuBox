from pathlib import Path

import matplotlib.pyplot as plt
from typing import Optional
from matplotlib.ticker import AutoMinorLocator, MultipleLocator


def init_plot_config(config: dict):
    plt.rcParams.update(config)


def plot_trans(trans: Optional[dict] = None, **kwargs):
    if trans is None:
        trans = dict()
    alpha = trans.get("alpha", 0.5)
    ls = trans.get("ls", "--")
    lw = trans.get("lw", 2.5)
    c = trans.get("c", "k")
    if ys := trans.get("ys"):
        for y in ys:
            plt.axhline(y=y, c=c, alpha=alpha, ls=ls, lw=lw)

    if xs := trans.get("xs"):
        for x in xs:
            plt.axvline(x=x, c=c, alpha=alpha, ls=ls, lw=lw)


def plot_legend(legend: Optional[dict] = None, **kwargs):
    if legend is None:
        legend = dict()
    fontsize = legend.get("fontsize", 25)
    mode = legend.get("mode", "auto")
    if mode == "outside":
        loc = legend.get("loc", "upper left")
        bbox_to_anchor = legend.get("bbox_to_anchor", (0.98, 0.8))
        plt.legend(fontsize=fontsize, loc=loc, bbox_to_anchor=bbox_to_anchor)
    elif mode == "auto":
        plt.legend(fontsize=fontsize, loc="best")
    else:
        raise NotImplementedError("legend mode must be one of ['outside', 'auto']")


def plot_locators(
    ax: Optional = None,
    xminor: Optional[int] = 5,
    yminor: Optional[int] = 5,
    xmajor: Optional[int] = None,
    ymajor: Optional[int] = None,
    **kwargs,
):
    if ax is None:
        ax = plt.gca()
    if xminor is not None:
        ax.xaxis.set_minor_locator(AutoMinorLocator(xminor))
    if yminor is not None:
        ax.yaxis.set_minor_locator(AutoMinorLocator(yminor))
    if xmajor is not None:
        ax.yaxis.set_major_locator(MultipleLocator(xmajor))
    if ymajor is not None:
        ax.yaxis.set_major_locator(MultipleLocator(ymajor))


def plot_savefig(
    obj: Optional = None,
    prefix: str = "",
    suffix: str = "",
    dpi: int = 150,
    save: str = "auto",
    **kwargs,
):
    if save == "auto" and hasattr(obj, 'path'):
        path = Path(obj.path)
        stem = path.stem
        stem = prefix + "_" + stem if prefix else stem
        stem = stem + "_" + suffix if suffix else stem
        path = path.parent / (stem + kwargs.get("fig_format", ".png"))
    else:
        path = save
    plt.savefig(path, dpi=dpi)
