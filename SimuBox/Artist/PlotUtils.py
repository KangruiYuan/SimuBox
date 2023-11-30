from pathlib import Path

import matplotlib.pyplot as plt
from typing import Optional

import numpy as np
from matplotlib.ticker import AutoMinorLocator, MultipleLocator
from ..Schema import PathType, ColorType
from typing import Union


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
    save: Union[PathType, bool] = False,
    **kwargs,
):
    if isinstance(save, bool):
        if not save:
            return
        else:
            assert hasattr(obj, "path"), "不支持自动保存，请传递路径信息给参数save"
            path = Path(obj.path)
    elif isinstance(save, PathType):
        path = Path(save)
    else:
        raise ValueError(f"save类型应为[Path, str, bool], 而当前是{type(save)}")

    stem = path.stem
    stem = prefix + "_" + stem if prefix else stem
    stem = stem + "_" + suffix if suffix else stem
    if path.is_file():
        path = path.parent / ".".join((stem, kwargs.get("fig_format", "png")))
    elif path.is_dir():
        path = path / ".".join((stem, kwargs.get("fig_format", "png")))
    plt.savefig(path, dpi=dpi)


def generate_colors(
    mode: Union[str, ColorType] = ColorType.RGB,
    num: int = 1,
    linear: bool = True,
    **kwargs,
):
    if mode == ColorType.RGB:
        color = np.random.choice(range(256), size=(num, 3)).tolist()
    elif mode == ColorType.HEX:
        color = [
            "#" + "".join(i)
            for i in np.random.choice(list("0123456789ABCDEF"), size=(num, 6))
        ]
    elif mode == ColorType.L:
        if linear:
            color = np.linspace(0, 255, num, dtype=int).tolist()
        else:
            color = np.random.choice(range(256), size=num).tolist()
    else:
        raise NotImplementedError(mode.value)
    return color
