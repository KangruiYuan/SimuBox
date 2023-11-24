import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from typing import Iterable, Union, Callable, Optional
from ..Schema import NumericType, PeakInfo, PeakFitResult
from ..Artist import plot_legend, plot_savefig


def gaussian_expansion(
    array: Union[Iterable, NumericType],
    amp: NumericType,
    ctr: NumericType,
    wid: NumericType,
):
    return amp * np.exp(-(((array - ctr) / wid) ** 2))


def curve_function(x: Union[Iterable, NumericType], *params):

    assert len(params) % 3 == 1

    res = 0
    for i in range(0, len(params) - 1, 3):
        res += gaussian_expansion(x, params[i], params[i + 1], params[i + 2])

    return res + params[-1]


def curve_split(x: Union[Iterable, NumericType], *params):
    assert len(params) % 3 == 1
    res = []
    for i in range(0, len(params) - 1, 3):
        res.append(
            gaussian_expansion(x, params[i], params[i + 1], params[i + 2]) + params[-1]
        )

    return np.array(res)


def peak_fit(
    x: np.ndarray,
    y: np.ndarray,
    peaks: np.ndarray,
    func: Callable = curve_function,
    amplitudes: Optional[np.ndarray] = None,
    widths: Optional[np.ndarray] = None,
    background: Optional[NumericType] = None,
    plot: bool = True,
    **kwargs
):
    x = np.asarray(x)
    y = np.asarray(y)
    y = y - y.min()
    background = background if background is not None else y.mean() / 4
    amplitudes = (
        amplitudes if amplitudes is not None else [float(y.mean() / 2)] * len(peaks)
    )
    widths = (
        widths
        if widths is not None
        else [float(np.diff(peaks).mean() / 4)] * len(peaks)
    )

    guess = np.array(list(zip(amplitudes, peaks, widths))).flatten()
    guess = np.append(guess, background)
    popt, _ = curve_fit(func, x, y, p0=guess)

    fit = func(x, *popt)
    y_split = curve_split(x, *popt)

    if plot:
        fig = plt.figure(figsize=kwargs.get("figsize", (8, 6)))
        plt.scatter(x, y, s=20, label="real")
        plt.plot(x, fit, ls="-", c="black", lw=1, label="fitted")

        baseline = np.zeros_like(x) + popt[-1]
        for n, i in enumerate(y_split):
            plt.fill_between(
                x, i, baseline, facecolor=cm.rainbow(n / len(y_split)), alpha=0.6
            )
        plot_legend(kwargs.get("legend", {"fontsize": 15}))
        plt.tight_layout()
        plot_savefig(**kwargs)

    peaks = []
    for i in range(0, len(popt) - 1, 3):
        peaks.append(
            PeakInfo(
                amplitude=popt[i],
                center=popt[i + 1],
                width=popt[i + 2],
                background=popt[-1],
            )
        )
    return PeakFitResult(
        raw_x=x, raw_y=y, peaks=peaks, fitted_curve=fit, split_curve=y_split
    )
