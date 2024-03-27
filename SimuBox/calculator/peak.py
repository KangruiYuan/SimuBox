import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from typing import Iterable, Union, Callable, Sequence
from ..schema import Numeric, Peak, PeakFitResult
from ..artist import plot_legend, plot_savefig, plot_locators
from sklearn.metrics import r2_score


def gaussian_expansion(
    array: Union[Iterable, Numeric],
    amp: Numeric,
    ctr: Numeric,
    wid: Numeric,
):
    return amp * np.exp(-(((array - ctr) / wid) ** 2))


def curve_function(x: Union[Iterable, Numeric], *params):

    assert len(params) % 3 == 1

    res = 0
    for i in range(0, len(params) - 1, 3):
        res += gaussian_expansion(x, params[i], params[i + 1], params[i + 2])

    return res + params[-1]

def curve_split(x: Union[Iterable, Numeric], *params):
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
    peaks: Sequence[Peak],
    func: Callable = curve_function,
    fix_background: bool = False,
    xlabel: str = r"Wavenumbers/${\rm cm}^{-1}$",
    ylabel: str = "Absorbance (a.u.)",
    plot: bool = True,
    interactive: bool = True,
    **kwargs
):
    x = np.asarray(x)
    y = np.asarray(y)
    y = y - y.min()
    centers = [p.center for p in peaks]
    backgrounds = [p.background for p in peaks if p.background is not None]
    if backgrounds:
        background = np.mean(backgrounds)
    else:
        background = y.mean() / 4

    amplitudes = []
    widths = []
    base_amplitude = float(y.mean() / 2)
    base_width = float(np.diff(centers).mean() / 4)
    for p in peaks:
        amplitudes.append(p.amplitude if p.amplitude is not None else base_amplitude)
        widths.append(p.width if p.width is not None else base_width)

    guess = np.array(list(zip(amplitudes, centers, widths))).flatten()
    guess = np.append(guess, background)

    ub = []
    lb = []
    lb_scale, ub_scale = kwargs.get("scale", (0.995, 1.005))
    for i in range(len(amplitudes)):
        for j, f in enumerate(peaks[i].fix):
            if f:
                lb.append(max(guess[3 * i + j] * lb_scale, 0))
                ub.append(guess[3 * i + j] * ub_scale)
            else:
                lb.append(0)
                ub.append(np.inf)
    # print(lb,ub)
    if fix_background:
        lb.append(max(guess[-1] * lb_scale, 0))
        ub.append(guess[-1] * ub_scale)
    else:
        lb.append(0)
        ub.append(np.inf)

    popt, _ = curve_fit(func, x, y, p0=guess, bounds=(lb, ub))

    fit = func(x, *popt)
    y_split = curve_split(x, *popt)

    fig = ax = None
    areas = np.zeros(len(y_split))
    if plot:
        fig = plt.figure(figsize=kwargs.get("figsize", (6, 5)))
        plt.scatter(x, y, s=20, label="real")
        plt.plot(x, fit, ls="-", c="black", lw=1, label="fitted")

        baseline = np.zeros_like(x) + popt[-1]

        for n, i_line in enumerate(y_split):
            plt.fill_between(
                x, i_line, baseline, facecolor=cm.rainbow(n / len(y_split)), alpha=0.6
            )
            areas[n] = np.trapz(i_line - baseline, x)

        plot_legend(kwargs.get("legend", {"fontsize": 15}))
        plt.tight_layout()
        plot_savefig(**kwargs)
        if xlabel:
            plt.xlabel(xlabel, fontsize=15)
        if ylabel:
            plt.ylabel(ylabel, fontsize=15)
        plot_locators(**kwargs)
        if interactive:
            plt.show()

    res_peaks = []
    for i_r, i_p in enumerate(range(0, len(popt) - 1, 3)):
        res_peaks.append(
            Peak(
                amplitude=popt[i_p],
                center=popt[i_p + 1],
                width=popt[i_p + 2],
                background=popt[-1],
                fix=peaks[i_r].fix,
                area=areas[i_r],
            )
        )

    r2 = r2_score(y, fit)
    n = len(x)
    p = len(popt)
    adj_r2 = 1 - ((1 - r2) * (n - 1) / (n - p - 1))

    return PeakFitResult(
        x=x,
        y=y,
        peaks=res_peaks,
        fitted_curve=fit,
        split_curve=y_split,
        fig=fig,
        ax=ax,
        r2=r2,
        adj_r2=adj_r2,
        area=np.trapz(y - np.ones_like(y) * popt[-1], x),
    )
