import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from typing import Iterable, Union, Callable, Optional, Sequence
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


def fix_param(params: np.ndarray, fix: Sequence):
    assert len(params) % 3 == 1
    fixed = []
    for i in range(0, len(params) - 1, 3):
        for idj, j in enumerate(fix[:-1]):
            idi = i + idj
            if j:
                fixed.append(params[idi])
            else:
                fixed.append(-1)
    if fix[-1]:
        fixed.append(params[-1])
    else:
        fixed.append(-1)
    fixed = np.array(fixed)

    def curve(x: Union[Iterable, NumericType], *args):
        args = np.array(args)
        args[fixed != -1] = fixed[fixed != -1]
        res = 0
        for i in range(0, len(args) - 1, 3):
            res += gaussian_expansion(x, params[i], params[i + 1], params[i + 2])
        return res + args[-1]

    return curve


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
    peaks: Sequence[PeakInfo],
    func: Callable = curve_function,
    fix: Sequence[bool] = (False, False, False, False),
    # peaks: np.ndarray,
    # amplitudes: Optional[np.ndarray] = None,
    # widths: Optional[np.ndarray] = None,
    # background: Optional[NumericType] = None,
    plot: bool = True,
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
    lb_scale, ub_scale = kwargs.get('scale', (0.995, 1.095))
    for i in range(len(amplitudes)):
        for j, f in enumerate(fix[:-1]):
            if f:
                lb.append(max(guess[3 * i + j] * lb_scale, 0))
                ub.append(guess[3 * i + j] * ub_scale)
            else:
                lb.append(0)
                ub.append(np.inf)

    if fix[-1]:
        lb.append(max(guess[-1] * lb_scale, 0))
        ub.append(guess[-1] * ub_scale)
    else:
        lb.append(0)
        ub.append(np.inf)

    popt, _ = curve_fit(func, x, y, p0=guess, bounds=(lb, ub))

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

    res_peaks = []
    for i in range(0, len(popt) - 1, 3):
        res_peaks.append(
            PeakInfo(
                amplitude=popt[i],
                center=popt[i + 1],
                width=popt[i + 2],
                background=popt[-1],
            )
        )
    return PeakFitResult(
        raw_x=x, raw_y=y, peaks=res_peaks, fitted_curve=fit, split_curve=y_split
    )
