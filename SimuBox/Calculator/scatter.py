from itertools import product
from typing import Union, Sequence, cast, Optional

import matplotlib.pyplot as plt
import numpy as np
import scipy.signal as sg

from .peak import gaussian_expansion
from ..Artist import plot_locators, plot_savefig
from ..Schema import ScatterResult, ScatterPlots

SCATTER_PLOT_CONFIG = {
    "font.family": "Times New Roman",
    # "font.family": "serif",
    "font.serif": ["SimSun"],
    "font.size": 16,
    "mathtext.fontset": "stix",
    "axes.unicode_minus": False,
    "xtick.direction": "in",
    "ytick.direction": "in",
    "xtick.major.width": 2,
    "xtick.minor.width": 2,
    "xtick.major.size": 8,
    "xtick.minor.size": 4,
    "ytick.major.width": 2,
    "ytick.minor.width": 2,
    "ytick.major.size": 8,
    "ytick.minor.size": 4,
    "xtick.labelsize": "medium",
    "ytick.labelsize": "medium",
    "axes.linewidth": 3,
    "legend.frameon": False,
    "legend.fontsize": "medium",
}


class Scatter:
    @staticmethod
    def k_vector(N: int):
        if -N / 2 + 1 > -1:
            vec = np.arange(0, N / 2 + 0.5, 1)
        elif -N / 2 + 1 == -1:
            vec = np.hstack([np.arange(0, N / 2 + 0.5, 1), np.array([-1])])
        else:
            vec = np.hstack([np.arange(0, N / 2 + 0.5, 1), np.arange(-N / 2 + 1, 0, 1)])
        return vec

    @classmethod
    def k_space(cls, *args):
        split_num = len(args) // 2
        NxNyNz = args[:split_num]
        lxlylz = args[split_num:]

        k_vecs = []
        for N, l in zip(NxNyNz, lxlylz):
            assert N % 2 == 0 or N == 1, f"离散格点数应为偶数或1，当前为{NxNyNz}"
            if N != 1:
                k_vec = cls.k_vector(N)
                k_vecs.append((2 * np.pi * k_vec / l) ** 2)

        K_VECS = np.meshgrid(*k_vecs)
        kxyz = sum(K_VECS)

        return kxyz, k_vecs

    @classmethod
    def sacttering_peak(
        cls,
        mat: np.ndarray,
        NxNyNz: Union[np.ndarray, Sequence],
        lxlylz: Union[np.ndarray, Sequence],
        cutoff: int = 30,
        threshold: float = 1e-12,
    ):

        kxyz, k_vecs = cls.k_space(*NxNyNz, *lxlylz)
        Nxyz = np.prod(NxNyNz)
        txyz = np.asarray(list(product(*k_vecs)))

        factor = 1 / Nxyz
        ipha = np.fft.fftn(mat)
        phaNS = ipha.reshape((Nxyz, 1))
        kxyzNS = kxyz.reshape((Nxyz, 1))

        kpp = np.hstack(
            [
                np.real(kxyzNS),
                factor * np.abs(phaNS**2),
                np.real(txyz),
            ]
        )

        kpp = kpp[np.argsort(kpp[:, 0]), :]

        q_Intensity = list()
        q_Intensity.append(kpp[0, 0:2])
        q_idx = dict()
        q_idx[0] = [(kpp[0, 2:])]

        for i in range(1, kpp.shape[0]):
            lenTmp = len(q_Intensity)
            if lenTmp > cutoff:
                break
            if np.abs(q_Intensity[lenTmp - 1][0] - np.abs(kpp[i, 0])) > threshold:
                q_Intensity.append(kpp[i, 0:2])
                q_idx[lenTmp] = [kpp[i, 2:]]
            else:
                q_Intensity[lenTmp - 1][1] += kpp[i, 1]
                if lenTmp - 1 not in q_idx:
                    q_idx[lenTmp - 1] = []
                q_idx[lenTmp - 1].append(kpp[i, 2:])

        q_Intensity = np.asarray(q_Intensity)
        q_Intensity[:, 0] = np.sqrt(q_Intensity[:, 0])

        return ScatterResult(q_Intensity=q_Intensity, q_idx=q_idx)

    @classmethod
    def show_peak(
        cls,
        res: ScatterResult,
        width: float = 0.5,
        step: int = 2000,
        cutoff: int = 20,
        min_height: int = 1,
        save: Optional[bool] = False,
        interactive: bool = False,
        **kwargs,
    ):

        q_Intensity = res.q_Intensity.copy()
        q_Intensity[0, 1] = 0
        # q_idx = res.q_idx.copy()

        q_Intensity = q_Intensity[q_Intensity[:, 0] <= cutoff]

        x = np.linspace(q_Intensity[0, 0], q_Intensity[-1, 0], step)

        plot_y = cast(np.ndarray, 0)
        for q_i in q_Intensity:
            plot_y += gaussian_expansion(array=x, ctr=q_i[0], amp=q_i[1], wid=width)

        peaks_location, peaks_height = sg.find_peaks(plot_y, height=min_height)
        q = x.copy()
        x = x / x[peaks_location[peaks_height["peak_heights"].argmax()]]

        fig = plt.figure(figsize=kwargs.get("figsize", (8, 6)))
        ax = plt.gca()
        plot_x = x if kwargs.get("scale", True) else q
        plt.plot(plot_x, plot_y, lw=2, c="k")
        plt.ylabel("Intensity", fontsize=20)
        plt.xlabel(r"$q/R_g^{-1}$", fontsize=20)

        plot_locators(ax=ax, **kwargs)
        plt.tight_layout()
        plot_savefig(save=save, **kwargs)
        if interactive:
            plt.show()
        return ScatterPlots(
            peaks_location=x[peaks_location],
            fig=fig,
            ax=ax,
            peaks_height=peaks_height,
            plot_x=plot_x,
            plot_y=plot_y,
        )
