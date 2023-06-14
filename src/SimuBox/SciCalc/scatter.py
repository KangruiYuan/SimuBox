
from ..SciTools import InfoReader
import numpy as np
from matplotlib.ticker import AutoMinorLocator, MultipleLocator
import scipy.signal as sg
import matplotlib.pyplot as plt
from itertools import product

class Scatter(InfoReader):
    def __init__(self, path: str, name_flag: bool = False, inverse_flag: bool = False):

        super().__init__(path, inverse_flag=inverse_flag, name_flag=name_flag)

    @staticmethod
    def showPic(pic):
        plt.figure()
        plt.imshow(pic, cmap='gray')
        plt.colorbar()

    def cal_kxyz(self, Nx, Ny, Nz, lx, ly, lz):
        assert (Nx % 2 == 0 or Nx ==
                1) and Nx != 0, 'please check the x dimension!'
        assert (Ny % 2 == 0 or Ny ==
                1) and Ny != 0, 'please check the y dimension!'
        assert (Nz % 2 == 0 or Nz ==
                1) and Nz != 0, 'please check the z dimension!'

        if -Nx / 2 + 1 > -1:
            ti = np.arange(0, Nx / 2 + 0.5, 1)
        elif -Nx / 2 + 1 == -1:
            ti = np.hstack([np.arange(0, Nx / 2 + 0.5, 1), np.array([-1])])
        else:
            ti = np.hstack([np.arange(0, Nx / 2 + 0.5, 1),
                    np.arange(-Nx / 2 + 1, 0, 1)])

        if -Ny / 2 + 1 > -1:
            tj = np.arange(0, Ny / 2 + 0.5, 1)
        elif -Ny / 2 + 1 == -1:
            tj = np.hstack([np.arange(0, Ny / 2 + 0.5, 1), np.array([-1])])
        else:
            tj = np.hstack([np.arange(0, Ny / 2 + 0.5, 1),
                    np.arange(-Ny / 2 + 1, 0, 1)])

        if Nz != 1:
            if -Nz / 2 + 1 > -1:
                tk = np.arange(0, Nz / 2 + 0.5, 1)
            elif -Nz / 2 + 1 == -1:
                tk = np.hstack([np.arange(0, Nz / 2 + 0.5, 1), np.array([-1])])
            else:
                tk = np.hstack([np.arange(0, Nz / 2 + 0.5, 1),
                            np.arange(-Nz / 2 + 1, 0, 1)])

        kx = np.expand_dims((2 * np.pi * ti / lx) ** 2, axis=0).T
        ky = np.expand_dims((2 * np.pi * tj / ly) ** 2, axis=0)
        kxyz = kx + ky
        if Nz != 1:
            kz = np.reshape((2 * np.pi * tk / lz) ** 2, (1, 1, Nz))
            kxyz = np.expand_dims(kxyz, axis=-1) + kz
            return kxyz, ti, tj, tk
        return kxyz, ti, tj

    def sacttering_peak(self, mat, cutoff: int = 30, threshold: float = 1e-12) -> tuple[list, dict]:

        Nx, Ny, Nz = self.NxNyNz
        lx, ly, lz = self.lxlylz

        if Nx != 1 and len(mat.shape) == 3:
            Nxyz = np.prod(self.NxNyNz)
            kxyz, ti, tj, tk = self.cal_kxyz(Nx, Ny, Nz, lx, ly, lz)
            txyz = np.asarray(list(product(ti, tj, tk)))
        else:
            Nxyz = Nx * Ny
            kxyz, ti, tj = self.cal_kxyz(Nx, Ny, Nz, lx, ly, lz)
            txyz = np.asarray(list(product(ti, tj)))


        factor = 1 / Nxyz
        ipha = np.fft.fftn(mat)
        phaNS = ipha.reshape((Nxyz, 1))
        kxyzNS = kxyz.reshape((Nxyz, 1))

        kpp = np.hstack(
            [
                np.real(kxyzNS),
                factor * np.abs(phaNS ** 2),
                np.real(txyz),
            ])
        I = np.argsort(kpp[:, 0])
        kpp = kpp[I, :]

        q_Intensity = list()
        q_Intensity.append(kpp[0, 0:2])
        qidx = dict()
        qidx[0] = [(kpp[0, 2:])]

        for i in range(1, kpp.shape[0]):
            lenTmp = len(q_Intensity)
            if lenTmp > cutoff:
                break
            if np.abs(q_Intensity[lenTmp - 1][0] - np.abs(kpp[i, 0])) > threshold:
                q_Intensity.append(kpp[i, 0:2])
                qidx[lenTmp] = [kpp[i, 2:]]
            else:
                q_Intensity[lenTmp - 1][1] += kpp[i, 1]
                if lenTmp - 1 not in qidx:
                    qidx[lenTmp - 1] = []
                qidx[lenTmp - 1].append(kpp[i, 2:])

        q_Intensity = np.asarray(q_Intensity)
        q_Intensity[:, 0] = np.sqrt(q_Intensity[:, 0])
        self.q_Intensity = q_Intensity
        self.qidx = qidx

    @staticmethod
    def GaussianExpansion(q_I, x, w=0.5):
        return q_I[1] * np.exp(-(x - q_I[0]) ** 2 / (2 * w ** 2))

    @staticmethod
    def extract_idx(qidx: list, index, num=5):
        Idx_str = []
        for i in range(num):
            cIdx = qidx[int(index[i])]
            cIdx = [np.append(j, 0) for j in cIdx]
            tmp = [np.sort(np.abs(j))[::-1] for j in cIdx]
            tmp = np.unique(tmp, axis=0)[0]
            text = ''.join(['{', *list(map(lambda x: str(int(x)), tmp)), '}'])
            Idx_str.append(text)
        return Idx_str

    def PeakShow(self, step=2000, cutoff=20, height=1, **kwargs):

        q_Intensity = self.q_Intensity[1:]
        q_Intensity = q_Intensity[q_Intensity[:, 0] <= cutoff]

        x = np.linspace(self.q_Intensity[0, 0], q_Intensity[-1, 0], step)

        y = 0
        for i in q_Intensity:
            y += self.GaussianExpansion(i, x=x, w=kwargs.get('w', 0.5))

        peaks, height_info = sg.find_peaks(y, height=height)
        q = x.copy()
        x = x / x[peaks[height_info['peak_heights'].argmax()]]

        plt.figure(figsize=(8, 6))
        plt.plot(q, y, lw=2, c='k')
        plt.ylabel('Intensity', fontsize=20)
        plt.xlabel(r'$q/R_g^{-1}$', fontsize=20)
        ax = plt.gca()
        if xminor := kwargs.get('xminor',False):
            ax.xaxis.set_minor_locator(AutoMinorLocator(xminor))
        if yminor := kwargs.get('yminor',False):
            ax.yaxis.set_minor_locator(AutoMinorLocator(yminor))

        if kwargs.get('mark', False):
            num = kwargs.get('num', 5)
            tmp_q_I = self.q_Intensity.copy()
            tmp_q_I[0, 1] = 0
            index = np.argsort(-tmp_q_I[:, 1])
            Idx_str = self.extract_idx(self.qidx, index, num=num)

            text_x = [tmp_q_I[index[i]][0] for i in range(num)]
            text_y = [tmp_q_I[index[i]][1] for i in range(num)]

            min_x = min(text_x)
            text_x = [i / min_x for i in text_x]

            for i in range(num):
                plt.text(x=text_x[i],
                         y=text_y[i],
                         s=Idx_str[i],
                         rotation=90, c='r')

        x0, x1, y0, y1 = plt.axis()
        plt.axis((x0, x1, y0, y1 + kwargs.get('y1', 40)))
        plt.tight_layout()
        plt.show()
        return x[peaks]
