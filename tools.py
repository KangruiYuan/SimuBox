import os
import numpy as np
import re
import pandas as pd
import json
from collections import OrderedDict
from matplotlib.ticker import AutoMinorLocator, MultipleLocator
import torch as t
import scipy.signal as sg
import matplotlib.pyplot as plt

__all__ = ['InfoReader', 'Scatter']

class InfoReader():
    """
    读取结构信息
    """

    def __init__(self, path, name_flag:bool=True, inverse_flag:bool=False) -> None:

        self.path = path
        self.printout = os.path.join(self.path, 'printout.txt')
        self.phout = os.path.join(self.path, 'phout.txt')
        self.json = os.path.join(self.path, 'input.json')

        self.freeE_re = re.compile('[.0-9e+-]+')
        self.name_flag = name_flag
        self.inverse_flag = inverse_flag
        pass

    def read_printout(self):
        try:
            lxlylz = open(self.printout, 'r').readlines(
            )[-3].strip().split(' ')
            self.lxlylz = np.array(
                list(map(float, lxlylz[:3])), dtype=np.float32)
        except:
            self.lxlylz = np.array([1, 1, 1])
        
        if self.inverse_flag: self.lxlylz = self.lxlylz[::-1]

    def read_phout(self, label=['A', 'B', 'C']):
        NxNyNz = open(self.phout).readline().strip().split(' ')
        NxNyNz = np.array(list(map(int, NxNyNz)), np.int32)
        data = pd.read_csv(self.phout, skiprows=1, header=None, sep=' ')
        # self.data = data.drop(len(data.columns) - 1, axis=1)
        self.data = data.dropna(axis=1, how='any')
        shape = NxNyNz[1:] if NxNyNz[0] == 1 else NxNyNz
        
        if self.inverse_flag:
            self.NxNyNz = NxNyNz[::-1]
        else:
            self.NxNyNz = NxNyNz
        self.shape = shape
        # print(len(data.columns), len(data.columns) // 2)
        # print(len(self.data.columns) // 2)
        if self.name_flag:
            for i in range(len(self.data.columns) // 2):
                # print(i)
                self.__dict__['phi'+label[i]] = self.data[i].values.reshape(shape)
            for i in range(len(self.data.columns)  // 2, len(self.data.columns)):
                # print(i - len(data.columns) // 2)
                # print(i)
                self.__dict__['omega'+label[i - len(data.columns) // 2]] = self.data[i].values.reshape(shape)
        else:
            for i in range(len(self.data.columns)):
                self.__dict__[
                    "phi"+str(i)] = self.data[i].values.reshape(shape)


    def read_json(self):
        try:
            with open(self.json, mode='r') as fp:
                self.jsonData = json.load(fp, object_pairs_hook=OrderedDict)
                # self.phi0 = round(
                #     float(self.jsonData["Specy"][0]["VolumeFraction"]), 6)
                # self.fA_total = 0
            self.phase = self.jsonData.get('_phase_log', 'C4')
        except:
            print('No json file.')

    def collect(self):
        self.read_phout()
        self.read_printout()
        self.read_json()
        
    def check(self):
        for attrName in ['phiA', 'phiB', 'lxlylz', 'NxNyNz']:
            if not hasattr(self, attrName):
                return False

        if any(self.lxlylz < 0):
            self.logger.error('Incorrect period.')
            return False
        else:
            return True
        
    def coordsMap(self, lxlylz=None, NxNyNz = None):
        lxlylz = self.lxlylz.copy() if not any(lxlylz) else lxlylz
        NxNyNz = self.NxNyNz.copy() if not any(NxNyNz) else NxNyNz
        lx_seq = np.linspace(0, lxlylz[1], NxNyNz[1])
        ly_seq = np.linspace(0, lxlylz[2], NxNyNz[2])
        X, Y = np.meshgrid(lx_seq, ly_seq)
        return X, Y
    
    def tile(self, mat, lxlylz=None, NxNyNz = None, expand=(3, 3)):
        
        assert len(mat.shape) == 2
        phiA = np.tile(mat, expand)
        lxlylz = self.lxlylz.copy() if not lxlylz else lxlylz
        NxNyNz = self.NxNyNz.copy() if not NxNyNz else NxNyNz
        lxlylz[1] *= expand[0]
        lxlylz[2] *= expand[1]
        NxNyNz[1] *= expand[0]
        NxNyNz[2] *= expand[1]
        return phiA, lxlylz, NxNyNz
    

class Scatter(InfoReader):
    def __init__(self, path: str, name_flag: bool = False, inverse_flag: bool = False):
        
        super().__init__(path, inverse_flag=inverse_flag, name_flag=name_flag)
        # self.reader = InfoReader(path)
        # self.NxNyNz = self.reader.NxNyNz
        # self.lxlylz = self.reader.lxlylz
    
    @staticmethod
    def showPic(pic):
        plt.figure()
        plt.imshow(pic, cmap='gray')
        plt.colorbar()
    
        
    def cal_kxyz(self, Nx, Ny, Nz, lx, ly, lz):
        assert (Nx % 2 == 0 or Nx == 1) and Nx != 0, 'please check the x dimension!'
        assert (Ny % 2 == 0 or Ny == 1) and Ny != 0, 'please check the y dimension!'
        assert (Nz % 2 == 0 or Nz == 1) and Nz != 0, 'please check the z dimension!'

        if -Nx / 2 + 1 > -1:
            ti = t.arange(0, Nx / 2 + 0.5, 1)
        elif -Nx / 2 + 1 == -1:
            ti = t.cat([t.arange(0, Nx / 2 + 0.5, 1), t.Tensor([-1])])
        else:
            ti = t.cat([t.arange(0, Nx / 2 + 0.5, 1), t.arange(-Nx / 2 + 1, 0, 1)])

        if -Ny / 2 + 1 > -1:
            tj = t.arange(0, Ny / 2 + 0.5, 1)
        elif -Ny / 2 + 1 == -1:
            tj = t.cat([t.arange(0, Ny / 2 + 0.5, 1), t.Tensor([-1])])
        else:
            tj = t.cat([t.arange(0, Ny / 2 + 0.5, 1), t.arange(-Ny / 2 + 1, 0, 1)])

        if Nz != 1:
            if -Nz / 2 + 1 > -1:
                tk = t.arange(0, Nz / 2 + 0.5, 1)
            elif -Nz / 2 + 1 == -1:
                tk = t.cat([t.arange(0, Nz / 2 + 0.5, 1), t.Tensor([-1])])
            else:
                tk = t.cat([t.arange(0, Nz / 2 + 0.5, 1),
                        t.arange(-Nz / 2 + 1, 0, 1)])

        kx = ((2 * np.pi * ti / lx) ** 2).unsqueeze(0).T
        ky = ((2 * np.pi * tj / ly) ** 2).unsqueeze(0)
        kxyz = kx + ky
        if Nz != 1:
            kz = ((2 * np.pi * tk / lz) ** 2).resize_((1, 1, Nz))
            kxyz = kxyz.unsqueeze(-1) + kz
            return kxyz, ti, tj, tk
        return kxyz, ti, tj
    
    def sacttering_peak(self, mat, cutoff: int = 30, threshold: float = 1e-12, d2:bool=False) -> tuple[list, dict]:
        '''
        Be careful! The x axis and z axis is inverse due to the writing method of pha!
        '''
        mat = t.Tensor(mat)
        Nx, Ny, Nz = self.NxNyNz
        lx, ly, lz = self.lxlylz
        
        if Nx != 1 and len(mat.size()) == 3:
            Nxyz = np.prod(self.NxNyNz)
            kxyz, ti, tj, tk = self.cal_kxyz(Nx, Ny, Nz, lx, ly, lz)
            txyz = []
            for i in range(Nx):
                for j in range(Ny):
                    for k in range(Nz):
                        txyz.append([ti[i], tj[j], tk[k]])
            txyz = t.Tensor(txyz)
        else:
            Nxyz = Nx * Ny
            kxyz, ti, tj = self.cal_kxyz(Nx, Ny, Nz, lx, ly, lz)
            txyz = []
            for i in range(Nx):
                for j in range(Ny):
                    txyz.append([ti[i], tj[j]])
            txyz = t.Tensor(txyz)

        factor = 1 / Nxyz
        ipha = t.fft.fftn(mat)
        phaNS = ipha.reshape((Nxyz, 1))

        kxyzNS = kxyz.reshape((Nxyz, 1))
        kp = t.cat([kxyzNS, phaNS, txyz], dim=1)
        if d2:
            return kp
        
        I = np.argsort(kp[:, 0])
        kp = kp[I, :]
        kpp = t.cat(
            [
                t.real(kp[:, 0]).reshape((Nxyz, 1)),
                factor * t.abs(kp[:, 1] ** 2).reshape((Nxyz, 1)),
                t.real(kp[:, 2:])
            ], dim=1
        )

        q_Intensity = list()
        q_Intensity.append(kpp[0, 0:2].numpy())
        qidx = dict()
        qidx[0] = [(kpp[0, 2:])]

        for i in range(1, kpp.shape[0]):
            lenTmp = len(q_Intensity)
            if lenTmp > cutoff:
                break
            if t.abs(q_Intensity[lenTmp - 1][0] - t.abs(kp[i, 0])) > threshold:
                q_Intensity.append(kpp[i, 0:2].numpy())
                qidx[lenTmp] = [kpp[i, 2:]]
            else:
                q_Intensity[lenTmp - 1][1] += kpp[i, 1].numpy()
                if lenTmp - 1 not in qidx:
                    qidx[lenTmp - 1] = []
                qidx[lenTmp-1].append(kpp[i, 2:])
        
        q_Intensity = np.asarray(q_Intensity)
        q_Intensity[:, 0] = np.sqrt(q_Intensity[:, 0])
        self.q_Intensity = q_Intensity
        self.qidx = qidx
    
    @staticmethod
    def GaussianExpansion(q_I, x, w=0.5):
        return q_I[1] * np.exp(-(x - q_I[0])**2/(2 * w**2))
    
    @staticmethod
    def extract_idx(qidx:list[t.Tensor], index, num=5):
        Idx_str = []
        for i in range(num):
            cIdx = qidx[int(index[i])]
            cIdx = [np.append(j.numpy(), 0) for j in cIdx]
            tmp = [np.sort(np.abs(j))[::-1] for j in cIdx]
            tmp = np.unique(tmp, axis=0)[0]
            text = ''.join(['{',*list(map(lambda x:str(int(x)), tmp)),'}'])
            Idx_str.append(text)
        return Idx_str
    
    def PeakShow(self, step=2000, cutoff=20, height=1, **kwargs):
        
        q_Intensity = self.q_Intensity[1:]
        q_Intensity = q_Intensity[q_Intensity[:,0] <= cutoff ]

        x = np.linspace(self.q_Intensity[0,0], q_Intensity[-1,0], step)
        
        y = 0
        for i in q_Intensity:
            y += self.GaussianExpansion(i, x=x, w=kwargs.get('w', 0.5))
            
        peaks, height_info = sg.find_peaks(y, height=height)
        q = x.copy()
        x = x/x[peaks[height_info['peak_heights'].argmax()]]
        
        plt.figure(figsize=(8, 6))
        plt.plot(q, y, lw =2, c='k')
        plt.ylabel('Intensity', fontsize=20)
        plt.xlabel(r'$q/R_g^{-1}$', fontsize=20)
        ax = plt.gca()
        ax.xaxis.set_minor_locator(AutoMinorLocator(5))
        ax.yaxis.set_minor_locator(AutoMinorLocator(5))
        
        if kwargs.get('mark', False):
            num = kwargs.get('num', 5)
            tmp_q_I = self.q_Intensity.copy()
            tmp_q_I[0,1] = 0
            index = np.argsort(-tmp_q_I[:,1])
            Idx_str = self.extract_idx(self.qidx, index, num=num)
            
            text_x = [tmp_q_I[index[i]][0] for i in range(num)]
            text_y = [tmp_q_I[index[i]][1] for i in range(num)]
            
            min_x = min(text_x)
            text_x = [i/min_x for i in text_x]
            
            for i in range(num):
                plt.text(x=text_x[i],
                         y=text_y[i],
                        s=Idx_str[i],
                        rotation=90, c='r')

        x0, x1, y0, y1 = plt.axis()
        plt.axis((x0,x1,y0,y1 + kwargs.get('y1',40)))
        plt.tight_layout()
        return x[peaks]
    
