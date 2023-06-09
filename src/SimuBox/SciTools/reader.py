import os
import numpy as np
import re
import pandas as pd
import json
from collections import OrderedDict
import matplotlib.pyplot as plt
from pandas.errors import ParserError

__all__ = ['InfoReader']


class InfoReader():
    """
    读取结构信息
    """

    def __init__(self, path,
                 name_flag: bool = False,
                 inverse_flag: bool = False,
                 scale_flag: bool = False,
                 filenames=None,
                 **kwargs
                 ) -> None:
        '''
        @param path: 
        @param name_flag: 
        @param inverse_flag: bool, 用于逆转lxlylz和NxNyNz的顺序，用以适应不同的密度文件写入顺序；通常情况下，gpu文件需要设置为True
        @param scale_flag: bool, 若无lxlylz信息，设置为Ture可使其随NxNyNz进行缩放
        @param filenames: 
        @param kwargs: 
        '''

        self.dataDict = None
        self.fet = None
        self.joint_data = None
        self.joint = None
        self.input = None
        self.block_data = None
        self.block = None
        self.printout = None
        self.lxlylz = None
        self.NxNyNz = None
        self.data = None
        self.phout_data = None
        self.phout = None
        self.path = path

        if filenames is None:
            filenames = {
                'printout': 'printout.txt',
                'phout': 'phout.txt',
                'input': 'input.json',
                'fet': 'fet.txt',
                'block': 'block.txt',
                'joint': 'joint.txt'

            }
        for k, v in filenames.items():
            setattr(self, k, os.path.join(self.path, v))

        self.freeE_re = re.compile('[.0-9e+-]+')
        self.name_flag = name_flag
        self.inverse_flag = inverse_flag
        self.scale_flag = scale_flag

        for k, v in kwargs.items():
            setattr(self, k, v)

    def read_printout(self):
        try:
            lxlylz = open(self.printout, 'r').readlines(
            )[-3].strip().split(' ')

            self.lxlylz_extend = np.array(list(map(float, lxlylz)))
            self.lxlylz = self.lxlylz_extend[:3]

        except FileNotFoundError:
            self.lxlylz = np.array([1, 1, 1])

        if self.inverse_flag: self.lxlylz = self.lxlylz[::-1]
        if self.scale_flag:
            self.lxlylz = self.lxlylz * self.NxNyNz / self.NxNyNz.min()

    def read_phi(self, filename=''):
        try:
            NxNyNz = open(filename).readline().strip().split(' ')
            NxNyNz = np.array(list(map(int, NxNyNz)), np.int32)
            _skip_rows = getattr(self, 'skiprows', 1)
            data = pd.read_csv(filename, skiprows=_skip_rows, header=None, sep=r'[ ]+', engine='python')
            self.__dict__[os.path.basename(filename).split('.')[0] + "_data"] = data.dropna(axis=1, how='any')
            shape = NxNyNz[1:] if NxNyNz[0] == 1 else NxNyNz
            if self.inverse_flag:
                self.NxNyNz = NxNyNz[::-1]
            else:
                self.NxNyNz = NxNyNz
            self.shape = shape
            return True
        except FileNotFoundError:
            print(f'File {filename} not found')
            return False

    def read_phout(self, label=None):

        if label is None:
            label = ['A', 'B', 'C']
        if self.read_phi(self.phout):
            if self.name_flag:
                for i in range(len(self.phout_data.columns) // 2):
                    self.__dict__['phi' + label[i]] = self.data[i].values.reshape(self.shape)
                for i in range(len(self.phout_data.columns) // 2, len(self.phout_data.columns)):
                    self.__dict__['omega' + label[i - len(self.data.columns) // 2]] = self.data[i].values.reshape(
                        self.shape)
            else:
                for i in range(len(self.phout_data.columns)):
                    self.__dict__[
                        "phi" + str(i)] = self.phout_data[i].values.reshape(self.shape)

    def read_block(self):
        if self.read_phi(self.block):
            for i in range(len(self.block_data.columns)):
                self.__dict__[
                    "block" + str(i)] = self.block_data[i].values.reshape(self.shape)

    def read_joint(self):
        if self.read_phi(self.joint):
            for i in range(len(self.joint_data.columns)):
                self.__dict__[
                    "joint" + str(i)] = self.joint_data[i].values.reshape(self.shape)

    def read_json(self):
        try:
            with open(self.input, mode='r') as fp:
                self.jsonData = json.load(fp, object_pairs_hook=OrderedDict)
        except FileNotFoundError:
            return

    def read_fet(self):
        try:
            with open(self.fet, mode='r') as fp:
                cont = fp.readlines()
            dataDict = [line.strip().split()[1:] for line in cont]
            dataDict = {line[0]: float(line[1])
                        for line in dataDict if len(line) == 2}
            self.dataDict = dataDict
        except FileNotFoundError:
            return

    def show(self, phi, asp=None):
        plt.figure()
        if asp is None:
            try:
                asp = self.lxlylz[2] / self.lxlylz[1]
            except AttributeError:
                asp = 1
        plt.imshow(phi, interpolation='spline16', aspect=asp)
        plt.show()

    def collect(self):
        self.read_phout()
        self.read_printout()
        self.read_json()
        self.read_block()
        self.read_joint()
        self.dim = np.sum(self.NxNyNz != 1)

    def coordsMap(self, lxlylz=None, NxNyNz=None):
        lxlylz = self.lxlylz.copy() if not any(lxlylz) else lxlylz
        NxNyNz = self.NxNyNz.copy() if not any(NxNyNz) else NxNyNz
        lx_seq = np.linspace(0, lxlylz[2], NxNyNz[2])
        ly_seq = np.linspace(0, lxlylz[1], NxNyNz[1])
        X, Y = np.meshgrid(lx_seq, ly_seq)
        return X, Y

    def tile(self, mat, lxlylz=None, NxNyNz=None, expand=(3, 3)):

        assert len(mat.shape) == 2
        phiA = np.tile(mat, expand)
        lxlylz = self.lxlylz.copy() if not lxlylz else lxlylz
        NxNyNz = self.NxNyNz.copy() if not NxNyNz else NxNyNz
        lxlylz[1] *= expand[0]
        lxlylz[2] *= expand[1]
        NxNyNz[1] *= expand[0]
        NxNyNz[2] *= expand[1]
        return phiA, lxlylz, NxNyNz
