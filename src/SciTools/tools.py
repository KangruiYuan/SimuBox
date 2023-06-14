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
                 name_flag:bool=False, 
                 inverse_flag:bool=False, 
                 scale_flag:bool=False,
                 filenames:list = ['printout.txt', 'phout.txt', 'input.json', 'fet.txt', 'block.txt']
                 ) -> None:

        self.path = path
        
        for fn in filenames:
            setattr(self, fn.split('.')[0], os.path.join(self.path, fn))

        self.freeE_re = re.compile('[.0-9e+-]+')
        self.name_flag = name_flag
        self.inverse_flag = inverse_flag
        self.scale_flag = scale_flag
        pass

    def read_printout(self):
        try:
            lxlylz = open(self.printout, 'r').readlines(
            )[-3].strip().split(' ')
            self.lxlylz = np.array(list(map(float, lxlylz[:3])))
        except FileNotFoundError:
            self.lxlylz = np.array([1, 1, 1])
        
        if self.inverse_flag: self.lxlylz = self.lxlylz[::-1]
        if self.scale_flag:
            self.lxlylz = self.lxlylz * self.NxNyNz / self.NxNyNz.min()

    def read_phout(self, label=['A', 'B', 'C']):
        try:
            NxNyNz = open(self.phout).readline().strip().split(' ')
            NxNyNz = np.array(list(map(int, NxNyNz)), np.int32)
            # data = pd.read_csv(self.phout, skiprows=1, header=None, delimiter=delimiter)
            data = pd.read_csv(self.phout, skiprows=1, header=None, sep=r'[ ]+', engine='python')
            self.data = data.dropna(axis=1, how='any')
            shape = NxNyNz[1:] if NxNyNz[0] == 1 else NxNyNz
            
            if self.inverse_flag:
                self.NxNyNz = NxNyNz[::-1]
            else:
                self.NxNyNz = NxNyNz
                
            self.shape = shape
            if self.name_flag:
                for i in range(len(self.data.columns) // 2):
                    self.__dict__['phi'+label[i]] = self.data[i].values.reshape(shape)
                for i in range(len(self.data.columns)  // 2, len(self.data.columns)):
                    self.__dict__['omega'+label[i - len(data.columns) // 2]] = self.data[i].values.reshape(shape)
            else:
                for i in range(len(self.data.columns)):
                    self.__dict__[
                        "phi"+str(i)] = self.data[i].values.reshape(shape)
        except FileNotFoundError:
            print('phout file not found')
            
    
    def read_block(self):
        try:
            NxNyNz = open(self.block).readline().strip().split(' ')
            NxNyNz = np.array(list(map(int, NxNyNz)), np.int32)
            data = pd.read_csv(self.block, skiprows=1, header=None, sep=' ')
            self.data = data.dropna(axis=1, how='any')
            shape = NxNyNz[1:] if NxNyNz[0] == 1 else NxNyNz

            if self.inverse_flag:
                self.NxNyNz = NxNyNz[::-1]
            else:
                self.NxNyNz = NxNyNz

            self.shape = shape
            for i in range(len(self.data.columns)):
                self.__dict__[
                    "block"+str(i)] = self.data[i].values.reshape(shape)
        except FileNotFoundError:
            print("block file not found")


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
                asp = self.lxlylz[2]/self.lxlylz[1]
            except AttributeError:
                asp = 1
        plt.imshow(phi, interpolation='spline16', aspect=asp)
        plt.show()
        

    def collect(self):
        self.read_phout()
        self.read_printout()
        self.read_json()
        
        
    def coordsMap(self, lxlylz=None, NxNyNz = None):
        lxlylz = self.lxlylz.copy() if not any(lxlylz) else lxlylz
        NxNyNz = self.NxNyNz.copy() if not any(NxNyNz) else NxNyNz
        lx_seq = np.linspace(0, lxlylz[2], NxNyNz[2])
        ly_seq = np.linspace(0, lxlylz[1], NxNyNz[1])   
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
    


    
