from tqdm import tqdm
import pandas as pd
import numpy as np
import re
from itertools import product
import xml


class XmlTransformer():

    def __init__(self, path: str, rcut: float = 2.0, map_dict=None):

        if map_dict is None:
            map_dict = {
                'A': 0,
                'A1': 0,
                'A2': 0,
                'B': 1,
                'B1': 1,
                'B2': 1,
                'C': 2,
                'D': 3,
                'F': 4
            }
        self.path = path
        self.rcut = rcut
        self.map_dict = map_dict

    def getXmlData(self,
                   tag=None,
                   NxNyNz=None):
        # global natoms, lxlylz, grid_spacing
        if tag is None:
            tag = ['position', 'type', 'box']
        dom = xml.dom.minidom.parse(self.path)
        root = dom.documentElement
        dataList = []
        nameList = []
        for _tag in tag:
            tmp = root.getElementsByTagName(_tag)[0]
            if _tag == 'box':
                lxlylz = np.array([float(tmp.getAttribute(i))
                                   for i in ['lx', 'ly', 'lz']])
                if NxNyNz:
                    NxNyNz = np.array(NxNyNz)
                    grid_spacing = lxlylz / NxNyNz
                else:
                    tmp = np.ceil(lxlylz, dtype=int)
                    NxNyNz = []
                    for i in tmp:
                        num = i if i % 2 == 0 else i + 1
                        NxNyNz.append(num)
                    NxNyNz = np.array(np.ceil(lxlylz, dtype=int))
                    grid_spacing = lxlylz / grid_spacing

                continue
            # print(root.getElementsByTagName(_tag))
            tmp_data = tmp.firstChild.read_csv
            # tmp_data = tmp_data.replace('\n', '').strip(' ')
            tmp_data = re.sub(r'[\n ]+', ' ', tmp_data)
            tmp_data = tmp_data.strip(' ')
            tmp_data_float = []
            if _tag == 'position':
                tmp_data_float = list(map(float, tmp_data.split(' ')))
                tmp_data_float = np.array(
                    tmp_data_float, dtype=np.float64).reshape((-1, 3))
                nameList.extend(['X', 'Y', 'Z'])
                natoms = int(tmp.getAttribute('num'))
            elif _tag == 'type':
                tmp_data_float = np.array(tmp_data.split(
                    ' '), dtype=str).reshape((-1, 1))
                if tmp_data_float[0][0].isalpha():
                    map_flag = True
                nameList.extend(['type'])
            # finally:
            dataList.append(tmp_data_float)
        dataAll = np.hstack(dataList)
        df = pd.DataFrame(data=dataAll, columns=nameList)
        self.atoms = dict(
            [(i, self.map_dict[i]) for i in df['type'].unique()]
        )
        if map_flag:
            df['type'] = df['type'].map(self.map_dict)
        df = df.astype({'X': float, 'Y': float, 'Z': float, 'type': int})

        self.data = df
        self.natoms = natoms
        self.NxNyNz = NxNyNz
        self.lxlylz = lxlylz
        self.grid_spacing = grid_spacing

    def phi_mode_sphere(self, xyz, grid_xyz, t, phi):
        grids = (self.res + grid_xyz) % self.NxNyNz
        grids_coord = grids * self.grid_spacing - self.lxlylz / 2
        dis = np.sqrt(np.sum((grids_coord - xyz) ** 2, axis=1))
        mask = (dis <= self.rcut)
        for grid in grids[mask]:
            phi[t][grid[0]][grid[1]][grid[2]] += 1

    def transform(self):
        phi = np.zeros([len(self.atoms), *self.NxNyNz])

        idx_lst = np.array(self.rcut * 2 / self.grid_spacing + 2, dtype=int)
        self.res = np.array(list(product(*[range(idx) for idx in idx_lst])), dtype=int)

        df_val = self.data.values
        xyz = df_val[:, :3]
        t = df_val[:, -1].astype(int)
        grid_xyz = np.array(
            (np.floor((xyz - self.rcut) / self.grid_spacing) *
             self.grid_spacing + self.lxlylz / 2.0) / self.grid_spacing, dtype=int)

        for i in tqdm(range(len(grid_xyz))):
            self.phi_mode_sphere(xyz[i], grid_xyz[i], t[i], phi)

        self.phi = phi

    def write(self, path='phout.txt', scale: bool = False):
        if scale:
            ph = self.phi / self.phi.sum(axis=0)
        ph_lst = [ph[i].reshape((-1, 1))
                  for i in range(len(self.atoms))]
        ph = np.c_[ph_lst]
        ph = np.squeeze(ph)
        ph = ph.T
        np.savetxt(path, ph, fmt="%.3f", delimiter=" ",
                   header=' '.join(map(str, self.NxNyNz)), comments="")
