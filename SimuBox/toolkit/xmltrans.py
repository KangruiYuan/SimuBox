from pathlib import Path
from typing import Union, Optional, Sequence

from tqdm import tqdm
import pandas as pd
import numpy as np
import re
from itertools import product
import xml

from ..schema import XML, XMLTransResult, PathLike

__all__ = ["XmlTransformer"]


class XmlTransformer:
    xml: XML

    def __init__(
        self,
        path: PathLike,
    ):
        self.path = Path(path)

    @classmethod
    def collect_types(cls, root):
        """
        获取粒子种类。

        :param root:
        :return:
        """
        txt = root.getElementsByTagName("type")[0].firstChild.wholeText
        types = re.sub(r"[\n ]+", " ", txt).strip(" ").split(" ")
        return types

    @classmethod
    def collect_positions(cls, root):
        """
        获取粒子坐标。

        :param root:
        :return:
        """
        txt = root.getElementsByTagName("position")[0].firstChild.wholeText
        positions = re.sub(r"[\n ]+", " ", txt).strip(" ").split(" ")
        positions = list(map(float, positions))
        positions = np.array(positions).reshape((-1, 3))
        return positions

    @classmethod
    def collect_lxlylz(cls, root):
        """
        获取盒子大小。

        :param root:
        :return:
        """
        temp = root.getElementsByTagName("box")[0]
        return np.array([float(temp.getAttribute(i)) for i in ["lx", "ly", "lz"]])

    def parse_xml(
        self,
        NxNyNz: Optional[Union[np.ndarray, Sequence[int]]] = None,
        atoms_mapping: Optional[dict] = None,
        merge: bool = True,
    ):
        """
        解析xml文件，提取其中的信息。

        :param NxNyNz: 设置后续转化为密度时的网格大小。
        :param atoms_mapping: 人工指定不同粒子的映射。
        :param merge: 是否对同类型粒子进行合并。如设置为True，A1和A2均会被视为同一种粒子，否则会被视为两种粒子。
        :return:
        """
        root = xml.dom.minidom.parse(self.path.open("r"))
        lxlylz = self.collect_lxlylz(root)
        types = self.collect_types(root)
        positions = self.collect_positions(root)
        data = pd.DataFrame(data=positions, columns=["x", "y", "z"])
        data["type"] = types
        atoms_type = sorted(data["type"].unique().tolist())
        if atoms_mapping is None:
            if merge:
                atoms_type = [re.sub(r"\d", "", i) for i in atoms_type]
                atoms_type = sorted(list(set(atoms_type)))
            atoms_mapping = dict(list(zip(atoms_type, range(len(atoms_type)))))
            # atoms_mapping = {v: k for k, v in atoms_mapping.items()}

        assert all(
            a in atoms_mapping for a in atoms_type
        ), f"需要建立所有原子类型（{atoms_type}）的映射"
        mapping_values = list(atoms_mapping.values())
        assert (
            all(np.diff(mapping_values) == 1) and min(mapping_values) == 0
        ), "映射值应为起始为0,等差为1的连续整数"
        data["index"] = data["type"].map(atoms_mapping)

        if NxNyNz is None:
            NxNyNz = [i if i % 2 == 0 else i + 1 for i in np.ceil(lxlylz).astype(int)]
        NxNyNz = np.array(NxNyNz, dtype=int)
        grid_spacing = lxlylz / NxNyNz

        res = XML(
            path=self.path,
            data=data,
            NxNyNz=NxNyNz,
            lxlylz=lxlylz,
            grid_spacing=grid_spacing,
            atoms_type=atoms_type,
            atoms_mapping=atoms_mapping,
        )
        self.xml = res
        return res

    @classmethod
    def transform(cls, xml: Optional[XML] = None, r_cut: Union[int, float] = 2.0):
        """
        将xml记录的粒子信息，转化为自洽场的密度信息。

        :param xml:
        :param r_cut:
        :return:
        """

        xml = xml if xml is not None else cls.xml
        assert xml is not None
        phi = np.zeros([len(xml.atoms_type), *xml.NxNyNz])

        idx_lst = np.array(r_cut * 2 / xml.grid_spacing + 2, dtype=int)

        res = np.array(list(product(*[range(idx) for idx in idx_lst])), dtype=int)

        xyz = xml.data[["x", "y", "z"]].values
        t = xml.data["index"].values
        grid_xyz = np.array(
            (
                np.floor((xyz - r_cut) / xml.grid_spacing) * xml.grid_spacing
                + xml.lxlylz / 2.0
            )
            / xml.grid_spacing,
            dtype=int,
        )

        for i in tqdm(range(len(grid_xyz))):
            grids = (res + grid_xyz[i]) % xml.NxNyNz
            grids_coord = grids * xml.grid_spacing - xml.lxlylz / 2
            dis = np.sqrt(np.sum((grids_coord - xyz[i]) ** 2, axis=1))
            mask = dis <= r_cut
            for grid in grids[mask]:
                phi[t[i]][grid[0]][grid[1]][grid[2]] += 1

        return XMLTransResult(xml=xml, phi=phi)
