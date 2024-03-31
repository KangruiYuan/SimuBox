import pandas as pd

from ..types import Vector
from .MixinStructs import MixinBaseModel
from typing import Optional, Union
from pathlib import Path
import numpy as np

__all__ = [
    "Printout",
    "Fet",
    "Density",
    "DensityParseResult",
    "XMLRaw",
    "XMLTransResult",
]


class Printout(MixinBaseModel):
    """
    TOPS产出的printout.txt文件的解析结果。
    """

    path: Optional[Path] = None
    box: np.ndarray
    lxlylz: np.ndarray
    step: int
    freeEnergy: float
    freeW: float
    freeS: float
    freeWS: float
    freeU: float
    inCompMax: float
    volumes: np.ndarray
    potentials: np.ndarray


class Fet(MixinBaseModel):
    """
    SCFT产出的fet.dat(.txt)文件的解析结果。
    """

    path: Optional[Path] = None
    xACN: Optional[float] = None
    xCAN: Optional[float] = None
    xABN: Optional[float] = None
    xBAN: Optional[float] = None
    xCBN: Optional[float] = None
    xBCN: Optional[float] = None
    stressX: Optional[float] = None
    stressY: Optional[float] = None
    stressZ: Optional[float] = None
    lx: Optional[float] = None
    ly: Optional[float] = None
    lz: Optional[float] = None
    lx_full: Optional[float] = None
    ly_full: Optional[float] = None
    lz_full: Optional[float] = None
    freeEnergy: Optional[float] = None
    freeAB_sum: Optional[float] = None
    freeAB: Optional[float] = None
    freeBA: Optional[float] = None
    freeAC: Optional[float] = None
    freeCA: Optional[float] = None
    freeBC: Optional[float] = None
    freeCB: Optional[float] = None
    freeW: Optional[float] = None
    freeS: Optional[float] = None
    freeDiff: Optional[float] = None
    inCompMax: Optional[float] = None
    Q0: Optional[float] = None
    VolumeFraction0: Optional[float] = None
    sumVolume: Optional[float] = None
    activity0: Optional[float] = None


class Density(MixinBaseModel):
    """
    密度类文件的解析结果。密度类文件包括：phout/phin/block/joint/component。
    """

    path: Optional[Path] = None
    data: pd.DataFrame
    NxNyNz: Optional[np.ndarray] = None
    lxlylz: Optional[np.ndarray] = None
    shape: Optional[np.ndarray] = None

    def repair_from_printout(self, printout: Printout):
        """
        基于printout文件对密度进行修复。

        :param printout: printout文件的解析结果。
        :return: None
        """
        if self.NxNyNz is not None:
            mask = self.NxNyNz != 1
        else:
            mask = None
        if self.lxlylz is None or (all(self.lxlylz == 1) and any(printout.lxlylz != 1)):
            self.lxlylz = printout.lxlylz if mask is None else printout.lxlylz[mask]

    def repair_from_fet(self, fet: Fet):
        """
        基于fet文件对密度进行修复。

        :param fet: fet文件的解析结果。
        :return:
        """
        if self.NxNyNz is not None:
            mask = self.NxNyNz != 1
        else:
            mask = None
        lxlylz = np.array([fet.lx, fet.ly, fet.lz])
        if self.lxlylz is None or (all(self.lxlylz == 1) and any(lxlylz != 1)):
            self.lxlylz = lxlylz if mask is None else lxlylz[mask]

    def repair_data(
        self,
        inputs: Union[dict, Vector, str] = "auto",
        species: Optional[dict] = None,
        constraint: Optional[Vector] = None,
    ):
        """
        对密度数据进行修复（当前仅针对block.txt文件）

        :param inputs: 每个嵌段的体积分数，该输入可以是input.json文件、嵌段体积分数的向量以及'auto'，'auto'代表通过线性方程求解来自动修复密度。
        :param species: 每种分子链的体积分数，字典形式（键为json文件中的SpecyID， 值为VolumeFraction）。
        :param constraint: 对于自动求解的模式，可以增加额外的限制变量。
        :return:
        """
        self.data = self.data.div(self.data.sum(axis=0))
        if isinstance(inputs, dict):
            if species is None:
                species = dict(
                    [(s["SpecyID"], s["VolumeFraction"]) for s in inputs["Specy"]]
                )
            blocks = [
                round(block["ContourLength"] * species[block["SpecyID"]], 6)
                for block in inputs["Block"]
            ]
        elif inputs == "auto":
            self.data = self.data * np.prod(self.shape)
            col_num = self.data.shape[1]
            if constraint is None:
                constraint = np.ones(col_num)
            else:
                assert len(constraint) == col_num
            solve_A = np.vstack([self.data.iloc[: col_num - 1, :].values, constraint])
            solve_b = np.ones(col_num)
            blocks = np.linalg.solve(solve_A, solve_b)
        else:
            blocks = inputs

        blocks = np.array(blocks)
        self.data = self.data * blocks
        self.data = self.data.div(self.data.sum(axis=1), axis=0)


class DensityParseResult(MixinBaseModel):
    """
    密度文件数据的二次加工结果
    """

    path: Optional[Path] = None
    raw_NxNyNz: np.ndarray
    raw_lxlylz: np.ndarray
    raw_mat: np.ndarray
    lxlylz: np.ndarray
    NxNyNz: np.ndarray
    mat: np.ndarray
    expand: np.ndarray
    target: list[int]


class XMLRaw(MixinBaseModel):
    path: Path
    NxNyNz: np.ndarray
    lxlylz: np.ndarray
    grid_spacing: np.ndarray
    data: pd.DataFrame
    atoms_type: list
    atoms_mapping: dict[str, int]


class XMLTransResult(MixinBaseModel):
    xml: XMLRaw
    phi: np.ndarray

    def write(
        self,
        phi: Optional[np.ndarray] = None,
        path: Optional[Union[str, Path]] = None,
        scale: bool = False,
    ):
        if phi is None:
            phi = self.phi
        if scale:
            phi = phi / phi.sum(axis=0)
        phi = np.c_[[i.reshape((-1, 1)) for i in phi]]
        phi = np.squeeze(phi).T
        if path is None:
            path = self.xml.path.with_suffix(".txt")
        np.savetxt(
            path,
            phi,
            fmt="%.3f",
            delimiter=" ",
            header=" ".join(map(str, self.xml.NxNyNz)),
            comments="",
        )
