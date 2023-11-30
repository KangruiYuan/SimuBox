import json
import os
import re
from collections import OrderedDict
from pathlib import Path
from typing import Optional, Iterable, Union, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ..Schema import Printout, Density, PathType, FetData, TileResult


def read_json(file: PathType):
    with open(file, mode="r") as fp:
        dic = json.load(fp, object_pairs_hook=OrderedDict)
    return dic


def read_printout(path: PathType):
    if isinstance(path, str):
        path = Path(path)
    if not path.is_file():
        path = path / "printout.txt"
    cont = open(path, "r").readlines()
    box = np.array(list(map(float, cont[-3].strip().split(" "))))
    lxlylz = box[:3]
    uws = re.findall("[.0-9e+-]+", cont[-1])
    uws = list(map(float, uws))

    return Printout(
        path=path,
        lxlylz=lxlylz,
        box=box,
        step=uws[0],
        freeEnergy=uws[1],
        freeU=uws[2],
        freeW=uws[3],
        freeS=uws[4],
        freeWS=uws[3] + uws[4],
        inCompMax=uws[-1],
    )


def read_density(path: PathType, parse_N: bool = True, parse_L: bool = False, **kwargs):
    """
    for phout, component, block
    :param parse_L:
    :param parse_N:
    :param path:
    :return:
    """
    if isinstance(path, str):
        path = Path(path)
    if not path.is_file():
        path = path / "phout.txt"
    cont = open(path, "r").readlines()

    skip = 0
    if parse_N:
        NxNyNz = np.array(list(map(int, cont[skip].strip().split(" "))))
        skip += 1
    else:
        NxNyNz = kwargs.get("NxNyNz")

    if parse_L:
        lxlylz = np.array(list(map(float, cont[skip].strip().split(" "))))
        skip += 1
    else:
        lxlylz = kwargs.get("lxlylz", np.array([1, 1, 1]))

    data = pd.read_csv(path, skiprows=skip, header=None, sep=r"[ ]+", engine="python")

    if NxNyNz is not None:
        shape = NxNyNz[NxNyNz != 1]
        lxlylz = lxlylz[NxNyNz != 1]
    else:
        shape = kwargs.get("shape", NxNyNz)

    return Density(path=path, data=data, NxNyNz=NxNyNz, lxlylz=lxlylz, shape=shape)


def read_csv(path: PathType, **kwargs):

    df = pd.read_csv(path)
    if subset := kwargs.get("subset", ["phase", "freeE"]):
        df = df.drop_duplicates(subset=subset)
    df["lylz"] = np.around(df["ly"].values / df["lz"].values, kwargs.get("acc", 3))
    df["lxly"] = np.around(df["lx"].values / df["ly"].values, kwargs.get("acc", 3))
    try:
        df[kwargs.get("var", "chiN")] = df[kwargs.get("var", "chiN")] / kwargs.get(
            "factor", 1
        )
    except KeyError:
        pass
    return df


def read_fet(path: PathType):
    if isinstance(path, str):
        path = Path(path)
    if not path.is_file():
        path = path / "fet.dat"

    with open(path, mode="r") as fp:
        cont = fp.readlines()
    res = [c.strip().split()[1:] for c in cont if c.strip()]
    res = dict([[c[0], float(c[1])] for c in res])
    return FetData(path=path, **res)


def parse_density(
    density: Density,
    target: Union[int, Iterable[int]] = 0,
    permute: Optional[Iterable[int]] = None,
    slices: Optional[tuple[int, int]] = None,
    **kwargs,
):
    assert density.shape is not None, "需要指定shape属性"
    shape = density.shape.copy()
    lxlylz = density.lxlylz.copy()
    if permute is None:
        permute = np.arange(len(shape))
    else:
        permute = np.array(permute)
        assert len(permute) == len(shape), "length of permute and shape mismatch"
    shape = shape[permute]
    lxlylz = lxlylz[permute]
    if slices is not None:
        f_mat = lambda x: np.take(x, indices=slices[0], axis=slices[1])
        f_vec = lambda x: np.delete(x, obj=slices[1])
    else:
        f_mat = lambda x: x
        f_vec = lambda x: x
    if isinstance(target, int):
        return (
            f_mat(density.data[target].values.reshape(shape)),
            f_vec(shape),
            f_vec(lxlylz),
        )
    elif isinstance(target, Iterable):
        return (
            [f_mat(density.data[t].values.reshape(shape)) for t in target],
            f_vec(shape),
            f_vec(lxlylz),
        )
    else:
        raise TypeError("Check the type for param: target")


def tile_density(
    density: Density,
    target: int = 0,
    permute: Optional[Iterable[int]] = None,
    expand: Union[int, Sequence[int]] = 3,
    **kwargs,
):
    mat, shape, lxlylz = parse_density(density, target, permute, **kwargs)
    if isinstance(expand, int):
        expand = np.array([expand] * mat.ndim)
    else:
        assert len(expand) == mat.ndim
    return TileResult(
        raw_NxNyNz=shape.copy(),
        raw_lxlylz=lxlylz.copy(),
        raw_mat=mat.copy(),
        lxlylz=lxlylz * expand,
        NxNyNz=shape * expand,
        mat=np.tile(mat, expand),
        expand=expand.copy(),
    )

