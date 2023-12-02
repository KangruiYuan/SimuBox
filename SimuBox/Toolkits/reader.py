import json
import re
from collections import OrderedDict
from pathlib import Path
from typing import Optional, Iterable, Union, Sequence
from io import BytesIO, TextIOWrapper
from copy import deepcopy
import tempfile

import numpy as np
import pandas as pd

from ..Schema import Printout, Density, PathType, FetData, DensityParseResult

def read_file(path:PathType, default_name: Optional[str] = None, **kwargs):
    encoding = kwargs.get('encoding', "utf-8")
    if isinstance(path, BytesIO):
        with TextIOWrapper(path, encoding=encoding) as txt:
            content = txt.readlines()
    elif isinstance(path, (Path, str)):
        path = Path(path)
        if not path.is_file():
            assert default_name is not None
            path = path / default_name
        content = path.open("r", encoding=encoding).readlines()
    else:
        raise NotImplementedError(f"{type(path)}的读取方式尚未建立。")
    return content, path


def read_json(path: PathType):
    with open(path, mode="r") as fp:
        dic = json.load(fp, object_pairs_hook=OrderedDict)
    return dic


def read_printout(path: PathType, **kwargs):
    cont, path = read_file(path, default_name=kwargs.get('default_name', "printout.txt"))
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
    cont, path = read_file(path, default_name=kwargs.get('default_name', "phout.txt"))

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

    try:
        data = pd.read_csv(path, skiprows=skip, header=None, sep=r"[ ]+", engine="python")
    except:
        return read_density(path, parse_N=True, parse_L=True, **kwargs)

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


def read_fet(path: PathType, **kwargs):
    cont, path = read_file(path, default_name=kwargs.get('default_name', "fet.dat"))

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
    expand: Union[int, Sequence[int]] = 1,
    **kwargs,
):
    assert density.shape is not None, "需要指定shape属性"
    shape = density.shape.copy()
    lxlylz = density.lxlylz.copy()
    if permute is None:
        permute = np.arange(len(shape))
    else:
        permute = np.array(permute)
        assert len(permute) == len(shape), f"length of permute({len(permute)}) and shape({len(shape)}) mismatch"
    shape = shape[permute]
    lxlylz = lxlylz[permute]
    if slices is not None:
        f_mat = lambda x: np.take(x, indices=slices[0], axis=slices[1])
        f_vec = lambda x: np.delete(x, obj=slices[1])
    else:
        f_mat = lambda x: x
        f_vec = lambda x: x

    target = [target] if isinstance(target, int) else target
    mats = [f_mat(density.data[t].values.reshape(shape)) for t in target]
    shape = f_vec(shape)
    lxlylz = f_vec(lxlylz)


    if isinstance(expand, int):
        expand = np.array([expand] * len(shape))
    else:
        assert len(expand) == len(shape), f"当前矩阵维度为3，拓展信息长度为{len(expand)}, 不匹配"

    return DensityParseResult(
            path=density.path,
            raw_NxNyNz=shape.copy(),
            raw_lxlylz=lxlylz.copy(),
            raw_mat=np.stack(mats),
            lxlylz=lxlylz * expand,
            NxNyNz=shape * expand,
            mat=np.stack([np.tile(mat, expand) for mat in mats]),
            expand=expand.copy(),
    )
