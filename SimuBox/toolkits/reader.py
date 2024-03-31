import json
import re
from collections import OrderedDict
from io import BytesIO, TextIOWrapper
from pathlib import Path
from typing import Optional, Iterable, Union, Sequence, Callable

import numpy as np
import pandas as pd
import os

from ..schema import (
    Printout,
    Density,
    Fet,
    DensityParseResult,
    Numeric,
    FileLike,
    PathLike,
    Operation,
    Operator,
)
from .function import process_dataframe


def check_filepath(path: PathLike, filename: str):
    return path if path.is_file() else path / filename


def read_json(
    path: PathLike,
    filename: str = "input.json",
    mode: str = "r",
    hook: Optional[Callable] = OrderedDict,
    **kwargs,
):
    with open(check_filepath(path, filename), mode=mode) as fp:
        content = json.load(fp, object_pairs_hook=hook)
    return content


def read_printout(
    path: PathLike,
    filename: str = "printout.txt",
    mode: str = "r",
    encoding: str = "utf-8",
    **kwargs,
):

    to_array = lambda x, y: np.array(list(map(float, x.strip().split(" ")[y:])))

    path = check_filepath(path, filename)
    content = path.open(mode=mode, encoding=encoding).readlines()
    box = to_array(content[-3], 0)
    lxlylz = box[:3].copy()
    potentials = to_array(content[-4], 2)
    volumes = to_array(content[-5], 2)
    uws = re.findall("[.0-9e+-]+", content[-1])
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
        volumes=volumes,
        potentials=potentials,
    )


def read_density(
    path: FileLike,
    parse_N: bool = True,
    parse_L: bool = False,
    filename: str = "phout.txt",
    binary: Sequence[str] = (".bin"),
    mode: str = "r",
    encoding: str = "utf-8",
    **kwargs,
) -> Density:
    """

    :param path: 文件路径或者文件所在的文件夹路径。
    :param filename: 文件名，在path为文件夹路径时生效。
    :param parse_N:
    :param parse_L:
    :param binary:
    :param mode: 读取模式，默认为仅读取（r）。
    :param encoding: 编码模式，默认为utf-8。
    :param kwargs:
    :return:
    """

    if isinstance(path, bytes):
        bin_read_function = np.frombuffer
        content = path
        text = False
    elif isinstance(path, PathLike):
        path = check_filepath(path, filename)
        if path.suffix == binary:
            bin_read_function = np.fromfile
            content = path
            text = False
        else:
            content = path.open(mode=mode, encoding=encoding).readlines()
            text = True
    else:
        raise ValueError(type(path))

    def _validate(vec: np.ndarray):
        return any(vec > 1000) or any(vec <= 0)

    skip = 0
    offset = 0
    if parse_N:
        if text:
            NxNyNz = np.array(list(map(int, content[skip].strip().split(" "))))
        else:
            # int32 for scft
            NxNyNz = bin_read_function(content, dtype=np.int32, count=3)
            if _validate(NxNyNz):
                # int64 for tops
                NxNyNz = bin_read_function(content, dtype=np.int64, count=3)
                if _validate(NxNyNz):
                    raise ValueError(f"NxNyNz解析错误: {NxNyNz}")
                offset += 3 * 8
            else:
                offset += 3 * 4

        skip += 1
    else:
        NxNyNz = kwargs.get("NxNyNz")

    if parse_L:
        if text:
            lxlylz = np.array(list(map(float, content[skip].strip().split(" "))))
        else:
            lxlylz = bin_read_function(
                content, dtype=np.float64, count=3, offset=offset
            )
            offset += 3 * 8
        skip += 1
    else:
        lxlylz = kwargs.get("lxlylz", np.ones(3))

    if text:
        try:
            data = pd.read_csv(
                path, skiprows=skip, header=None, sep=r"[ ]+", engine="python"
            )
        except pd.errors.ParserError as pe:
            if not parse_L or not parse_N:
                return read_density(
                    path,
                    parse_N=True,
                    parse_L=True,
                    filename=filename,
                    mode=mode,
                    **kwargs,
                )
            else:
                raise pe
    else:
        if isinstance(content, PathLike):
            count = os.path.getsize(content)
        else:
            count = len(content)
        count -= offset
        count //= 8

        data = bin_read_function(content, dtype=np.float64, count=count, offset=offset)
        assert NxNyNz is not None
        data = pd.DataFrame(
            data.reshape((int(np.prod(NxNyNz)), -1), order=kwargs.get("order", "F"))
        )

    if NxNyNz is not None:
        shape = NxNyNz[NxNyNz != 1]
        lxlylz = lxlylz[NxNyNz != 1]
    else:
        shape = kwargs.get("shape", NxNyNz)

    return Density(
        path=path if isinstance(path, PathLike) else None,
        data=data,
        NxNyNz=NxNyNz,
        lxlylz=lxlylz,
        shape=shape,
    )


def read_csv(
    path: PathLike,
    skiprows: int = 0,
    accuracy: int = 3,
    Operations: Optional[Union[Operation, Sequence[Operation]]] = None,
    subset: Optional[Union[str, Sequence[str]]] = ("phase", "freeE"),
    **kwargs,
):
    assert path.is_file(), f"{path} 并非有效的文件路径。"
    if path.suffix == ".csv":
        data = pd.read_csv(path, skiprows=skiprows)
    else:
        data = pd.read_excel(path, skiprows=skiprows)
    if "freeE" in data.columns:
        data["freeE"] = np.around(data["freeE"], 8)
    if subset:
        data = data.drop_duplicates(subset=subset)

    if Operations is None:
        Operations = [
            Operation(
                left="ly",
                right="lz",
                operator=Operator.div,
                accuracy=accuracy,
                name="lylz",
            ),
            Operation(
                left="lx",
                right="ly",
                operator=Operator.div,
                accuracy=accuracy,
                name="lxly",
            ),
            Operation(
                left=kwargs.get("var", "chiN"),
                factor=kwargs.get("factor", 1),
                operator=Operator.div,
                accuracy=accuracy,
                name=kwargs.get("var", "chiN"),
            ),
        ]
    elif isinstance(Operations, Operation):
        Operations = [Operations]

    for Opr in Operations:
        try:
            data = process_dataframe(data, Opr)
        except KeyError:
            continue

    return data


def read_fet(
    path: FileLike,
    filename: str = "fet.dat",
    mode: str = "r",
    encoding: str = "utf-8",
    **kwargs,
) -> Fet:
    """
    读取SCFT输出的fet文件。

    :param path: 文件路径或者文件所在的文件夹路径。
    :param filename: 文件名，在path为文件夹路径时生效。
    :param mode: 读取模式，默认为仅读取（r）。
    :param encoding: 编码模式，默认为utf-8。
    :param kwargs:
    :return:
    """
    path = check_filepath(path, filename=filename)
    cont, path = path.open(mode=mode, encoding=encoding).readlines()

    cont = [c.strip().split()[1:] for c in cont if c.strip()]
    res = dict()
    for key, val in cont:
        res[key] = float(val)
    return Fet(path=path, **res)


def periodic_extension(arr: np.ndarray, periods: Sequence[int]):
    """
    对三维数组进行周期性延拓

    Parameters:
    - arr: 三维数组
    - periods: 三维延拓的周期数，例如 (period_x, period_y, period_z)

    Returns:
    - 周期性延拓后的数组
    """
    shape = arr.shape
    result = np.zeros(
        (shape[0] * periods[0], shape[1] * periods[1], shape[2] * periods[2]),
        dtype=arr.dtype,
    )

    for i in range(periods[0]):
        for j in range(periods[1]):
            for k in range(periods[2]):
                result[
                    i * shape[0] : (i + 1) * shape[0],
                    j * shape[1] : (j + 1) * shape[1],
                    k * shape[2] : (k + 1) * shape[2],
                ] = arr

    return result


def parse_density(
    density: Density,
    target: Union[int, Iterable[int], str] = 0,
    permute: Optional[Iterable[int]] = None,
    slices: Optional[tuple[int, int]] = None,
    expand: Union[Numeric, Sequence[Numeric]] = 1,
    **kwargs,
):
    """
    对密度信息进行二次处理，如延拓、交换轴、切片等。

    :param density:
    :param target:
    :param permute:
    :param slices:
    :param expand:
    :param kwargs:
    :return:
    """
    assert density.shape is not None, "需要指定shape属性"
    shape = density.shape.copy()
    lxlylz = density.lxlylz.copy()
    if permute is None:
        permute = np.arange(len(shape))
    else:
        permute = np.array(permute)
        assert len(permute) == len(
            shape
        ), f"length of permute({len(permute)}) and shape({len(shape)}) mismatch"
    shape = shape[permute]
    lxlylz = lxlylz[permute]

    if slices is not None:
        f_mat = lambda x: np.take(x, indices=slices[0], axis=slices[1])
        f_vec = lambda x: np.delete(x, obj=slices[1])
    else:
        f_mat = lambda x: x
        f_vec = lambda x: x

    if target == "all":
        target = list(density.data.columns)
    elif isinstance(target, int):
        target = [target]
    mats = [f_mat(density.data[t].values.reshape(shape)) for t in target]
    shape = f_vec(shape)
    lxlylz = f_vec(lxlylz)

    if isinstance(expand, Numeric):
        expand = np.array([max(expand, 1)] * len(shape))
    else:
        assert len(expand) == len(shape), f"当前矩阵维度为3，拓展信息长度为{len(expand)}, 不匹配"
        expand = np.array(expand)
        expand[expand < 1] = 1

    expand_for_pad = np.around((expand - 1) / 2 * shape).astype(int)

    return DensityParseResult(
        path=density.path,
        raw_NxNyNz=shape.copy(),
        raw_lxlylz=lxlylz.copy(),
        raw_mat=np.stack(mats),
        lxlylz=lxlylz * expand,
        NxNyNz=shape + expand_for_pad * 2,
        mat=np.stack(
            [np.pad(mat, [(e, e) for e in expand_for_pad], "wrap") for mat in mats]
        ),
        # mat=np.stack([np.tile(mat, expand) for mat in mats]),
        # mat=np.stack([periodic_extension(mat, expand) for mat in mats]),
        expand=expand.copy(),
        target=[int(i) for i in target],
    )
