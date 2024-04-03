import re
from pathlib import Path
from typing import Optional, Union, Sequence

import numpy as np
import pandas as pd

from ..schema import RealNum, Vector, PathLike, Operation, OPERATOR_FUNCTION_MAP


__all__ = [
    "arange",
    "find_nearest_1d",
    "replace_target",
    "match_target",
    "process_dataframe",
]


def arange(
    start: Union[int, float],
    end: Union[int, float],
    step: Union[int, float],
    accuracy: int = 6,
):
    """
    从start到end，以指定的step作为间隔生成数值序列。强制包含两个端点，同时默认保持六位小数精度。

    :param start: 起始值
    :param end: 终止值
    :param step: 步长
    :param accuracy: 保留的小数精度
    :return: np.ndarray
    """
    if start > end:
        start, end = end, start
    gap = end - start
    num = gap // abs(step) + 1
    asc = np.arange(num) * abs(step)

    res = np.around(asc + start, accuracy)
    if end not in res:
        res = np.append(res, end)
    return np.sort(res)


def find_nearest_1d(array: Vector, value: RealNum) -> int:
    """
    寻找序列（array）中与指定值（value）最接近的数值的索引。

    :param array: 任意向量
    :param value: 任意数值
    :return: 索引值（整型）
    """
    if not isinstance(array, np.ndarray):
        array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx


def replace_target(
    path: PathLike, target: Union[str, RealNum], key: str = "phase"
) -> Optional[Path]:
    """
    替换路径中的信息。

    :param path: 原路径。
    :param target: 需要替换的值。
    :param key: 值对应的名称。
    :return: 替换值之后的路径。
    """

    path = str(path)
    pattern = re.compile(key + r"(\w+)_?")
    target = str(target)
    match = pattern.search(path)
    # 如果找到匹配项，则进行替换
    if match:
        # 获取匹配的字符串
        matched_string = match.group(1)
        modified_string = path.replace(matched_string, target)
        return Path(modified_string)
    else:
        print(f"路径：{path} 修改失败。")
        return


def match_target(candidates: Sequence[PathLike], **kwargs):
    """
    按照指定的关键字匹配路径。

    :param candidates: 等待匹配的对象。
    :param kwargs:
    :return: 匹配到的对象。
    """
    criteria = [k + str(v) for k, v in kwargs.items()]
    res = []
    for candidate in candidates:
        candidate = str(candidate)
        for c in criteria:
            if c not in candidate:
                break
        else:
            res.append(Path(candidate))
    if len(res) == 0:
        print(f"{criteria}匹配失败。")
        return
    else:
        return res if len(res) > 1 else res[0]


def process_dataframe(data: pd.DataFrame, operation: Operation):
    """
    对dataframe数据进行便捷的预处理。

    :param data:
    :param operation:
    :param accuracy:
    :return:
    """

    if operation.left in data.columns:
        left = data[operation.left].values
        if operation.right in data.columns:
            right = data[operation.right].values
        elif operation.factor is not None:
            right = operation.factor
        else:
            raise KeyError(
                f"right: {operation.right} 或 factor: {operation.factor} 应至少有一个有效"
            )

        data[operation.name] = np.around(
            OPERATOR_FUNCTION_MAP[operation.operator](left, right), operation.accuracy
        )
        return data
    else:
        raise KeyError(f"left: {operation.left}不是有效数据列")
