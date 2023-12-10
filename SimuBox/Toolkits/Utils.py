import re
from pathlib import Path
from typing import Optional, Sequence

import numpy as np

from ..Schema import Numeric, Vector, PathLike


def find_nearest_1d(array: Vector, value: Numeric) -> Numeric:
    if not isinstance(array, np.ndarray):
        array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx


def replace_target(path: PathLike, target: str, key: str = "phase") -> Optional[Path]:
    path = str(path)
    pattern = re.compile(key + r"(\w+)_?")
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


def match_path(paths, **kwargs):
    criteria = [k + str(v) for k, v in kwargs.items()]
    for path in paths:
        path = str(path)
        for c in criteria:
            if c not in path:
                break
        else:
            return Path(path)
    print(f"{criteria}匹配失败。")
    return
