from pathlib import Path
from typing import Union, Iterable, Sequence
from io import BytesIO
from numpy import ndarray

__all__ = ["RealNum", "Vector", "PathLike", "FileLike", "Numeric"]

RealNum = Union[int, float]
Numeric = Union[RealNum, complex]
Vector = Union[Sequence[RealNum], Iterable[RealNum], ndarray]
PathLike = Union[Path, str]
FileLike = Union[PathLike, BytesIO, bytes]
