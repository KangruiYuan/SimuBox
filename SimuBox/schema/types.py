from pathlib import Path
from typing import Union, Iterable, Sequence
from io import BytesIO
from numpy import ndarray

__all__ = ["Numeric", "Vector", "PathLike", "FileLike"]

Numeric = Union[int, float]
Vector = Union[Sequence[Numeric], Iterable[Numeric], ndarray]
PathLike = Union[Path, str]
FileLike = Union[PathLike, BytesIO, bytes]
