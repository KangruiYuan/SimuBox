from pathlib import Path
from typing import Union, Iterable, Sequence
from io import BytesIO

Numeric = Union[int, float]
Vector = Union[Sequence[Numeric], Iterable[Numeric]]
PathLike = Union[Path, str]
FileLike = Union[PathLike, BytesIO, bytes]

