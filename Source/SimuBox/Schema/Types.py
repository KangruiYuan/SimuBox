from pathlib import Path
from typing import Union, Iterable, BinaryIO, Sequence

Numeric = Union[int, float]
Vector = Union[Sequence[Numeric], Iterable[Numeric]]
PathLike = Union[Path, str, BinaryIO]

