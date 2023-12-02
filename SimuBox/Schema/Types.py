from pathlib import Path
from typing import Union, Iterable, BinaryIO

NumericType = Union[int, float]
VectorType = Iterable[NumericType]
PathType = Union[Path, str, BinaryIO]

