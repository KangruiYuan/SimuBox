from dataclasses import dataclass
from typing import Union, Sequence, Optional
from matplotlib.figure import Figure, Axes
from numpy import ndarray
from enum import Enum
from pandas import DataFrame


NumericType = Union[int, float]
VectorType = Sequence[NumericType]


class ColorType(str, Enum):

    RGB = "RGB"
    HEX = "HEX"


@dataclass()
class CompareResult:
    df: DataFrame
    plot_dict: dict
    mat: ndarray
    fig: Optional[Figure] = None
    ax: Optional[Axes] = None


class DetectionMode(str, Enum):
    HORIZONTAL = "hori"
    VERTICAL = "vert"
    BOTH = "both"
    INTERP = "interp"
    MIX = "mix"


@dataclass()
class PointInfo:
    phase: str
    x: float
    y: float
    freeEnergy: float
