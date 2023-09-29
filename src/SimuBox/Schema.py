from pathlib import Path
from typing import Union, Sequence, Optional, Any

import numpy as np
from matplotlib.figure import Figure, Axes
from numpy import ndarray
from pandas import DataFrame
from pydantic.dataclasses import dataclass
from enum import Enum
NumericType = Union[int, float]
VectorType = Sequence[NumericType]
PathType = Union[Path, str]


class Config:
    arbitrary_types_allowed = True


@dataclass(config=Config)
class Printout:
    box: ndarray
    lxlylz: ndarray
    step: int
    freeEnergy: float
    freeW: float
    freeS: float
    freeWS: float
    freeU: float
    inCompMax: float
    # dimensions: list[int]


@dataclass(config=Config)
class Density:
    data: ndarray
    NxNyNz: Optional[ndarray] = None
    lxlylz: Optional[ndarray] = None
    shape: Optional[ndarray] = None

    # reshape: Optional[dict[int, np.ndarray]] = None

    @property
    def reshaped(self):

        if self.shape is None:
            raise ValueError("shape must be specified")
        else:
            return [self.data[:, i].reshape(self.shape) for i in range(self.data.shape[1])]


class ColorType(str, Enum):
    RGB = "RGB"
    HEX = "HEX"


@dataclass(config=Config)
class CompareResult:
    df: DataFrame
    plot_dict: dict
    mat: ndarray
    fig: Optional[Figure] = None
    ax: Optional[Axes] = None


@dataclass(config=Config)
class LandscapeResult:
    freeEMat: np.ndarray
    ly: Union[np.ndarray, list]
    lz: Union[np.ndarray, list]
    levels:  Union[np.ndarray, list]
    ticks:  Union[np.ndarray, list]
    fig: Optional[Figure] = None
    ax: Optional[Axes] = None
    contourf_fig: Optional[Any] = None
    contour_fig: Optional[Any] = None
    clb: Optional[Any] = None


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
