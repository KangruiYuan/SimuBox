
from pydantic import BaseModel
from typing import Optional, Union, Any

import numpy as np
import pandas as pd
from matplotlib.figure import Figure, Axes

class ExtendedModel(BaseModel):
    class Config:
        arbitrary_types_allowed = True

class Printout(ExtendedModel):

    box: np.ndarray
    lxlylz: np.ndarray
    step: int
    freeEnergy: float
    freeW: float
    freeS: float
    freeWS: float
    freeU: float
    inCompMax: float

class Density(ExtendedModel):

    data: np.ndarray
    NxNyNz: Optional[np.ndarray] = None
    lxlylz: Optional[np.ndarray] = None
    shape: Optional[np.ndarray] = None

    @property
    def reshaped(self):

        if self.shape is None:
            raise ValueError("shape must be specified")
        else:
            return [self.data[:, i].reshape(self.shape) for i in range(self.data.shape[1])]


class CompareResult(ExtendedModel):
    df: pd.DataFrame
    plot_dict: dict
    mat: np.ndarray
    fig: Optional[Figure] = None
    ax: Optional[Axes] = None

class LandscapeResult(ExtendedModel):
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

class PointInfo(ExtendedModel):
    phase: str
    x: float
    y: float
    freeEnergy: float