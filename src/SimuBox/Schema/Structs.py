
from typing import Optional, Any
import numpy as np
import pandas as pd
from matplotlib.figure import Figure, Axes
from pydantic import BaseModel


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

    data: pd.DataFrame
    NxNyNz: Optional[np.ndarray] = None
    lxlylz: Optional[np.ndarray] = None
    shape: Optional[np.ndarray] = None

class CompareResult(ExtendedModel):
    df: pd.DataFrame
    plot_dict: dict
    mat: np.ndarray
    fig: Optional[Figure] = None
    ax: Optional[Axes] = None

class LandscapeResult(ExtendedModel):
    freeEMat: np.ndarray
    ly: np.ndarray
    lz: np.ndarray
    levels:  np.ndarray
    ticks:  np.ndarray
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