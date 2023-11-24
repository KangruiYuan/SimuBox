from pathlib import Path
from typing import Optional, Any, Sequence

import numpy as np
import pandas as pd
from matplotlib.figure import Figure, Axes
from pydantic import BaseModel


class ExtendedModel(BaseModel):
    class Config:
        arbitrary_types_allowed = True


class Printout(ExtendedModel):
    path: Optional[Path] = None
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
    path: Optional[Path] = None
    data: pd.DataFrame
    NxNyNz: Optional[np.ndarray] = None
    lxlylz: Optional[np.ndarray] = None
    shape: Optional[np.ndarray] = None


class FetData(ExtendedModel):
    path: Optional[Path] = None
    xACN: Optional[float] = None
    xCAN: Optional[float] = None
    xABN: Optional[float] = None
    xBAN: Optional[float] = None
    xCBN: Optional[float] = None
    xBCN: Optional[float] = None
    stressX: Optional[float] = None
    stressY: Optional[float] = None
    stressZ: Optional[float] = None
    lx: Optional[float] = None
    ly: Optional[float] = None
    lz: Optional[float] = None
    lx_full: Optional[float] = None
    ly_full: Optional[float] = None
    lz_full: Optional[float] = None
    freeEnergy: Optional[float] = None
    freeAB_sum: Optional[float] = None
    freeAB: Optional[float] = None
    freeBA: Optional[float] = None
    freeAC: Optional[float] = None
    freeCA: Optional[float] = None
    freeBC: Optional[float] = None
    freeCB: Optional[float] = None
    freeW: Optional[float] = None
    freeS: Optional[float] = None
    freeDiff: Optional[float] = None
    inCompMax: Optional[float] = None
    Q0: Optional[float] = None
    VolumeFraction0: Optional[float] = None
    sumVolume: Optional[float] = None
    activity0: Optional[float] = None


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
    levels: np.ndarray
    ticks: np.ndarray
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

class PeakInfo(ExtendedModel):

    amplitude: float
    center: float
    width: float
    background: float

class PeakFitResult(ExtendedModel):

    raw_x: np.ndarray
    raw_y: np.ndarray
    peaks: Sequence[PeakInfo]
    fitted_curve: np.ndarray
    split_curve: np.ndarray




