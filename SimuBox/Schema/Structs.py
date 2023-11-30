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

    def repair(self, printout: Printout):
        if self.NxNyNz is not None:
            mask = self.NxNyNz != 1
        else:
            mask = None
        if self.lxlylz is None or (all(self.lxlylz == 1) and any(printout.lxlylz != 1)):
            self.lxlylz = printout.lxlylz if mask is None else printout.lxlylz[mask]

class Summation(ExtendedModel):
    path: Optional[Path] = None
    box: Optional[np.ndarray] = None
    NxNyNz: Optional[np.ndarray] = None
    lxlylz: Optional[np.ndarray] = None
    shape: Optional[np.ndarray] = None
    data: pd.DataFrame
    step: int
    freeEnergy: float
    freeW: float
    freeS: float
    freeWS: float
    freeU: float
    inCompMax: float




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

    center: Optional[float]
    amplitude: Optional[float] = None
    width: Optional[float] = None
    background: Optional[float] = None

class PeakFitResult(ExtendedModel):

    raw_x: np.ndarray
    raw_y: np.ndarray
    peaks: Sequence[PeakInfo]
    fitted_curve: np.ndarray
    split_curve: np.ndarray

class ScatterResult(ExtendedModel):
    q_Intensity: np.ndarray
    q_idx: dict


class CVResult(ExtendedModel):
    data: np.ndarray
    lxlylz: np.ndarray
    NxNyNz: np.ndarray
    facets: list
    centers: np.ndarray





