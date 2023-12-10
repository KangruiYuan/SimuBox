from pathlib import Path
from typing import Optional, Any, Sequence, Union

import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from pydantic import BaseModel
from scipy.spatial import Voronoi


class MixinBaseModel(BaseModel):
    class Config:
        arbitrary_types_allowed = True


class FigureAxesMixin(MixinBaseModel):

    fig: Optional[Figure] = None
    ax: Optional[Axes] = None

class PointMixin(MixinBaseModel):

    x :Optional[float] = None
    y :Optional[float] = None
    z :Optional[float] = None
    value: Optional[float] = None


class Printout(MixinBaseModel):
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


class FetData(MixinBaseModel):
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


class Density(MixinBaseModel):
    path: Optional[Path] = None
    data: pd.DataFrame
    NxNyNz: Optional[np.ndarray] = None
    lxlylz: Optional[np.ndarray] = None
    shape: Optional[np.ndarray] = None

    def repair_from_printout(self, printout: Printout):
        if self.NxNyNz is not None:
            mask = self.NxNyNz != 1
        else:
            mask = None
        if self.lxlylz is None or (all(self.lxlylz == 1) and any(printout.lxlylz != 1)):
            self.lxlylz = printout.lxlylz if mask is None else printout.lxlylz[mask]

    def repair_from_fet(self, fet: FetData):
        if self.NxNyNz is not None:
            if self.NxNyNz is not None:
                mask = self.NxNyNz != 1
            else:
                mask = None
        lxlylz = np.array([fet.lx, fet.ly, fet.lz])
        if self.lxlylz is None or (all(self.lxlylz == 1) and any(lxlylz != 1)):
            self.lxlylz = lxlylz if mask is None else lxlylz[mask]


class Summation(MixinBaseModel):
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


class CompareResult(FigureAxesMixin):
    df: pd.DataFrame
    plot_dict: dict
    mat: np.ndarray


class LandscapeResult(FigureAxesMixin):
    mat: np.ndarray
    x_ticks: np.ndarray
    y_ticks: np.ndarray
    levels: np.ndarray
    ticks: np.ndarray
    contourf_fig: Optional[Any] = None
    contour_fig: Optional[Any] = None
    clb: Optional[Any] = None


class PhasePointData(MixinBaseModel):
    phase: str
    x: float
    y: float
    val: float


class PeakData(MixinBaseModel):

    center: Optional[float]
    amplitude: Optional[float] = None
    width: Optional[float] = None
    background: Optional[float] = None
    fix: Optional[Sequence[bool]] = (False, False, False)
    area: Optional[float] = None


class PeakFitResult(FigureAxesMixin):

    x: np.ndarray
    y: np.ndarray
    peaks: Sequence[PeakData]
    fitted_curve: np.ndarray
    split_curve: np.ndarray
    r2: float
    adj_r2: float
    area: float


class DensityParseResult(MixinBaseModel):
    path: Path
    raw_NxNyNz: np.ndarray
    raw_lxlylz: np.ndarray
    raw_mat: np.ndarray
    lxlylz: np.ndarray
    NxNyNz: np.ndarray
    mat: np.ndarray
    expand: np.ndarray
    target: list[int]


class ScatterResult(MixinBaseModel):
    parsed_density: DensityParseResult
    q_Intensity: np.ndarray
    q_idx: dict


class ScatterPlot(FigureAxesMixin):
    peaks_location: np.ndarray
    peaks_height: dict
    plot_y: np.ndarray
    plot_x: np.ndarray


class OpenCVResult(MixinBaseModel):
    path: Path
    parsed_density: DensityParseResult
    facets: list
    centers: np.ndarray


class VoronoiAnalyzeResult(FigureAxesMixin):
    path: Path
    cv_result: OpenCVResult
    voronoi: Optional[Voronoi] = None
    triangle: Optional[list] = None
    coord_dict: Optional[dict] = None


class XMLRaw(MixinBaseModel):
    path: Path
    NxNyNz: np.ndarray
    lxlylz: np.ndarray
    grid_spacing: np.ndarray
    data: pd.DataFrame
    atoms_type: list
    atoms_mapping: dict[str, int]


class XMLTransResult(MixinBaseModel):
    xml: XMLRaw
    phi: np.ndarray

    def write(
        self,
        phi: Optional[np.ndarray] = None,
        path: Optional[Union[str, Path]] = None,
        scale: bool = False,
    ):
        if phi is None:
            phi = self.phi
        if scale:
            phi = phi / phi.sum(axis=0)
        phi = np.c_[[i.reshape((-1, 1)) for i in phi]]
        phi = np.squeeze(phi).T
        if path is None:
            path = self.xml.path.with_suffix(".txt")
        np.savetxt(
            path,
            phi,
            fmt="%.3f",
            delimiter=" ",
            header=" ".join(map(str, self.xml.NxNyNz)),
            comments="",
        )


class LineData(MixinBaseModel):
    x: np.ndarray
    y: np.ndarray
    label: str


class LineCompareResult(FigureAxesMixin):
    lines: list[LineData]
    xlabel: str
    ylabel: str


class ODTResult(FigureAxesMixin):

    xN: np.ndarray
    f: np.ndarray
    expression: Any
    xlabel: str
    ylabel: str


class TopoPlot(FigureAxesMixin):
    kind_color: dict[str, str]
    rad: float
