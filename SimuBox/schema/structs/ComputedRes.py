from pathlib import Path

from .MixinStructs import FigureAxesMixin, MixinBaseModel
from .DataFile import DensityParseResult
import pandas as pd
import numpy as np
from typing import Optional, Sequence, Any
from scipy.spatial import Voronoi


class Peak(MixinBaseModel):
    center: Optional[float]
    amplitude: Optional[float] = None
    width: Optional[float] = None
    background: Optional[float] = None
    fix: Sequence[bool] = (False, False, False)
    area: Optional[float] = None

class PeakFitResult(FigureAxesMixin):
    x: np.ndarray
    y: np.ndarray
    peaks: Sequence[Peak]
    fitted_curve: np.ndarray
    split_curve: np.ndarray
    r2: float
    adj_r2: float
    area: float

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

class ODTResult(FigureAxesMixin):

    xN: np.ndarray
    f: np.ndarray
    expression: Any
    xlabel: str
    ylabel: str

class TopoPlot(FigureAxesMixin):
    kind_color: dict[str, str]
    rad: float
