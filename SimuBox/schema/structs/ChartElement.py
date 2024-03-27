import numpy as np
import pandas as pd

from .MixinStructs import MixinBaseModel, FigureAxesMixin
from typing import Optional, Any
from matplotlib.contour import QuadContourSet
from matplotlib.colorbar import Colorbar

__all__ = [
    "Point",
    "Line",
    "LandscapeResult",
    "LineCompareResult",
    "Contour",
    "PhaseCompareResult",
]


class Point(MixinBaseModel):
    """
    点类数据的基础数据结构体
    """

    x: Optional[float] = None
    y: Optional[float] = None
    z: Optional[float] = None
    value: Optional[float] = None
    label: Optional[str] = ""


class Line(MixinBaseModel):
    """
    线类数据的基础数据结构体
    """

    x: Optional[np.ndarray] = None
    y: Optional[np.ndarray] = None
    z: Optional[np.ndarray] = None
    value: Optional[np.ndarray] = None
    label: Optional[str] = ""


class PhaseCompareResult(FigureAxesMixin):
    data: pd.DataFrame
    mat: np.ndarray
    phase_map: dict


class LineCompareResult(FigureAxesMixin):
    """
    曲线比较结果
    """

    lines: list[Line]
    xlabel: str
    ylabel: str


class Contour(MixinBaseModel):
    """
    等高线图中等高线的轮廓数据结构体
    """

    level: float
    area: float
    length: float
    IQ: float
    vertices: Any


class LandscapeResult(FigureAxesMixin):
    """
    用于绘制等高线图的数据结构体
    """

    mat: np.ndarray
    xticks: np.ndarray
    yticks: np.ndarray
    levels: np.ndarray
    contourf_fig: Optional[QuadContourSet] = None
    contour_fig: Optional[QuadContourSet] = None
    clb: Optional[Colorbar] = None
    IQs: Optional[list[Contour]] = None
