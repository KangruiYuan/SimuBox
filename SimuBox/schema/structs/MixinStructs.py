from typing import Optional

from matplotlib.axes import Axes
from matplotlib.figure import Figure
from pydantic import BaseModel

__all__ = ["MixinBaseModel", "FigureAxesMixin"]

class MixinBaseModel(BaseModel):
    """
    数据结构体模板，定义了所有结构体的属性
    """
    class Config:
        arbitrary_types_allowed = True


class FigureAxesMixin(MixinBaseModel):
    """
    画布类数据的基础数据结构体。
    """

    fig: Optional[Figure] = None
    ax: Optional[Axes] = None
