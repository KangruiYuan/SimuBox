from .MixinEnums import StrEnum

__all__ = [
    "ColorMode",
    "CompareMode",
    "TopoCreateMode",
    "DetectionMode",
    "AnalyzeMode",
    "WeightedMode",
]


class ColorMode(StrEnum):
    """
    绘图的颜色模式。
    """

    RGB = "rgb"
    HEX = "hex"
    L = "l"
    RANDOM = "random"


class CompareMode(StrEnum):
    """
    曲线比较的模式。
    """

    DIFF = "diff"
    ABS = "abs"
    MULTI = "multi"


class TopoCreateMode(StrEnum):
    """
    创建拓扑结构的模式。
    """

    JSON = "json"
    AMBN = "AmBn"
    DENDRIMER = "dendrimer"
    LINEAR = "linear"
    STAR = "star"


class DetectionMode(StrEnum):
    """
    检测相边界的模式。
    """
    HORIZONTAL = "hori"
    VERTICAL = "vert"
    BOTH = "both"
    INTERP = "interp"
    MIX = "mix"


class AnalyzeMode(StrEnum):
    """
    Voronoi剖分的模式。
    """
    VORONOI = "voronoi"
    TRIANGLE = "triangle"
    WEIGHTED = "weighted"


class WeightedMode(StrEnum):
    """
    带权重的Voronoi图的绘制模式。
    """
    ADDITIVE = "additive"
    POWER = "power"
