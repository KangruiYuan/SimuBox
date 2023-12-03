from enum import Enum


class ExtendedEnum(Enum):
    @classmethod
    def names(cls):
        return [c.name for c in cls]

    @classmethod
    def values(cls):
        return [c.value for c in cls]

    @classmethod
    def items(cls):
        return [(c.name, c.value) for c in cls]


class StrEnum(str, ExtendedEnum):
    ...


class IntEnum(int, ExtendedEnum):
    ...


class ColorType(StrEnum):
    RGB = "RGB"
    HEX = "HEX"
    L = "L"
    # RANDOM = "RANDOM"


class DetectionMode(StrEnum):
    HORIZONTAL = "hori"
    VERTICAL = "vert"
    BOTH = "both"
    INTERP = "interp"
    MIX = "mix"

class AnalyzeMode(StrEnum):
    VORONOI = "voronoi"
    TRIANGLE = "triangle"
    WEIGHTED = "weighted"


class WeightedMethod(StrEnum):
    additive = "additive"
    power = "power"
