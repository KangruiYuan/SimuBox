from .MixinEnums import StrEnum, IntEnum

__all__ = ["Cells", "Ensemble", "Servers"]

class Servers(StrEnum):
    """
    选择运算服务器。
    """
    cpuTOPS = "cpuTOPS"
    gpuSCFT = "gpuSCFT"
    gpuTOPS = "gpuTOPS"

class Ensemble(StrEnum):
    """
    正则/巨正则系综
    """
    CANONICAL = "CANONICAL"
    GRANDCANONICAL = "GRANDCANONICAL"

class Cells(IntEnum):
    """
    枚举元胞参数。
    """

    lx = 0
    ly = 1
    lz = 2
    lyx = 3
    lzx = 4
    lzy = 5