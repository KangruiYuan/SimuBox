

from typing import Optional
from .MixinStructs import MixinBaseModel
from ..enums import Operator

__all__ = ['Operation']

class Operation(MixinBaseModel):
    """
    用于定义对DataFrame执行的操作。
    """

    left: str
    right: Optional[str] = None
    factor: Optional[float] = None
    operator: Operator
    name: Optional[str] = None
    accuracy: int = 3

    @property
    def _name(self):
        if self.name is not None:
            return self.name
        assert self.right is not None or self.factor is not None
        return (
            self.left + self.operator.value + str(self.factor)
            if self.right is None
            else self.right
        )