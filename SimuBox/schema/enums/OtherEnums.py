from .MixinEnums import StrEnum

__all__ = [
    "Operator",
    "OPERATOR_FUNCTION_MAP",
]


class Operator(StrEnum):
    """
    定义了可对DataFrame对象进行的操作。
    """

    add = "+"
    sub = "-"
    mul = "*"
    div = "/"
    pow = "**"
    mod = "%"


OPERATOR_FUNCTION_MAP = {
    Operator.add: lambda a, b: a + b,
    Operator.sub: lambda a, b: a - b,
    Operator.mul: lambda a, b: a * b,
    Operator.div: lambda a, b: a / b,
    Operator.pow: lambda a, b: a**b,
    Operator.mod: lambda a, b: a % b,
}
