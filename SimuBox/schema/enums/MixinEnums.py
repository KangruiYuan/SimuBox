from enum import Enum

__all__ = ["ExtendedEnum", "StrEnum", "IntEnum"]

from typing import Any


class ExtendedEnum(Enum):
    """
    定义基础的枚举类。
    """
    @classmethod
    def names(cls) -> list:
        """

        :return: 包含所有枚举元素名称的列表。
        """
        return [c.name for c in cls]

    @classmethod
    def values(cls) -> list:
        """

        :return: 包含所有枚举元素值的列表。
        """
        return [c.value for c in cls]

    @classmethod
    def items(cls) -> list[tuple[str, Any]]:
        """

        :return: 包含所有枚举元素(名称，值)的列表。
        """
        return [(c.name, c.value) for c in cls]


class StrEnum(str, ExtendedEnum):
    """
    值为字符串类型的枚举基类。
    """
    ...


class IntEnum(int, ExtendedEnum):
    """
    值为整型的枚举基类。
    """
    ...