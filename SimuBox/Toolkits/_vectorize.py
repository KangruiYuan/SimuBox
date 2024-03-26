from copy import deepcopy
from typing import Any, Optional

import numpy as np


def verify_scalar(value: Any):
    """
    对参数值进行检验，如果是标量，返回True；如果是向量，返回False
    :param value: 任何形式的参数
    :return: 布尔值，True或者False

    >>> verify_scalar(1)
    True
    >>> verify_scalar(3.14)
    True
    >>> verify_scalar("p")
    True
    >>> verify_scalar(1+2j)
    True
    >>> verify_scalar([1,1])
    False
    """

    scalar_types = (int, float, complex, str)
    return isinstance(value, scalar_types)


def vectorize(value, length: Optional[int] = None, fill_value: Optional[Any] = None):
    """
    :param value:
    :param length:
    :param fill_value:
    :return:
    >>> vectorize(1)
    [1]
    >>> vectorize("p")
    ['p']
    >>> vectorize([1,2,3], 2)
    [1, 2]
    >>> vectorize((1,2,3), 2)
    (1, 2)
    >>> vectorize(np.array([1,2,3]), 5, 8)
    array([1, 2, 3, 8, 8])
    >>> vectorize([1,2,3], 5, "p")
    [1, 2, 3, 'p', 'p']
    >>> vectorize((1,2,3), 4, 4)
    (1, 2, 3, 4)
    """
    vector = [value] if verify_scalar(value) else deepcopy(value)
    if length is not None:
        if length <= len(vector):
            return vector[:length]
        else:
            padding_length = length - len(vector)
            if fill_value is None:
                padding = [vector[-1]] * padding_length
            else:
                padding = [fill_value] * padding_length
            if isinstance(vector, np.ndarray):
                padding = np.array(padding)
                return np.concatenate((vector, padding))
            elif isinstance(vector, list):
                return vector + padding
            elif isinstance(vector, tuple):
                return tuple(vector + tuple(padding))
            else:
                raise ValueError("Unsupported vector type")
    else:
        return vector


if __name__ == "__main__":
    import doctest

    doctest.testmod()
