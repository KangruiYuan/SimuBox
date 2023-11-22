import numpy as np
from .Schema import NumericType, VectorType, ColorType
from typing import Union
import matplotlib.pyplot as plt

def find_nearest_1d(array: VectorType, value: NumericType) -> NumericType:
    if not isinstance(array, np.ndarray):
        array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx





