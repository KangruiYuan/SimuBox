import numpy as np
from .Schema import NumericType, VectorType, ColorType
from typing import Union
import matplotlib.pyplot as plt

def find_nearest_1d(array: VectorType, value: NumericType) -> NumericType:
    if not isinstance(array, np.ndarray):
        array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx

def generate_colors(mode: Union[str, ColorType] = ColorType.RGB, num: int = 1):
    if mode == ColorType.RGB:
        color = np.random.choice(range(256), size=(num, 3)).tolist()
    elif mode == ColorType.HEX:
        color = ["#" + "".join(i) for i in np.random.choice(list("0123456789ABCDEF"), size=(num, 6))]
    else:
        raise NotImplementedError(mode.value)
    return color

def init_plot_config(config: dict):
    plt.rcParams.update(config)

