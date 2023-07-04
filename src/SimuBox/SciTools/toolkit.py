import numpy as np

class Tools():
    
    @classmethod
    def find_nearest_1d(cls, array, value):
        array = np.asarray(array)
        idx = (np.abs(array - value)).argmin()
        return idx
    
    