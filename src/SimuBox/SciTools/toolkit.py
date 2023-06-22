import numpy as np

class Tools():
    
    @classmethod
    def find_nearest_1d(array, value):
        array = np.asarray(array)
        idx = (np.abs(array - value)).argmin()
        return idx
    
    