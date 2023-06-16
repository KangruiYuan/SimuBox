import pyvista as pv
import numpy as np
from src import InfoReader
from src import IsoSurf
import os

iso = IsoSurf(os.path.join(os.path.dirname(__file__),
                    '../datasets/Iso'), inverse_flag=True, scale_flag=True)
iso.collect()
iso.isosurf('phi1', level=[0.3,0.7])