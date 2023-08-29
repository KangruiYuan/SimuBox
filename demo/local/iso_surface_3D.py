
from src import IsoSurf
import os


iso3d = IsoSurf(os.path.join(os.path.dirname(__file__),
                    '../datasets/Iso'), inverse_flag=True, scale_flag=True)
iso3d.collect()
iso3d.iso3D('phi1', level=[0.5], backend='mayavi')

iso3d.iso3D('phi1', level=[0.5], backend='mpl')

iso3d.iso3D('phi1', level=[0.5], backend='vista')





