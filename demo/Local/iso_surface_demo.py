
from src import IsoSurf
import os
import sys


iso3d = IsoSurf(os.path.join(os.path.dirname(__file__),
                    '../datasets/Iso'), inverse_flag=True, scale_flag=True)
iso3d.collect()
iso3d.iso3D('phi1', level=[0.5], backend='mayavi')

iso3d.iso3D('phi1', level=[0.5], backend='mpl')

iso3d.iso3D('phi1', level=[0.5], backend='vista')




# phi2d = IsoSurf(os.path.join(os.path.dirname(__file__),'../datasets/scatter'))
# phi2d.collect()
# phi2d.iso2D(['block0', 'block1', 'block2'])

