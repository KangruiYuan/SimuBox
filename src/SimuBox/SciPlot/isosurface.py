
from ..SciTools import InfoReader
from skimage.measure import marching_cubes
import pyvista as pv
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Tuple, Sequence, List, Union

class IsoSurf(InfoReader):

    def __int__(self, path, **kwargs):
        super().__init__(path, **kwargs)

    def isosurf(self, phi:Union[str, np.ndarray],
                level:Union[float,List[float]]=0.5,
                backend:str = 'vista',
                **kwargs):

        vol = getattr(self, phi) if type(phi) == str else phi
        lxlylz = getattr(self, 'lxlylz')
        NxNyNz = getattr(self, 'NxNyNz')

        x, y, z = np.mgrid[
                  0:lxlylz[0]:complex(0, NxNyNz[0]),
                  0:lxlylz[1]:complex(0, NxNyNz[1]),
                  0:lxlylz[2]:complex(0, NxNyNz[2])]

        if backend == 'vista':
            grid = pv.StructuredGrid(x, y, z)
            grid["vol"] = vol.flatten()
            _level = level if type(level) == list else [level]
            contours = grid.contour(_level)
            pv.set_plot_theme(kwargs.get('theme', 'document'))
            p = pv.Plotter(line_smoothing=True,
                           polygon_smoothing=True,
                           lighting='light kit')
            p.add_mesh(contours,
                       color=kwargs.get('color', 'white'),
                       show_edges=False,
                       style='surface',
                       smooth_shading=True,
                       # Enable physics based rendering(PBR)
                       pbr=True,
                       roughness=0
                       )
            p.show()
        elif backend == 'mpl':
            assert type(level) == float, "等值面应为浮点数"
            verts, faces, _, _ = marching_cubes(
                vol, level, spacing=(1, 1, 1))
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            ax.plot_trisurf(verts[:, 0], verts[:, 1], faces, verts[:, 2], lw=1)
            plt.show()
