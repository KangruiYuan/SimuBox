import pyvista as pv
import numpy as np
from src import InfoReader
import os

reader = InfoReader(os.path.join(os.path.dirname(__file__),
                    'datasets/Iso'), inverse_flag=True, scale_flag=True)
reader.collect()

x, y, z = np.mgrid[0:reader.lxlylz[0]:complex(0, reader.NxNyNz[0]),
                   0:reader.lxlylz[1]:complex(0, reader.NxNyNz[1]),
                   0:reader.lxlylz[2]:complex(0, reader.NxNyNz[2])]

vol = reader.phi1

grid = pv.StructuredGrid(x, y, z)
grid["vol"] = vol.flatten()
contours = grid.contour([0.3])

pv.set_plot_theme('document')
p = pv.Plotter()
# p.add_mesh(contours, scalars=contours.points[:, 2], show_scalar_bar=True)
p.add_mesh(contours, color='white')
p.show()
