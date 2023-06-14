import numpy as np
from skimage.measure import marching_cubes
import matplotlib.pyplot as plt
from src import InfoReader
import os

reader = InfoReader(os.path.join(os.path.dirname(__file__), '../datasets/Iso'))
reader.collect()

x, y, z = np.mgrid[0:reader.lxlylz[0]:complex(0, reader.NxNyNz[0]), 
                   0:reader.lxlylz[1]:complex(0, reader.NxNyNz[1]),
                   0:reader.lxlylz[2]:complex(0, reader.NxNyNz[2])]

vol = reader.phi0
iso_val = 0.5
verts, faces, _, _ = marching_cubes(
    vol, iso_val, spacing=(0.1, 0.1, 0.1))

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_trisurf(verts[:, 0], verts[:, 1], faces, verts[:, 2], lw=1)
plt.show()
