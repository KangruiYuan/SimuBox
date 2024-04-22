import tkinter as tk
from tkinter import filedialog
from SimuBox import read_density
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import math


def find_closest_product(n):
    start = math.isqrt(n)  # 开始的整数从平方根开始，以确保两个因子接近
    while True:
        if start**2 >= n:
            return start, start
        elif start * (start + 1) >= n:
            return start, (start + 1)
        start += 1


if __name__ == "__main__":

    root = tk.Tk()
    root.withdraw()
    paths = []
    while True:
        file_path = filedialog.askopenfilename()
        if not file_path:
            break
        paths.append(Path(file_path))

    selected_column = 0
    mask_column = -1
    color_map = "jet"
    strict = 1

    grid = find_closest_product(len(paths))
    fig, axes = plt.subplots(*grid, figsize=(grid[0] * 4, grid[1] * 4))
    axes = axes.flatten() if len(paths) > 1 else [axes]
    min_v = 1e5
    max_v = -1e5
    images = []
    for i in range(len(paths)):
        path = paths[i]
        density = read_density(path)
        data = density.data.values
        shape = density.NxNyNz[density.NxNyNz != 1]

        stress = data[:, selected_column]

        stress = stress.reshape(shape)
        mask = data[:, mask_column].reshape(shape)


        mask_stress = stress[mask == 1]

        std = np.std(mask_stress)
        mean = np.mean(mask_stress)

        stress[stress < mean - strict * std] = 0
        stress[stress > mean + strict * std] = 0

        min_v = min(min_v, stress.min())
        max_v = max(max_v, stress.max())

        masked_image = np.ma.masked_where(mask == 0, stress)
        images.append(masked_image)

    for i in range(len(paths)):
        path = paths[i]
        masked_image = images[i]
        im = axes[i].imshow(masked_image, cmap=color_map, vmax=max_v, vmin=min_v)
        plt.colorbar(im, fraction=0.05)
        axes[i].set_title(path.parent.name)

    for i in range(len(paths), len(axes)):
        axes[i].axis("off")

    fig.tight_layout()
    plt.show()
