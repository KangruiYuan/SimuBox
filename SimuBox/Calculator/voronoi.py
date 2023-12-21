from collections import defaultdict
from itertools import product
from math import acos
from typing import Union, Tuple, Optional, Sequence

import cv2
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from scipy.spatial import Voronoi, voronoi_plot_2d
from scipy.spatial import distance, Delaunay
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor

from ..Artist import generate_colors
from ..Schema import (
    Density,
    OpenCVResult,
    AnalyzeMode,
    ColorMode,
    WeightedMode,
    VoronoiAnalyzeResult,
    PathLike,
)
from ..Toolkits import parse_density
from ..Artist import plot_savefig


class VoronoiCell:
    @staticmethod
    def OpenCV(
        density: Density,
        expand: Union[int, Sequence[int]] = 3,
        **kwargs,
    ):
        # ------------读取原生数据-----------------
        parsed = parse_density(density, expand=expand, **kwargs)
        mat = parsed.mat.squeeze()
        assert mat.ndim == 2, f"目前仅支持二维Voronoi剖分，当前矩阵维度为{mat.ndim}"
        gray_image = np.array(mat * 255, dtype=np.uint8)
        size = gray_image.shape
        rect = (0, 0, size[1], size[0])

        #  Subdiv2D类用于对一组 2D 点（表示为 Point2f 的向量）执行各种平面细分
        subdiv = cv2.Subdiv2D(rect)
        ret, thresh = cv2.threshold(gray_image, 127, 255, 0)
        # ------------二值化-----------------
        contours, hierarchy = cv2.findContours(
            thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
        )

        for c in contours:
            M = cv2.moments(c)
            cX = M["m10"] / (M["m00"] + 1e-6)
            cY = M["m01"] / (M["m00"] + 1e-6)
            subdiv.insert((cX, cY))
        # ------------获得图域中心-----------------
        # Output vector of the Voroni facets
        (facets, centers) = subdiv.getVoronoiFacetList([])
        centers = np.array(centers, dtype=float)
        # facets = np.array(facets, dtype=int)

        return OpenCVResult(
            path=density.path,
            parsed_density=parsed,
            facets=facets,
            centers=centers,
        )

    @staticmethod
    def in_lim(point, xlim, ylim) -> bool:
        return (
            (xlim[0] < point[:, 0])
            & (point[:, 0] < xlim[1])
            & (ylim[0] < point[:, 1])
            & (point[:, 1] < ylim[1])
        )

    @classmethod
    def Analyze(cls, density: Density, mode: AnalyzeMode, **kwargs):

        cv_res = cls.OpenCV(density, **kwargs)
        if "expand" in kwargs:
            kwargs.pop("expand")
        vecs = [
            np.linspace(0, l, N)
            for l, N in zip(cv_res.parsed_density.lxlylz, cv_res.parsed_density.NxNyNz)
        ]
        vecs = vecs[::-1]
        X, Y = np.meshgrid(*vecs)
        centers = np.around(cv_res.centers).astype(int)
        centers_lxly = np.stack(
            [X[centers[:, 1], centers[:, 0]], Y[centers[:, 1], centers[:, 0]]], axis=1
        )

        if mode == AnalyzeMode.VORONOI:
            return cls.voronoi(cv_res, centers_lxly, **kwargs)
        elif mode == AnalyzeMode.TRIANGLE:
            return cls.triangle(cv_res, centers_lxly, **kwargs)
        else:
            raise NotImplementedError(f"{mode} 解析模式尚未建立。")

    @staticmethod
    def get_lim_attributes(key: str, default: Sequence, expand: int, **kwargs):
        attr = kwargs.get(key, default)
        return np.array(attr) + expand / 2

    @classmethod
    def voronoi(
        cls,
        cv_res: OpenCVResult,
        centers: np.ndarray,
        interactive: bool = True,
        save: Union[PathLike, bool] = False,
        **kwargs,
    ):
        point_color = kwargs.get("pc", "b")
        line_color = kwargs.get("lc", "k")
        centers_cut = cls.set_point_limits(cv_res, centers, **kwargs)

        vor = Voronoi(centers, incremental=True)
        asp = cv_res.parsed_density.lxlylz[1] / cv_res.parsed_density.lxlylz[0]
        plt.figure()
        # plt.figure(figsize=kwargs.get("figsize", (6, 6 / asp)))
        ax = plt.gca()
        fig = voronoi_plot_2d(
            vor, ax, show_points=False, show_vertices=False, line_colors=line_color
        )
        if kwargs.get("point", True):
            plt.scatter(centers_cut[:, 0], centers_cut[:, 1], s=20, c=point_color)
        cls.set_axis_limits(cv_res, **kwargs)
        ax.set_aspect(1 / asp)
        plt.xticks([])
        plt.yticks([])
        plt.tight_layout()
        plot_savefig(
            cv_res,
            save=save,
            prefix="voronoi",
            suffix=str(cv_res.parsed_density.target[0]),
            **kwargs,
        )
        if interactive:
            plt.show()
        return VoronoiAnalyzeResult(
            path=cv_res.path, cv_result=cv_res, fig=fig, ax=ax, voronoi=vor
        )

    @classmethod
    def triangle(
        cls,
        cv_res: OpenCVResult,
        centers: np.ndarray,
        interactive: bool = True,
        save: Union[PathLike, bool] = False,
        **kwargs,
    ):
        point_color = kwargs.get("pc", "b")
        line_color = kwargs.get("lc", "k")
        centers_cut = cls.set_point_limits(cv_res, centers, **kwargs)

        theta_max = kwargs.get("theta_max", np.pi / 2)
        dist_matrix = distance.cdist(centers, centers, "euclidean")
        triangles_ori = Delaunay(centers).simplices
        triangles = []
        for t in triangles_ori:
            e12 = dist_matrix[t[0], t[1]]
            e13 = dist_matrix[t[0], t[2]]
            e23 = dist_matrix[t[1], t[2]]

            a, b, c = sorted([e12, e13, e23])
            costh = (a**2 + b**2 - c**2) / (2 * a * b)
            theta = acos(costh)
            if theta > theta_max:
                continue
            triangles.append(t)

        coord_dict = defaultdict(set)
        for t in triangles:
            coord_dict[t[0]].update([t[1], t[2]])
            coord_dict[t[1]].update([t[0], t[2]])
            coord_dict[t[2]].update([t[0], t[1]])

        asp = cv_res.parsed_density.lxlylz[1] / cv_res.parsed_density.lxlylz[0]
        fig = plt.figure(figsize=kwargs.get("figsize", (6, 6 / asp)))
        ax = plt.gca()
        plt.triplot(
            centers[:, 0], centers[:, 1], triangles, linewidth=1, color=line_color
        )
        if kwargs.get("point", True):
            plt.scatter(centers_cut[:, 0], centers_cut[:, 1], s=20, c=point_color)
        cls.set_axis_limits(cv_res, **kwargs)
        plt.xticks([])
        plt.yticks([])
        ax.set_aspect(asp)
        plt.tight_layout()
        plot_savefig(
            cv_res,
            save=save,
            prefix="triangle",
            suffix=str(cv_res.parsed_density.target[0]),
            **kwargs,
        )
        if interactive:
            plt.show()
        return VoronoiAnalyzeResult(
            path=cv_res.path,
            cv_result=cv_res,
            fig=fig,
            ax=ax,
            triangle=triangles,
            coord_dict=coord_dict,
        )

    @classmethod
    def set_point_limits(cls, density: OpenCVResult, centers: np.ndarray, **kwargs):
        point_xlim = cls.get_lim_attributes(
            "point_xlim", (-0.1, 1.1), density.parsed_density.expand[0], **kwargs
        )
        point_ylim = cls.get_lim_attributes(
            "point_ylim", (-0.1, 1.1), density.parsed_density.expand[1], **kwargs
        )

        mask = cls.in_lim(
            point=centers,
            xlim=density.parsed_density.raw_lxlylz[1] * point_xlim,
            ylim=density.parsed_density.raw_lxlylz[0] * point_ylim,
        )
        centers_cut = centers[mask]
        return centers_cut

    @classmethod
    def set_axis_limits(cls, density: OpenCVResult, **kwargs):
        axis_xlim = cls.get_lim_attributes(
            "axis_xlim", (-0.5, 1.5), density.parsed_density.expand[0], **kwargs
        )
        axis_ylim = cls.get_lim_attributes(
            "axis_ylim", (-0.5, 1.5), density.parsed_density.expand[1], **kwargs
        )
        plt.xlim(density.parsed_density.raw_lxlylz[1] * axis_xlim)
        plt.ylim(density.parsed_density.raw_lxlylz[0] * axis_ylim)

    @staticmethod
    def additive(points, arr, weights):
        dis = np.linalg.norm(points - arr, axis=1) - weights
        return dis

    @staticmethod
    def power(points, arr, weights):
        dis = (
            np.linalg.norm(points - arr + np.arange(len(points))[:, np.newaxis], axis=1)
            ** 2
            - weights
        )
        return dis

    @classmethod
    def weighted_voronoi_diagrams(
        cls,
        points: Union[list, np.ndarray],
        weights: Optional[Union[float, int, list, np.ndarray]] = 0,
        method: WeightedMode = "additive",
        size: Tuple = (500, 500),
        color_mode: ColorMode = "L",
        interactive: bool = True,
        num_threads: int = 8,
        **kwargs,
    ):
        image = Image.new(color_mode, size)
        putpixel = image.putpixel
        imgx, imgy = image.size
        if isinstance(points, list):
            points = np.array(points)

        if isinstance(weights, (list, np.ndarray)):
            weights = np.array(weights)
        elif isinstance(weights, (int, float)):
            pass
        else:
            raise ValueError(f"Weights must be an array")

        colors = generate_colors(mode=color_mode, num=len(points), **kwargs)

        arrs = np.array(list(product(range(imgx), range(imgy))), dtype=int)

        func = getattr(
            cls, method.value if isinstance(method, WeightedMode) else method
        )
        for arr in tqdm(arrs):
            dis = func(points, arr, weights)
            min_idx = np.argmin(dis)
            putpixel(arr, colors[min_idx])

        # def process_arr(arr):
        #     nonlocal func
        #     dis = func(points, arr, weights)
        #     min_idx = np.argmin(dis)
        #     putpixel(arr, colors[min_idx])
        #
        # with ThreadPoolExecutor(max_workers=num_threads) as executor:
        #     executor.map(process_arr, arrs)

        plot = kwargs.get("plot", "imshow")
        im = np.array(image)
        fig = plt.figure(figsize=kwargs.get("figsize", (5, 5)))
        ax = plt.gca()
        plt.scatter(
            points[:, 0],
            points[:, 1],
            s=max(size) * 0.1,
            c="white",
            zorder=10,
            edgecolors="black",
        )
        if plot == "imshow":
            if color_mode == ColorMode.L:
                plt.imshow(im, "gray", zorder=5, origin="lower")
            elif color_mode == ColorMode.RGB:
                plt.imshow(im, zorder=5, origin="lower")
        elif plot == "vertices":
            if color_mode == "RGB":
                im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
            canny = cv2.Canny(im, threshold1=40, threshold2=80, apertureSize=5)
            plt.imshow(canny, "gray", zorder=5)
        if interactive:
            plt.show()
        return fig, ax
