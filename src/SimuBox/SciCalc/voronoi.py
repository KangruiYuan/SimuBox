
import cv2
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import distance, Delaunay
from math import acos
from collections import defaultdict
from scipy.spatial import Voronoi, voronoi_plot_2d
from tqdm import tqdm
from ..SciTools import InfoReader
from itertools import product
from typing import Union, Tuple, Optional
from PIL import Image

__all__ = ['VoronoiCell']

class VoronoiCell(InfoReader):

    def __init__(self, path="", name_flag:bool=False, inverse_flag:bool=False) -> None:
        super().__init__(path, inverse_flag=inverse_flag, name_flag=name_flag)
        pass

    def OpenCV(self, phi, plot=True):
        # ------------读取原生数据-----------------
        phaMAT_resize, lxlylz, NxNyNz = self.tile(
            phi, expand=(3, 3))
        gray_image = np.array(phaMAT_resize * 255, dtype=np.uint8)
        phaMAT_orig = gray_image.copy()
        size = gray_image.shape
        # print(size)
        rect = (0, 0, size[1], size[0])

        #  Subdiv2D类用于对一组 2D 点（表示为 Point2f 的向量）执行各种平面细分
        subdiv = cv2.Subdiv2D(rect)
        ret, thresh = cv2.threshold(gray_image, 127, 255, 0)
        # ------------二值化-----------------
        contours, hierarchy = cv2.findContours(
            thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        for c in contours:
            M = cv2.moments(c)
            cX = M["m10"] / (M["m00"]+1e-6)
            cY = M["m01"] / (M["m00"]+1e-6)
            subdiv.insert((cX, cY))
        # ------------获得图域中心-----------------
        img_voronoi = np.zeros(phaMAT_orig.shape, dtype=phaMAT_orig.dtype)
        # Output vector of the Voroni facets
        (facets, centers) = subdiv.getVoronoiFacetList([])
        # centers = list(np.array(centers, dtype=np.uint8))
        centers_float = np.array(centers)
        if plot:
            centers = centers.astype(np.uint32)
            for i in range(0, len(facets)):
                ifacet_arr = []
                for f in facets[i]:
                    ifacet_arr.append(f)

                ifacet = np.array(ifacet_arr, np.int32)
                color = (251, 239, 0)

                cv2.fillConvexPoly(img_voronoi, ifacet, color, cv2.LINE_AA, 0)
                ifacets = np.array([ifacet])
                cv2.polylines(img_voronoi, ifacets, True,
                            (0, 0, 0), 1, cv2.LINE_AA, 0)
                cv2.circle(img_voronoi, (centers[i][0], centers[i][1]),
                        3, (0, 0, 0), cv2.FILLED, cv2.LINE_AA, 0)
            self.img_voronoi = img_voronoi
            plt.imshow(self.img_voronoi, aspect=lxlylz[2]/lxlylz[1], interpolation='spline16')
        
        self.size = size
        self.facets = facets
        self.centers = centers_float
        return lxlylz, NxNyNz
    
    @staticmethod
    def pnpoly(vertices, testp):
        n = len(vertices)
        j = n - 1
        res = False
        for i in range(n):
            if (vertices[i][1] > testp[1]) != (vertices[j][1] > testp[1]) and \
                    testp[0] < (vertices[j][0] - vertices[i][0]) * (testp[1] - vertices[i][1]) / (
                    vertices[j][1] - vertices[i][1]) + vertices[i][0]:
                res = not res
            j = i
        return res

    @staticmethod
    def prolongation(ori_mat, lx, ly):
        positions_period = np.empty(shape=[0, 2])
        for idx_x in range(-1, 2):
            for idx_y in range(-1, 2):
                offset = np.stack([np.ones(len(ori_mat)) * idx_x * lx,
                                np.ones(len(ori_mat)) * idx_y * ly], axis=1)
                positions_period = np.append(
                    positions_period, ori_mat + offset, axis=0)
        return positions_period
    
    @staticmethod
    def in_lim(point, xlim, ylim) -> bool:
        return (xlim[0] < point[:, 0]) & (point[:, 0] < xlim[1]) & (ylim[0] < point[:, 1]) & (point[:, 1] < ylim[1])
    
    def Voronoi(self, phi, **kwargs):
        
        lxlylz, NxNyNz = self.OpenCV(phi, plot=False)
        X, Y = self.coordsMap(lxlylz, NxNyNz)
        centers = np.around(self.centers)
        centers = centers.astype(int)
        centers_lxly = np.stack([X[centers[:, 1], centers[:, 0]],
                                Y[centers[:, 1], centers[:, 0]]], axis=1)
        
        self.centers_lxly = centers_lxly
        
        mask = self.in_lim(point=centers_lxly,
                    xlim=(0.9*self.lxlylz[2], 2.1*self.lxlylz[2]),
                    ylim=(0.9*self.lxlylz[1], 2.1*self.lxlylz[1]))
        centers_cut = centers_lxly[mask]
        
        self.vor = Voronoi(centers_lxly, incremental=True)

        plt.figure()
        # plt.figure(figsize=kwargs.get('figsize', (6, 6)))
        ax = plt.gca()
        fig = voronoi_plot_2d(
            self.vor, ax,
            show_points=False,
            show_vertices=False, )
        
        plt.scatter(centers_cut[:, 0], centers_cut[:, 1], s=20, c='k')
        plt.xlim(self.lxlylz[2]*0.5, 2.5*self.lxlylz[2])
        plt.ylim(self.lxlylz[1]*0.5, 2.5*self.lxlylz[1])
        plt.xticks([])
        plt.yticks([])
        plt.tight_layout()
        plt.show()
    
    def stats(self, thresh=0.3):
        # 每条边对应的点的索引
        # ridge_vertices = self.vor.ridge_vertices
        # V cell顶角对应坐标
        vertices = self.vor.vertices
        # ridge_points = self.vor.ridge_points
        regions = self.vor.regions
        # VC中心的points
        points = self.vor.points
        # 与points对应的region index
        point_region = self.vor.point_region
        mask_points = self.in_lim(point=points,
                     xlim=(self.lxlylz[1], 2*self.lxlylz[1]),
                     ylim=(self.lxlylz[2], 2*self.lxlylz[2]))
        # points_cut = points[mask_points]
        point_region_cut = point_region[mask_points]
        
        CN_VC = []
        for i in point_region_cut:
            tmp_region = vertices[regions[i]]
            lengths = []
            for j in range(-1, len(tmp_region)-1):
                lengths.append(np.linalg.norm(tmp_region[j]-tmp_region[j+1]))
            CN_VC.append(np.sum(np.array(lengths) >= thresh))
        return CN_VC

    def trianglesCal(self, theta_max=np.pi/2):
        # 距离矩阵
        # dist_matrix = distance.cdist(self.centers, self.centers, "euclidean")
        dist_matrix = distance.cdist(self.centers_lxly, self.centers_lxly, "euclidean")

        # 三角剖分
        triangles_ori = Delaunay(self.centers_lxly).simplices

        # 排除最大角度大于给定值的三角形
        triangles = []
        for t in triangles_ori:
            e12 = dist_matrix[t[0], t[1]]
            e13 = dist_matrix[t[0], t[2]]
            e23 = dist_matrix[t[1], t[2]]

            a, b, c = sorted([e12, e13, e23])
            costh = (a ** 2 + b ** 2 - c ** 2) / (2 * a * b)
            theta = acos(costh)
            if theta > theta_max:
                continue
            triangles.append(t)
        return triangles
    
    def plotTri(self, **kwargs):
        
        triangles = self.trianglesCal()
        self.triangles = triangles
        coord_dict = defaultdict(set)
        for t in triangles:
            coord_dict[t[0]].update([t[1], t[2]])
            coord_dict[t[1]].update([t[0], t[2]])
            coord_dict[t[2]].update([t[0], t[1]])
        
        plt.figure(figsize=(6,6))
        tri_X = self.centers_lxly[:, 0] - self.lxlylz[1]
        tri_Y = self.centers_lxly[:, 1] - self.lxlylz[2]
        plt.triplot(
            tri_X,
            tri_Y,
            triangles,
            linewidth=1,
            color='b')
        plt.xlim(xmin=0, xmax=self.lxlylz[1])
        plt.ylim(ymin=0, ymax=self.lxlylz[2])
        plt.show()

    @staticmethod
    def additive(points, arr, weights):
        dis = np.linalg.norm(points - arr, axis=1) - weights
        return dis

    @staticmethod
    def power(points, arr, weights):
        dis = np.linalg.norm(points - arr + np.arange(len(points))[:,np.newaxis], axis=1)**2 - weights
        return dis

    def weighted_voronoi_diagrams(self,
                                 points: Union[list, np.ndarray],
                                 weights: Optional[Union[float, int, list, np.ndarray]] = 0 ,
                                 method='additive',
                                 size: Tuple = (500, 500),
                                 colors:str ='linear',
                                 color_mode:str = 'L',
                                             **kwargs):
        """
        Args:
            points:
            weights:
            size:
            colors:
            mode:
            **kwargs:

        Returns:

        """
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

        if isinstance(colors, str):
            if color_mode == 'RGB':
                colors = np.random.randint(0, 255, size=(len(points), 3))
                colors = list(map(tuple, colors))
            elif color_mode == 'L':
                if colors == 'linear':
                    colors = np.linspace(0, 255, len(points), dtype=int).tolist()
                elif colors == 'random':
                    colors = np.random.randint(0, 255, size=(len(points)), dtype=int).tolist()

        arrs = np.array(list(product(range(imgx), range(imgy))), dtype=int)

        func = getattr(self, method)
        for arr in tqdm(arrs):
            dis = func(points, arr, weights)
            min_idx = np.argmin(dis)
            putpixel(arr, colors[min_idx])
        plot = kwargs.get('plot', 'imshow')
        im = np.array(image)
        plt.figure(figsize=kwargs.get('figsize', (8, 8)))
        plt.scatter(points[:, 0], points[:, 1], s=max(size) * 0.1, c='white')
        if plot == 'imshow':
            if color_mode == 'L':
                plt.imshow(im, 'gray')
            elif color_mode == 'RGB':
                plt.imshow(im)
        elif plot == 'vertices':
            if color_mode == 'RGB':
                im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
            canny = cv2.Canny(im, threshold1=40, threshold2=80, apertureSize=5)
            plt.imshow(canny, 'gray')
        plt.show()
        return image
        
                
class TYPE():
    C6_coords = np.flip(np.array([[33.5, 103.25],
                    [64.,  88.],
                    [93.5, 102.75],
                    [93.5, 153.25],
                    [64., 168.],
                    [33.5, 152.75]]), axis=1)
