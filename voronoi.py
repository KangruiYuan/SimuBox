
import cv2
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import distance, Delaunay
from math import acos
from collections import defaultdict
from scipy.spatial import Voronoi, voronoi_plot_2d
from tqdm import tqdm
from .tools import InfoReader

__all__ = ['VoronoiCell']

class VoronoiCell(InfoReader):

    def __init__(self, path, name_flag:bool=False, inverse_flag:bool=False) -> None:
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

    # def genMask(self):
    #     phase = self.info.phase
    #     maskMat = np.zeros(shape=self.size)
        
    #     if phase == 'C6':
    #         for i in range(maskMat.shape[0]):
    #             for j in range(maskMat.shape[1]):
    #                 if self.pnpoly(TYPE.C6_coords, (i, j)):
    #                     maskMat[i][j] = 1
    #     elif phase in ['C4', 'Crect']:
    #         maskMat[34:97, 34:97] = 1
    #     elif phase == 'L':
    #         maskMat[:, 64:97] = 1
        
    #     self.maskMat = maskMat
        
    # def saveMask(self, path="./phin_with_mask.txt"):
    #     self.info.data[self.info.data.shape[1]] = self.maskMat.flatten()
    #     np.savetxt(path, self.info.data.values, fmt="%.6f",
    #                delimiter=" ", header=' '.join(list(map(str, self.info.NxNyNz))), comments="")
    
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
        centers_lxly = np.stack([X[centers[:, 0], centers[:, 1]],
                                Y[centers[:, 0], centers[:, 1]]], axis=1)
        
        self.centers_lxly = centers_lxly
        
        mask = self.in_lim(point=centers_lxly,
                    xlim=(self.lxlylz[1], 2*self.lxlylz[1]),
                    ylim=(self.lxlylz[2], 2*self.lxlylz[2]))
        centers_cut = centers_lxly[mask]
        
        self.vor = Voronoi(centers_lxly, incremental=True)

        plt.figure(figsize=kwargs.get('figsize', (6, 6)))
        ax = plt.gca()
        fig = voronoi_plot_2d(
            self.vor, ax,                 show_points=False,
            show_vertices=False, )
        
        plt.scatter(centers_cut[:, 0], centers_cut[:, 1], s=20, c='k')
        plt.xlim(centers_cut[:, 0].min(), centers_cut[:, 0].max())
        plt.ylim(centers_cut[:, 1].min(), centers_cut[:, 1].max())
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
        
                
class TYPE():
    C6_coords = np.flip(np.array([[33.5, 103.25],
                    [64.,  88.],
                    [93.5, 102.75],
                    [93.5, 153.25],
                    [64., 168.],
                    [33.5, 152.75]]), axis=1)
