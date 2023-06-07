
from decimal import Decimal
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import AutoMinorLocator, MultipleLocator
from scipy.interpolate import griddata
from shapely.geometry import Polygon

class Landscaper():
    
    def __init__(self, path: str, label_dict={
        'ly': r'$L_y/ R_g$',
        'lz': r'$L_x/ R_g$',
        'gamma': r"Lzy/$R_g$"
    }) -> None:
        
        self.path = path
        self.label_dict = label_dict
    
    # @staticmethod
    def read(self):
        df = pd.read_csv(self.path)
        self.df = df
    
    @staticmethod
    def get_w_s(num, Res):
        xs = Decimal(str(num.min())).as_tuple()
        return 10**(-abs(xs[2])-Res)
    
    @staticmethod
    def levels_IQ(contour_set, levels=[0.001, 0.01]):
        # 获取等高面的信息
        for i in range(len(contour_set.collections)):
            # for collection in contour_set.collections:
            level = contour_set.levels[i]
            if level not in levels:
                continue
            collection = contour_set.collections[i]
            path_list = collection.get_paths()
            for path in path_list:
                vertices = path.vertices
                polygon = Polygon(vertices)
                area = polygon.area
                length = polygon.length
                IQ = 4*area*np.pi/length**2
                # centroid = polygon.centroid
                print("level: ", level, "面积：", area, "周长：", length, 'IQ: ', IQ)

    def prospect(self,
                 AxisOne:str='ly', 
                 AxisTwo:str='lz', 
                 Vals:str='freeE',
                 precision:int=3,
                 save:str=None,
                 tick_num=11,
                 asp=1,
                 **kwargs):
        
        df = self.df.copy()
        ly = np.sort(df[AxisOne].unique())
        lz = np.sort(df[AxisTwo].unique())
        min_data = df[df[Vals] == df[Vals].min()]
        
        min_data = min_data.drop_duplicates(subset=['lz','ly'])
        
        for idx in min_data.index:
            print("极小值点: ly: {}, lz:{}, freeE: {}".format(
                min_data.loc[idx, AxisOne], min_data.loc[idx, AxisTwo], min_data.loc[idx, Vals]))
        
        y_all = df[AxisOne].values.reshape(-1, 1)
        x_all = df[AxisTwo].values.reshape(-1, 1)
        yx_all = np.hstack([y_all, x_all])
        step_ly = self.get_w_s(ly, precision)
        step_lz = self.get_w_s(lz, precision)
        grid_ly, grid_lz = np.mgrid[ly.min():ly.max(
        ):step_ly, lz.min():lz.max():step_lz]
        # print(grid_ly.shape)
        grid_freeE = griddata(
            yx_all,
            df[Vals].values,
            (grid_ly, grid_lz),
            method="cubic")

        freeEMat = grid_freeE.copy()
        print(f"free energy mat shape: {grid_freeE.shape}")
        ly=np.unique(grid_ly)
        lz=np.unique(grid_lz)
        
        if kwargs.get('minsub', True):
            freeEMat = freeEMat - freeEMat.min()
        
        if not kwargs.get('levels', False):
            levels = np.linspace(
                np.min(freeEMat), np.max(freeEMat), tick_num)
            ticks = levels
        else:
            levels = kwargs.get('levels', False)
            ticks = levels
            
        plt.figure(figsize=kwargs.get('figsize', (16, 12)))
        cut = kwargs.get('cut', 3)
        contourf_fig = plt.contourf(
            lz, ly, freeEMat, levels=levels, cmap='viridis')
        contour_fig = plt.contour(contourf_fig, colors='w', linewidths=2.5)
        self.levels_IQ(contour_fig)
        
        manual = kwargs.get('manual', False)
        if not manual:
            plt.clabel(contour_fig, fontsize=30, colors=[
                'w']*(len(ticks)-cut)+['k']*cut, fmt='%g')
        else:
            plt.clabel(contour_fig, fontsize=30, colors=[
                'w']*(len(ticks)-cut)+['k']*cut, fmt='%g', manual=manual, zorder=7)

        shrink = kwargs.get('shrink', 1.0)
        clb = plt.colorbar(contourf_fig, ticks=ticks, shrink=shrink, pad=-0.15)
        clb.set_ticklabels(np.around(ticks, kwargs.get('clbacc', 6)), fontsize=35)
        # clb.set_ylabel(r'$\Delta F/k_{\rm{B}}T$', fontsize=20)
        clb.ax.set_title(r'$\Delta F/nk_{B}T$', fontsize=40, pad=18)
        clb.ax.tick_params(which='major', width=2)

        plt.xlabel(self.label_dict[AxisTwo], fontdict={'size': 45})
        plt.ylabel(self.label_dict[AxisOne], fontdict={'size': 45})
        
        ax = plt.gca()
        if asp is not None:
            if asp == 'auto':
                ax.set_aspect(1./ax.get_data_ratio())
            elif asp == 'square' or asp == 'equal':
                plt.axis(asp)
            else:
                ax.set_aspect(asp)

        point_list = kwargs.get('point_list', False)
        if point_list:
            # p = [x, y, marker, c, size]
            for p in point_list:
                plt.scatter(p[0], p[1], s=p[-1], c=p[-2],
                            marker=p[2], alpha=1, zorder=6)
                
        ax.xaxis.set_major_locator(MultipleLocator(kwargs.get('xmajor', 0.2)))
        ax.yaxis.set_major_locator(MultipleLocator(kwargs.get('ymajor', 0.2)))
        ax.xaxis.set_minor_locator(
            AutoMinorLocator(kwargs.get('xminor', 2)))
        ax.yaxis.set_minor_locator(
            AutoMinorLocator(kwargs.get('yminor', 2)))
        
        plt.tight_layout()
        if save:
            plt.savefig(save, dpi=300)
