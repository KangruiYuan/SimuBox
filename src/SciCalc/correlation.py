
from ..SciTools import InfoReader
import numpy as np
from tqdm import tqdm
from collections import defaultdict
import matplotlib.pylab as plt

__author__ = ['Alkaid Yuan']

class Corr(InfoReader):
    
    def __init__(self, path:str, name_flag:bool=False, inverse_flag:bool=False):
        
        super().__init__(path, inverse_flag=inverse_flag, name_flag=name_flag)
    
    def correlation2D(self, 
                      mat_A, 
                      mat_B, 
                      bins:int = 20, 
                      plot:bool=True,
                      **kwargs
                      ):
        """
        2D version
        """
        
        Lx = self.lxlylz[1]
        Ly = self.lxlylz[2]

        Nx = self.NxNyNz[1]
        Ny = self.NxNyNz[2]
        
        x_max = Lx/2
        y_max = Ly/2
        dx = Lx/(Nx -1)
        dy = Ly/(Ny -1)
        max_dis = min([np.sqrt(x_max**2 + y_max**2), x_max, y_max])
        
        plong_x = plong_y = 0
        
        x = np.linspace(-plong_x*dx, Lx+plong_x*dx, Nx+2*plong_x)
        y = np.linspace(-plong_y*dy, Ly+plong_y*dy, Ny+2*plong_y)
        
        xcoords, ycoords = np.meshgrid(x - x_max, y - y_max)
        distance_mat = np.sqrt(xcoords**2 + ycoords**2)
        
        size = mat_A.shape
        cut_size = (size[1]-plong_x, 2*size[1]+plong_x, size[0]-plong_y, 2*size[0]+plong_y)
        
        A_tile = np.tile(mat_A, (3,3))
        A_real = A_tile[cut_size[2]:cut_size[3], cut_size[0]:cut_size[1]]
        B_tile = np.tile(mat_B, (3,3))
        B_real = B_tile[cut_size[2]:cut_size[3], cut_size[0]:cut_size[1]]
        
        assert B_real.shape == distance_mat.shape
       
        real_sum = A_real + B_real
        if kwargs.get('fill', False):
            real_sum[real_sum == 0] = np.min(real_sum)
        psi = (A_real - B_real)/(real_sum)

        Dx, Dy = psi.shape
        mean_psi = np.mean(psi)
        square_mean_psi = mean_psi**2
        
        # Non-uniform sampling point
        bin_edges = np.append(np.linspace(0, max_dis/4, 10),
                              np.linspace(max_dis*(1/4+1/12), max_dis, 10))
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        
        bin_list = []
        bin_sum_list = []
        bin_centers_true = []
        for i in range(len(bin_centers)):
            r_min = bin_edges[i]
            r_max = bin_edges[i+1]
            in_bin = (distance_mat >= r_min) & (distance_mat < r_max)
            tmp_sum = np.sum(in_bin)
            if tmp_sum > 0:
                bin_sum_list.append(tmp_sum)
                bin_list.append(in_bin)
                bin_centers_true.append(bin_centers[i])
            
        correlation = defaultdict(int)
        for i in tqdm(range(len(bin_centers))):
            correlation_sum = 0
            tmp_bin = bin_list[i]
            for xroll in range(Dx):
                for yroll in range(Dy):
                    psi_roll = np.roll(psi, -xroll, axis=1)
                    psi_roll = np.roll(psi_roll, -yroll, axis=0)
                    psi_00 = psi_roll[0, 0]
                    psi_shift = np.fft.fftshift(psi_roll)
                    correlation_sum += np.sum(psi_shift[tmp_bin] * psi_00)
            correlation_num = Dx * Dy * bin_sum_list[i]
            correlation[i] = correlation_sum/correlation_num - square_mean_psi
        
        correlation = sorted(correlation.items(), key=lambda x:x[1], reverse=True)
        correlation = [v[1] for v in correlation]
        correlation = [np.mean(psi**2) - square_mean_psi] + correlation
        bin_centers_true = [0] + bin_centers_true
        
        self.bin_centers = np.array(bin_centers_true)
        self.correlation = np.array(correlation)
        
    def plotCorr(self, ax=None, **kwargs):
        if not ax:
            plt.figure()
            ax = ax if ax else plt.gca()
        ax.plot(self.bin_centers, self.correlation, ls='-', marker='o', lw=2, label=kwargs.get('label', 'None'))
        ax.set_xlabel(r"$\bf r$", fontsize=20)
        ax.set_ylabel(r'C($\bf r$)', fontsize=20)
        return ax
            
            
    
    
    
        
        
