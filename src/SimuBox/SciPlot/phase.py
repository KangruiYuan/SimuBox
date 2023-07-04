import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator, MultipleLocator
import pandas as pd
from numpy.polynomial import Chebyshev
from scipy.io import loadmat


class PhaseDiagram():

    def __init__(self, xlabel, ylabel,
                 color_dict=None,
                 label_dict=None,
                 **kwargs
                 ) -> None:

        if label_dict is None:
            label_dict = {
                'tau': r'$\tau$',
                'ksi': r'$\xi$',
                'volH': r'$\phi_{\rm{H}}$',
                'fA': r'$f_{\rm{A}}$',
                'chiN': r'$\chi \rm{N}$'
            }
        if color_dict is None:
            color_dict = {
                'C4': 'r',
                'Crect': 'dodgerblue',
                'L': 'deeppink',
                'DG': 'orange',
                'iHPa': 'blue',
                'SG': 'magenta',
                'SC': 'goldenrod',
                'C6': 'tan',
                'C3': 'm',
                'HCP': 'crimson',
                'FCC': 'yellowgreen',
                'PL': 'darkorchid',
                'BCC': 'limegreen',
                'sdgn': 'teal',
                'O70': 'crimson',
                'unknown': 'k',
                'Disorder': 'y'
            }
        self.color_dict = color_dict
        self.color_dict.update(kwargs.get('color', {}))
        self.label_dict = label_dict
        self.label_dict.update(kwargs.get('label', {}))
        self.xlabel = xlabel
        self.ylabel = ylabel
        pass

    def query_point(self, df, xval=None, yval=None, **kwargs):
        res = df.copy()
        if xval:
            res = res[res[self.xlabel] == xval]
        if yval:
            res = res[res[self.ylabel] == yval]
        query_dict = kwargs.get('query_dict', {})
        for key, val in query_dict.items():
            res = res[res[key] == val]
        return res

    def query(self, df: pd.DataFrame, query_dict=None, **kwargs):
        if query_dict is None:
            query_dict = {}
        if len(query_dict) == 0: return df
        pass

    @staticmethod
    def checkin(phase, candidates):
        if len(phase) == 0: return False
        for i in phase:
            if i in candidates:
                return True
        return False

    def data(self, path, **kwargs):
        dropset = kwargs.get('dropset', ['lx', 'ly', 'lz', 'phase', 'freeE'])
        div = kwargs.get('div', 1)
        div_var = kwargs.get('div_var', 'chiN')
        acc = kwargs.get('acc', 3)

        df = pd.read_csv(path)
        if dropset:
            df = df.drop_duplicates(subset=dropset)
        df['lylz'] = np.around(df['ly'] / df['lz'], acc)
        df['lxly'] = np.around(df['lx'] / df['ly'], acc)
        try:
            df[div_var] = df[div_var] / div
        except KeyError:
            pass
        return df

    def compare(self,
                path: str,
                plot: bool = True,
                acc: int = 3,
                **kwargs):

        df = self.data(path=path, acc=acc)
        print(f"Include phase: {set(df['phase'].values)}")

        plot_dict = dict()
        y_set = np.sort(df[self.ylabel].unique())
        x_set = np.sort(df[self.xlabel].unique())

        exclude = kwargs.get('exclude', [])

        mat = np.zeros((len(y_set), len(x_set)), dtype=object)
        for i, y in enumerate(y_set):
            for j, x in enumerate(x_set):
                phase = 'unknown'
                temp_data = df[(df[self.ylabel] == y) & (df[self.xlabel] == x)]
                min_data = temp_data[temp_data['freeE'] == temp_data['freeE'].min()]
                if len(min_data) == 0:
                    continue
                freeE = min_data['freeE'].min()
                min_label = min_data['phase'].unique()
                lxly = min_data['lxly'].unique()
                lylx = np.around(1 / lxly, acc)
                lylz = min_data['lylz'].unique()
                lzly = np.around(1 / lylz, acc)

                if self.checkin(exclude, min_label):
                    phase = min_label[0]
                elif 'L' in min_label:
                    phase = 'L'
                elif self.checkin(['C4', 'Crect'], min_label):
                    if any(lylz == 1):
                        phase = 'C4'
                    else:
                        phase = 'Crect'
                elif len(min_label) == 1:
                    if self.checkin(['C6', 'C3'], min_label):
                        if lylz == np.around(np.sqrt(3), acc) or lzly == np.around(1 / np.sqrt(3), acc):
                            phase = min_label[0]
                    elif self.checkin(['iHPa'], min_label):
                        if lxly == np.around(np.sqrt(3), acc) or lylx == np.around(1 / np.sqrt(3), acc):
                            phase = 'iHPa'
                        elif lylz == np.around(np.sqrt(2), acc) or lxly == 1:
                            phase = 'SC'
                    elif 'PL' in min_label:
                        if lxly == np.around(np.sqrt(3), acc) or lylx == np.around(1 / np.sqrt(3), acc):
                            phase = 'PL'
                    elif self.checkin(['L', 'Disorder', 'O70'], min_label):
                        phase = min_label[0]
                    elif self.checkin(['SC', 'SG', 'DG', 'BCC', 'FCC', 'sdgn'], min_label):
                        if lxly == 1 and lylz == 1:
                            phase = min_label[0]
                    else:
                        phase = '_'.join([phase, min_label[0]])
                mat[i][j] = [phase, x, y, freeE]
                if phase in plot_dict:
                    for attr, val in zip([self.xlabel, self.ylabel, 'freeE', 'lylz', 'lxly'],
                                         [x, y, freeE, lylz, lxly]):
                        plot_dict[phase][attr].append(val)
                else:
                    plot_dict[phase] = {self.xlabel: [x], self.ylabel: [y], 'freeE': [
                        freeE], 'lylz': [lylz], 'lxly': [lxly]}

        if plot:
            fig, ax = plt.subplots(1, 1, figsize=(8, 6))
            for key, value in plot_dict.items():
                ax.scatter(
                    value[self.xlabel],
                    value[self.ylabel],
                    c=self.color_dict[key],
                    label=key)
            ax.tick_params(top='on', right='on', which='both')
            ax.tick_params(which='both', width=2, length=4, direction='in')
            ax.xaxis.set_minor_locator(AutoMinorLocator(5))
            ax.yaxis.set_minor_locator(AutoMinorLocator(5))
            ax.set_xlabel(self.label_dict.get(self.xlabel, self.xlabel),
                          fontdict={'size': 20})
            ax.set_ylabel(self.label_dict.get(self.ylabel, self.ylabel),
                          fontdict={'size': 20})
            ax.legend(frameon=False, loc='upper left',
                      bbox_to_anchor=(0.98, 0.8))
            fig.tight_layout()
            return df, plot_dict, mat, fig, ax
        else:
            return df, plot_dict, mat

    @staticmethod
    def cross_point(x1, y1, x2, y2, x3, y3, x4, y4):
        b1 = (y2 - y1) * x1 + (x1 - x2) * y1
        b2 = (y4 - y3) * x3 + (x3 - x4) * y3
        D = (x2 - x1) * (y4 - y3) - (x4 - x3) * (y2 - y1)
        D1 = b2 * (x2 - x1) - b1 * (x4 - x3)
        D2 = b2 * (y2 - y1) - b1 * (y4 - y3)
        return (D1 / D, D2 / D)

    @staticmethod
    def de_unknown(mat):
        for i in range(mat.shape[0]):
            for j in range(mat.shape[1]):
                try:
                    mat[i][j][0] = mat[i][j][0].lstrip('unknown_')
                except:
                    continue
        return mat

    def scan(self,
             folder,
             ann_dict,
             **kwargs):
        filelist = os.listdir(folder)
        fig, ax = plt.subplots(figsize=kwargs.get('figsize', (8, 6)))
        skip = kwargs.get('skip', [])
        extract = kwargs.get('extract', [])
        inverse_lst = kwargs.get('inverse', [])
        deunknown = kwargs.get('deunknown', [])

        for filename in filelist:
            if not filename.endswith('.csv'):
                continue
            tmp_name = os.path.splitext(filename)[0]
            tmp_dict = ann_dict.get(tmp_name, {})
            print(tmp_name, end='\t')
            if tmp_name in skip:
                print('Skip...')
                continue
            else:
                tmp_data, _, tmp_mat = self.compare(
                    path=os.path.join(folder, filename), plot=False, acc=tmp_dict.get('acc', 2))

            if tmp_name in deunknown or deunknown == 'All':
                tmp_mat = self.de_unknown(tmp_mat)

            if tmp_name in extract:
                tmp_xys = self.extract_edge(
                    tmp_mat,
                    which=tmp_dict.get('which', 'C4'),
                    mode=tmp_dict.get('mode', 'middle'),
                    factor=tmp_dict.get('factor', 0.5))
            else:
                tmp_xys = self.boundary(tmp_data, tmp_mat, mode=['hori', 'ver'])

            if tmp_name in inverse_lst:
                tmp_xys = tmp_xys[np.argsort(tmp_xys[:, 1])]
                self.draw_line(
                    tmp_xys, ax=ax, tmp_ann_dict=tmp_dict, inverse=True)
            else:
                tmp_xys = tmp_xys[np.argsort(tmp_xys[:, 0])]
                self.draw_line(tmp_xys, ax=ax, tmp_ann_dict=tmp_dict)

        if phase_name := ann_dict.get('phase_name', False):
            for key, value in phase_name.items():
                ax.text(value[0], value[1], key, fontsize=20, color="k")

        phase_name_arrow = ann_dict.get('phase_name_with_arrow', False)
        if phase_name_arrow:
            for key, value in phase_name_arrow.items():
                ax.annotate(key, xy=value[:2],
                            xycoords="data",
                            xytext=value[2:4], textcoords="data",
                            weight="bold",
                            color="k", fontsize=20,
                            arrowprops=dict(arrowstyle="->",
                                            connectionstyle="arc3",
                                            color='k',
                                            lw=2))

        save_path = kwargs.get('path', '')
        if mat_path := kwargs.get('mat_path', None):
            wrongline = loadmat(mat_path)['origin_wrong']
            self.draw_line(wrongline, ax=ax,
                           tmp_ann_dict=kwargs.get('mat', {
                               'adde': 0.001, 'ls': ':', 'shrinks': 7, 'alpha': 0.5}))

        plt.margins(0)
        ax.xaxis.set_minor_locator(AutoMinorLocator(5))
        ax.yaxis.set_minor_locator(AutoMinorLocator(5))
        ax.yaxis.set_major_locator(MultipleLocator(kwargs.get('ymajor', 0.1)))
        ax.set_xlabel(self.label_dict[self.xlabel], fontdict={'size': 30})
        ax.set_ylabel(self.label_dict[self.ylabel], fontdict={'size': 30})
        plt.tick_params(labelsize=20, pad=8)
        fig.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=200)

    def boundary(self, df, mat, mode=None):
        if mode is None:
            mode = ['ver', 'hori']
        xys = []
        for i in range(mat.shape[0]):
            for j in range(mat.shape[1]):
                point1 = mat[i][j]
                for k, w in zip([-1, 0, 1, 0], [0, 1, 0, -1]):
                    ii = i + k
                    jj = j + w
                    if (i == 0 and ii == -1) or (j == 0 and jj == -1):
                        continue
                    try:
                        point3 = mat[ii][jj]
                        if point1[0] == point3[0] or 'unknown' in point1[0] or 'unknown' in point3[0]:
                            continue
                        point2 = df[(df.phase == point1[0]) & (
                                df[self.xlabel] == point3[1]) & (df[self.ylabel] == point3[2])]
                        if len(point2) == 0:
                            continue
                        point4 = df[(df.phase == point3[0]) & (
                                df[self.xlabel] == point1[1]) & (df[self.ylabel] == point1[2])]
                        if len(point4) == 0:
                            continue

                        if 'hori' in mode:
                            if i == ii:
                                x0, _ = self.cross_point(point1[1], point1[3], point2[self.xlabel].values[0],
                                                         point2.freeE.values[0],
                                                         point3[1], point3[3], point4[self.xlabel].values[0],
                                                         point4.freeE.values[0])
                                if [x0, point1[2]] not in xys:
                                    xys.append([x0, point1[2]])
                        if 'ver' in mode:
                            if j == jj:
                                x0, _ = self.cross_point(point1[2], point1[3], point2[self.ylabel].values[0],
                                                         point2.freeE.values[0],
                                                         point3[2], point3[3], point4[self.ylabel].values[0],
                                                         point4.freeE.values[0])
                                if [point1[1], x0] not in xys:
                                    xys.append([point1[1], x0])
                    except BaseException:
                        continue
        return np.array(xys)

    @staticmethod
    def draw_line(xys, ax=None, tmp_ann_dict=None, inverse=False):

        if tmp_ann_dict is None:
            tmp_ann_dict = {}
        if cuts := tmp_ann_dict.get('cuts', None):
            xys = xys[cuts:]
        if cute := tmp_ann_dict.get('cute', None):
            xys = xys[:cute]

        if inverse:
            xs = xys[:, 1].copy()
            ys = xys[:, 0].copy()
        else:
            xs = xys[:, 0].copy()
            ys = xys[:, 1].copy()

        try:
            coefs = Chebyshev.fit(xs, ys, tmp_ann_dict.get('order', 3))
        except BaseException:
            print(xs, ys)
            return

        new_x = np.linspace(
            xs.min() - tmp_ann_dict.get('adds', 0),
            xs.max() + tmp_ann_dict.get('adde', 0),
            300)
        new_y = coefs(new_x)

        if shrinks := tmp_ann_dict.get('shrinks', None):
            new_x = new_x[shrinks:]
            new_y = new_y[shrinks:]
        if shrinke := tmp_ann_dict.get('shrinke', None):
            new_x = new_x[:shrinke]
            new_y = new_y[:shrinke]

        if inverse:
            new_x, new_y = new_y, new_x

        if ax:
            ax.plot(new_x, new_y, c='k', lw=3,
                    ls=tmp_ann_dict.get('ls', '-'),
                    alpha=tmp_ann_dict.get('alpha', 1))
        else:
            plt.plot(new_x, new_y, c='k', lw=3)

    @staticmethod
    def extract_edge(mat, which='Disorder', mode='middle', factor=0.5):
        edge_data = []
        for co in range(mat.shape[1]):
            tmp_co = mat[:, co]
            tmp_co = tmp_co[tmp_co != 0]
            for ro in range(1, len(tmp_co)):
                try:
                    if tmp_co[ro][0] != which and tmp_co[ro - 1][0] == which:
                        if mode == 'middle':
                            edge_data.append(
                                [
                                    tmp_co[ro][1] * factor + tmp_co[ro - 1][1] * (1 - factor),
                                    tmp_co[ro][2] * factor + tmp_co[ro - 1][2] * (1 - factor)
                                ])
                        elif mode == 'down':
                            edge_data.append([tmp_co[ro - 1][1],
                                              tmp_co[ro - 1][2]])
                        elif mode == 'up':
                            edge_data.append([tmp_co[ro][1],
                                              tmp_co[ro][2]])
                        break
                    else:
                        continue
                except BaseException:
                    continue
        edge_data = np.array(edge_data)
        return edge_data
