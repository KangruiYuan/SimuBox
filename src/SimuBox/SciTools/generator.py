
import sympy as sym
from sympy import Symbol
import scipy.optimize as opt
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib as mpl
from typing import Union, Dict, List, Tuple
from collections import Counter

__all__ = ['TopoCreater']

fA = Symbol('fA', real=True, postive=True)
fB = Symbol('fA', real=True, postive=True)
x = Symbol('x', real=True, postive=True)

class TopoCreater(nx.DiGraph):

    def __init__(self, verbose=True, simple=True):
        super(TopoCreater, self).__init__()
        self.verbose = verbose
        self.simple = simple

    def init_nodes(self, node_nums:int, node_names:list[str]=None):
        self.node_nums = node_nums
        if node_names:
            self.node_names = node_names
            for i in range(node_nums):
                self.add_node(i, name=node_names[i])
        else:
            for i in range(node_nums):
                self.add_node(i, name=str(i))

    def print(self, s):
        if self.verbose:
            print(s)

    @staticmethod
    def stats(blocks):
        if isinstance(blocks, str):
            return Counter(blocks.upper())
        elif isinstance(blocks, list):
            return Counter(list(map(lambda x:x.upper(), blocks)))
        else:
            raise ValueError(f"Blocks type {type(blocks)} is not supported!")

    def add_di_edge(self, pairs:Tuple[int,int], fraction, name, **kwargs):
        self.add_edge(*pairs, fraction=fraction, name=name, **kwargs)
        self.add_edge(*pairs[::-1], fraction=fraction, name=name, **kwargs)
    def linear(self,
               blocks:Union[List[str], str],
               fractions:List[float]=[],
               method='auto',
               **kwargs):
        self.clear()

        self.type = 'linear'
        blocks_info = self.stats(blocks)
        self.init_nodes(sum(blocks_info.values())+1)
        if method == 'auto' and len(fractions) == 0:
            fraction = np.around(1/sum(blocks_info.values()), 3)
            for i, b in enumerate(blocks):
                self.add_di_edge((i,i+1), fraction=fraction, name='_'.join([b, str(i+1)]), kind=b)
        elif method == 'manual' or len(fractions) != 0:
            if len(fractions) == 0 or len(fractions) != sum(blocks_info.values()):
                raise ValueError(f"Fractions for blocks (length={len(fractions)}) is not right!")
            if sum(fractions) != 1:
                raise ValueError(f"Fractions do not sum to one, value={sum(fractions)}")
            for i, (b, f) in enumerate(zip(blocks, fractions)):
                self.add_di_edge((i,i+1), fraction=f, name='_'.join([b, str(i+1)]), kind=b)
        else:
            raise NotImplementedError(method)
        self.get_info()

    def get_info(self):

        self.edge_info = self.edges(data=True)
        self.edge_kinds = {}
        for ix, iy, dic in self.edge_info:
            kind = dic['kind']
            self.edge_kinds[kind] = self.edge_kinds.get(kind, []) + [(ix,iy)]

    def show_topo(
            self,
            colorlist=[ 'r','b','g','blueviolet','cyan'],
            node_size=200,
            node_color='gray',
            pos=None,
            figsize=(10,10),
            save_path=None):

        fig, ax = plt.subplots(figsize=figsize)
        if pos is None:
            pos = nx.spring_layout(self)

        nx.draw_networkx_nodes(
            self, pos,
            node_size=node_size,
            node_color=node_color,
            ax=ax)

        for i, (_kind, _edge) in enumerate(self.edge_kinds.items()):
            nx.draw_networkx_edges(
                self,
                pos,
                edgelist=_edge,
                width=3,
                alpha=1,
                edge_color=colorlist[i],
                label=_kind,
                arrows=False,
                ax=ax)

        nx.draw_networkx_edge_labels(self, pos, edge_labels=nx.get_edge_attributes(self, 'fraction'), ax=ax)
        nx.draw_networkx_labels(
            self,
            pos,
            nx.get_node_attributes(self, 'name'),
            font_size=12,
            font_family='sans-serif',
            ax=ax)
        ax.legend()
        plt.axis('on')
        plt.show()
        if save_path:
            plt.savefig(save_path)


class PhiCreater():

    def __int__(self, NxNyNz, lxlylz, **kwargs):

        self.NxNyNz = NxNyNz
        self.lxlylz = lxlylz

    def generate(self, template):
        pass