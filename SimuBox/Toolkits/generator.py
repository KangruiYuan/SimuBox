from collections import Counter, defaultdict
from collections import OrderedDict

import json
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import scipy.optimize as opt
import sympy as sym
from matplotlib.ticker import AutoMinorLocator, MultipleLocator
from sympy import Symbol
from typing import Union, Dict, List, Tuple

__all__ = ['TopoCreater', 'fA', 'fB', 'x', 'Joint']

fA = Symbol('fA', real=True, postive=True)
fB = Symbol('fB', real=True, postive=True)
x = Symbol('x', real=True, postive=True)


class Joint:
    count = 0

    def __init__(self, _id:int, new: bool = False):

        if new: Joint.count = 0
        self.id = _id
        self.count_id = Joint.count
        self.connection = []
        Joint.count += 1

    def __str__(self):
        """
        Returns None
        -------
        >>> str(Joint(_id=0))
        'id=0;count_id=0'
        """
        return f"id={self.id};count_id={self.count_id}"

    def __hash__(self):
        return hash(str(self))

    def __contains__(self, item):
        return item in self.connection

    def add(self, joint) -> None:

        if isinstance(joint, Joint):
            joint = [joint]
        for j in joint:
            if self in j and j in self:
                continue
            if self not in j:
                j.connection.append(self)
            if j not in self:
                self.connection.append(j)

    def search_by_count_id(self, count_id: int):
        _log = set()
        q = []
        q.append(self)
        while q:
            joint = q.pop()
            _id = joint.id
            _count_id = joint.count_id
            if _count_id not in _log:
                _log.add(_count_id)
                if _count_id == count_id:
                    return joint
                else:
                    q.extend(joint.connection)
            else:
                continue
        print(f"Joint with count id = {count_id} do not exist!")

    def stats_connection(self, verbose: bool=True):
        print("format: id(count_id)")
        exp_log = set()
        q = []
        q.append(self)
        res = defaultdict(list)
        while q:
            joint = q.pop()
            _id = joint.id
            _count_id = joint.count_id
            for child in joint.connection:
                child_id = child.id
                child_count_id = child.count_id
                exp = f"{_id}({_count_id})-{child_id}({child_count_id})"
                exp_inv = f"{child_id}({child_count_id})-{_id}({_count_id})"
                if exp not in exp_log or exp_inv not in exp_log:
                    res[tuple(sorted((_id, child_id)))].append((_count_id, child_count_id))
                    exp_log.add(exp)
                    exp_log.add(exp_inv)
                    if verbose:
                        print(exp)
                    q.append(child)
                else:
                    continue
        return res

    def attach_by_id(self, _id, other_id):
        _log = set()
        q = []
        q.append(self)
        while q:
            joint = q.pop()
            tmp_id = joint.id
            _count_id = joint.count_id
            if _count_id not in _log:
                _log.add(_count_id)
                for child in joint.connection:
                    if child.count_id not in _log:
                        q.append(child)
                if _id == tmp_id:
                    tmp_joint = Joint(_id=other_id)
                    joint.add(tmp_joint)
            else:
                continue

    @property
    def joint_num(self):
        return Joint.count


class TopoCreater(nx.DiGraph):

    def __init__(self, verbose=True, simple=True):
        super(TopoCreater, self).__init__()
        self.final_func = None
        self.verbose = verbose
        self.simple = simple

    def init_nodes(self, node_nums: int, node_names: list[str] = None):
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
    def stats(blocks, multiplier=1):
        """
        >>> tp = TopoCreater()
        >>> print(tp.stats({'A':3, 'a':5}))
        Counter({'A': 8})
        """

        if isinstance(blocks, str):
            res = Counter(blocks.upper())
        elif isinstance(blocks, list):
            res = Counter(list(map(lambda x: x.upper(), blocks)))
        elif isinstance(blocks, dict):
            res = defaultdict(int)
            for k, v in blocks.items():
                res[k.upper()] += v
            res = Counter(res)
        else:
            raise ValueError(f"Blocks type {type(blocks)} is not supported!")

        for k, v in res.items():
            res[k] = v * multiplier
        return res

    def add_di_edge(self, pairs: Tuple[int, int], fraction, name, **kwargs):
        self.add_edge(*pairs, fraction=fraction, name=name, **kwargs)
        self.add_edge(*pairs[::-1], fraction=fraction, name=name, **kwargs)

    def linear(self,
               blocks: Union[List[str], str, Dict],
               fractions=None,
               method='auto',
               **kwargs):
        if fractions is None:
            fractions = []
        self.clear()

        self.type = 'linear'
        blocks_info = self.stats(blocks)
        if isinstance(blocks, dict): blocks = blocks_info.elements()

        self.init_nodes(sum(blocks_info.values()) + 1)
        if method == 'auto' and len(fractions) == 0:
            fraction = np.around(1 / sum(blocks_info.values()), 3)
            for i, b in enumerate(blocks):
                self.add_di_edge((i, i + 1), fraction=fraction, name='_'.join([b, str(i + 1)]), kind=b)
        elif method == 'manual' or len(fractions) != 0:
            if len(fractions) == 0 or len(fractions) != sum(blocks_info.values()):
                raise ValueError(f"Fractions for blocks (length={len(fractions)}) is not right!")
            if round(sum(fractions), 3) != 1:
                raise ValueError(f"Fractions do not sum to one, value={sum(fractions)}")
            for i, (b, f) in enumerate(zip(blocks, fractions)):
                self.add_di_edge((i, i + 1), fraction=f, name='_'.join([b, str(i + 1)]), kind=b)
        else:
            raise NotImplementedError(method)
        self.get_info()

    def AmBn(self,
             blocks: Union[str, List[str], Dict],
             fractions=None,
             method='auto',
             **kwargs):

        if fractions is None:
            fractions = []
        self.clear()
        self.type = 'AmBn'
        blocks_info = self.stats(blocks)
        if isinstance(blocks, dict): blocks = blocks_info.elements()

        self.init_nodes(sum(blocks_info.values()) + 1)
        if method == 'auto' and len(fractions) == 0:
            fraction = np.around(1 / sum(blocks_info.values()), 3)
            for i, b in enumerate(blocks):
                self.add_di_edge((0, i + 1), fraction=fraction, name='_'.join([b, str(i + 1)]), kind=b)
        elif method == 'manual' or len(fractions) != 0:
            if len(fractions) == 0 or len(fractions) != sum(blocks_info.values()):
                raise ValueError(f"Fractions for blocks (length={len(fractions)}) is not right!")
            if round(sum(fractions), 3) != 1:
                raise ValueError(f"Fractions do not sum to one, value={sum(fractions)}")
            for i, (b, f) in enumerate(zip(blocks, fractions)):
                self.add_di_edge((0, i + 1), fraction=f, name='_'.join([b, str(i + 1)]), kind=b, minlen=f)
        else:
            raise NotImplementedError(method)
        self.get_info()

    def star(self,
             blocks: Union[str, List[str]],
             fractions=None,
             arm: int = 5,
             head: bool = True,
             method='auto',
             **kwargs):

        if fractions is None:
            fractions = []
        self.clear()
        self.type = 'star'
        blocks_info = self.stats(blocks, arm)
        if not head: blocks = blocks[::-1]

        self.init_nodes(sum(blocks_info.values()) + 1)
        if method == 'auto' and len(fractions) == 0:
            fraction = np.around(1 / sum(blocks_info.values()), 3)
            for a in range(arm):
                for i, b in enumerate(blocks):
                    idx_head = a * len(blocks) + i if i != 0 else 0
                    idx_tail = a * len(blocks) + i + 1
                    self.add_di_edge((idx_head, idx_tail), fraction=fraction, name='_'.join([b, str(idx_tail)]), kind=b)

        elif method == 'manual' or len(fractions) != 0:
            if len(fractions) * 5 != sum(blocks_info.values()):
                raise ValueError(
                    f"Fractions for blocks (length={len(fractions)}*{arm}={len(fractions) * arm}) is not right!")
            if round(sum(fractions) * arm, 3) != 1:
                raise ValueError(f"Fractions do not sum to one, value={round(sum(fractions) * arm, 3)}")
            for a in range(arm):
                for i, (b, f) in enumerate(zip(blocks, fractions)):
                    idx_head = a * len(blocks) + i if i != 0 else 0
                    idx_tail = a * len(blocks) + i + 1
                    self.add_di_edge((idx_head, idx_tail), fraction=f, name='_'.join([b, str(idx_tail)]), kind=b,
                                     minlen=f)
        else:
            raise NotImplementedError(method)
        self.get_info()

    @classmethod
    def parseJson(cls, path: str) -> np.ndarray:
        with open(path, mode='r') as fp:
            data = json.load(fp, object_pairs_hook=OrderedDict)
        topo_mat = []
        for block in data['Block']:
            id1 = block['LeftVertexID']
            id2 = block['RightVertexID']
            mul = block['Multiplicity']
            # if mul > 1:
            #     raise NotImplementedError(f"Mehthod for multiplicity larger thar {mul} is not supported.")
            if block['BranchDirection'] == 'LEFT_BRANCH':
                direction = 0
                id_branch = id1
            elif block['BranchDirection'] == 'RIGHT_BRANCH':
                direction = 1
                id_branch = id2
            else:
                raise ValueError
            id1, id2 = (id1, id2) if id1 < id2 else (id2, id1)
            fraction = block['ContourLength']
            kind = block['ComponentName']
            topo_mat.append([id1, id2, mul, direction, id_branch, fraction, kind])
        topo_mat = np.array(topo_mat, dtype=object)
        return topo_mat

    def fromJson(self, path: str, **kwargs):
        topo_mat = self.parseJson(path)
        edges_info = defaultdict(dict)
        count = 1
        for i, t in enumerate(topo_mat):
            core_id = t[t[3]]
            branch_id = t[1 - t[3]]
            branch_mul = t[2]
            edges_info[tuple(sorted((core_id, branch_id)))] = {'fraction': t[-2], 'kind': t[-1],
                                                               'name': '_'.join([t[-1], str(count)])}
            if i == 0:
                joint = Joint(_id=core_id, new=True)
            for j in range(branch_mul):
                joint.attach_by_id(_id=core_id, other_id=branch_id)
            count += 1
        connections = joint.stats_connection(verbose=kwargs.get('verbose', True))
        node_num = joint.joint_num
        self.init_nodes(node_num)

        for k, v in connections.items():
            for e in v:
                self.add_di_edge(e, **edges_info[k])
        self.get_info()

    @staticmethod
    def count_nums(layers, branch):
        return sum([branch ** i for i in range(layers)])

    def dendrimer(self,
                  Ablock_layer: int = 1,
                  Bblock_layer: int = 1,
                  A_branch: int = 2,
                  B_branch: int = 2,
                  fractions=None,
                  method='auto',
                  **kwargs):

        if fractions is None:
            fractions = []
        self.clear()
        self.type = 'dendrimer'
        Ablock_num = self.count_nums(Ablock_layer, A_branch)
        Ablock_out_num = Ablock_num - \
                         self.count_nums(Ablock_layer - 1, A_branch)
        Bblock_num = self.count_nums(
            Bblock_layer, B_branch) * B_branch * Ablock_out_num
        total_blocks = Ablock_num + Bblock_num
        total_nodes = total_blocks + 1
        self.init_nodes(total_nodes)

        if method == 'auto' and len(fractions) == 0:
            fraction = np.around(1 / total_nodes, 3)
            count = 0
            for i in range(Ablock_layer):
                if i == 0:
                    self.add_di_edge((0, 1), fraction=fraction, name='_'.join(['A', str(count + 1)]), kind='A')
                    count += 1
                    last_layer_nodes = [count]
                else:
                    this_layer_nodes = np.arange(
                        count + 1, count + A_branch ** i + 1, 1)
                    for j in last_layer_nodes:
                        tmp = 1
                        while (tmp <= A_branch):
                            count += 1
                            self.add_di_edge((j, count), fraction=fraction, name='_'.join(['A', str(count + 1)]),
                                             kind='A')
                            tmp += 1
                        last_layer_nodes = this_layer_nodes
            for i in range(Ablock_layer, Ablock_layer + Bblock_layer, 1):
                beginning_node_num = len(last_layer_nodes)
                this_layer_nodes = np.arange(
                    count + 1,
                    count + 1 + beginning_node_num * B_branch,
                    1
                )
                for j in last_layer_nodes:
                    tmp = 1
                    while (tmp <= B_branch):
                        count += 1
                        self.add_di_edge((j, count), fraction=fraction, name='_'.join(['B', str(count + 1)]), kind='B')
                        tmp += 1
                last_layer_nodes = this_layer_nodes
        else:
            raise NotImplementedError(method)
        self.get_info()

    def get_info(self):

        self.edge_info = self.edges(data=True)
        self.edge_kinds = {}
        for ix, iy, dic in self.edge_info:
            kind = dic['kind']
            self.edge_kinds[kind] = self.edge_kinds.get(kind, []) + [(ix, iy)]

    def show_topo(
            self,
            colorlist=None,
            node_size=200,
            node_color='gray',
            pos=None,
            figsize=(10, 10),
            save_path=None, **kwargs):

        if colorlist is None:
            colorlist = ['r', 'b', 'g', 'blueviolet', 'cyan']
        fig, ax = plt.subplots(figsize=figsize)
        if pos is None:
            pos = nx.kamada_kawai_layout(self)

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
        ax.legend(loc=kwargs.get('loc', 'best'), frameon=False)
        plt.axis('on')
        plt.show()
        if save_path:
            plt.savefig(save_path)

    @classmethod
    def h(cls, f, x):
        return 2 / (x ** 2) * (f * x + sym.exp(-f * x) - 1)

    @classmethod
    def S(cls, f1, fdis, f2, x):
        result = 1 / 2 * (cls.h(f1 + fdis + f2, x) -
                          cls.h(f1 + fdis, x) - cls.h(f2 + fdis, x) + cls.h(fdis, x))
        return result

    @classmethod
    def Sself(cls, f, x):
        return 1 / 2 * cls.h(f, x)

    def nodes_dis(self, start, end):
        path = nx.dijkstra_path(self, start, end)
        return sum([self.get_edge_data(path[i], path[i + 1])['fraction'] for i in range(len(path) - 1)])

    def RPA(self, latex=False):
        global x
        edgeNum = self.number_of_edges()
        edgeNum_half = int(edgeNum / 2)

        edges = []
        edge_fraction = []
        edge_kinds = []

        for i, j, dic in self.edge_info:
            if i < j:
                edges.append((i, j))
                edge_fraction.append(dic['fraction'])
                edge_kinds.append(dic['kind'])

        dismat = np.zeros((edgeNum_half, edgeNum_half), dtype=object)
        for i in range(edgeNum_half):
            for j in range(edgeNum_half):
                if i == j:
                    dismat[i][j] = 0
                    continue

                tmppath = nx.dijkstra_path(
                    self, edges[i][0], edges[j][1])
                if edges[i][1] in tmppath and edges[j][0] in tmppath:
                    tmpDis = self.nodes_dis(
                        edges[i][1], edges[j][0])
                elif edges[i][1] in tmppath and edges[j][0] not in tmppath:
                    tmpDis = self.nodes_dis(
                        edges[i][1], edges[j][1])
                elif edges[i][1] not in tmppath and edges[j][0] in tmppath:
                    tmpDis = self.nodes_dis(
                        edges[i][0], edges[j][0])
                else:
                    tmpDis = self.nodes_dis(
                        edges[i][0], edges[j][1])
                dismat[i][j] = tmpDis

        Slist = np.zeros((edgeNum_half, edgeNum_half), dtype=object)
        for i in range(edgeNum_half):
            for j in range(i + 1):
                if j == i:
                    Slist[i][j] = self.Sself(edge_fraction[i], x)
                else:
                    Slist[i][j] = self.S(
                        edge_fraction[i],
                        dismat[i][j],
                        edge_fraction[j],
                        x)
        self.SAA = 0
        self.SBB = 0
        self.SAB = 0

        for i in range(edgeNum_half):
            for j in range(i + 1):
                if edge_kinds[i] == edge_kinds[j] == 'A':
                    self.SAA += 2 * Slist[i][j]
                elif edge_kinds[i] == edge_kinds[j] == 'B':
                    self.SBB += 2 * Slist[i][j]
                else:
                    self.SAB += Slist[i][j]

        self.final_func = 1 / 2 * (self.SAA + self.SBB + 2 * self.SAB) / \
                          (self.SAA * self.SBB - self.SAB * self.SAB)

        if latex is True:
            sym.print_latex(self.final_func)

    def ODT(self,
            fAs: Union[List[float], np.ndarray] = np.arange(0.1, 1.0, 0.1),
            paint: bool = True,
            symbol=fA,
            **kwargs
            ):
        global x
        xN = []
        _lim = kwargs.get('lim', (0, 100))
        for i in fAs:
            func = self.final_func.evalf(subs={symbol: i})
            func = sym.lambdify(x, func, 'scipy')
            xN.append(opt.fminbound(func, *_lim, full_output=True)[1])

        if len(np.unique(xN)) == 1:
            print(f'ODT is {xN[0]}')
            return

        if paint:
            fig, ax = plt.subplots(figsize=kwargs.get('figsize', (8, 6)))
            ax.plot(fAs, xN, 'r-', lw=2)
            ax.set_xlim([0, 1])
            ax.set_ylim(_lim)
            if xminor := kwargs.get('xminor', 5):
                ax.xaxis.set_minor_locator(AutoMinorLocator(xminor))
            if yminor := kwargs.get('yminor', 5):
                ax.yaxis.set_minor_locator(AutoMinorLocator(yminor))
            if xmain := kwargs.get('xmain', 0.1):
                ax.yaxis.set_major_locator(MultipleLocator(xmain))
            if ymain := kwargs.get('ymain', 10):
                ax.yaxis.set_major_locator(MultipleLocator(ymain))
            ax.set_xlabel(kwargs.get('xlabel', r"$f_{A}$"))
            ax.set_ylabel(kwargs.get('ylabel', r"$\chi N$"))
            plt.show()


class PhiCreater():

    def __int__(self, NxNyNz, lxlylz, **kwargs):
        self.NxNyNz = NxNyNz
        self.lxlylz = lxlylz

    def generate(self, template):
        pass


if __name__ == "__main__":
    import doctest

    doctest.testmod()
