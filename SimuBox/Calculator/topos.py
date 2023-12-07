import json
from collections import Counter, defaultdict, OrderedDict
from pathlib import Path
from typing import Union, Dict, List, Tuple, Sequence, Optional, Any, Mapping

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import scipy.optimize as opt
import sympy as sym
from sympy import Symbol
from ..Artist import plot_locators, plot_legend
from ..Schema import TopoBlockSet, ODTResult, TopoShow

__all__ = ["TopoCreater", "fA", "fB", "x", "Joint"]

fA = Symbol("fA", real=True, postive=True)
fB = Symbol("fB", real=True, postive=True)
fC = Symbol("fC", real=True, postive=True)
x = Symbol("x", real=True, postive=True)


class Joint:
    __count = 0

    def __init__(self, _id: int, new: bool = False):

        if new:
            Joint.__count = 0
        self.id = _id
        self.count_id = Joint.__count
        self.connection = []
        Joint.__count += 1

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

    def add(self, other_joint: Union["Joint", Sequence["Joint"]]) -> None:

        if isinstance(other_joint, Joint):
            other_joint = [other_joint]
        for j in other_joint:
            if self in j and j in self:
                continue
            if self not in j:
                j.connection.append(self)
            if j not in self:
                self.connection.append(j)

    def search_by_count_id(self, count_id: int):
        _log = set()
        q = [self]
        while q:
            current_joint = q.pop()
            _id = current_joint.id
            _count_id = current_joint.count_id
            if _count_id not in _log:
                _log.add(_count_id)
                if _count_id == count_id:
                    return current_joint
                else:
                    q.extend(current_joint.connection)
            else:
                continue
        print(f"Joint with count id = {count_id} do not exist!")

    def stats_connection(self, verbose: bool = True):
        if verbose:
            print("format: id(count_id)")
        exp_log = set()
        q = [self]
        res: dict[tuple, list] = defaultdict(list)
        while q:
            current_joint = q.pop()
            _id = current_joint.id
            _count_id = current_joint.count_id
            for child in current_joint.connection:
                child_id = child.id
                child_count_id = child.count_id
                exp = f"{_id}({_count_id})-{child_id}({child_count_id})"
                exp_inv = f"{child_id}({child_count_id})-{_id}({_count_id})"
                if exp not in exp_log or exp_inv not in exp_log:
                    res[tuple(sorted((_id, child_id)))].append(
                        (_count_id, child_count_id)
                    )
                    exp_log.add(exp)
                    exp_log.add(exp_inv)
                    if verbose:
                        print(exp)
                    q.append(child)
                else:
                    continue
        return res

    def attach_by_id(self, _id: int, other_id: int):
        _log = set()
        q = [self]
        while q:
            current_joint = q.pop()
            tmp_id = current_joint.id
            _count_id = current_joint.count_id
            if _count_id not in _log:
                _log.add(_count_id)
                for child in current_joint.connection:
                    if child.count_id not in _log:
                        q.append(child)
                if _id == tmp_id:
                    tmp_joint: Joint = Joint(_id=other_id)
                    current_joint.add(tmp_joint)
            else:
                continue

    @property
    def joint_num(self):
        return Joint.__count


class TopoCreater(nx.DiGraph):
    def __init__(self, verbose: bool = True):
        super(TopoCreater, self).__init__()
        self.SAB = None
        self.SBB = None
        self.SAA = None
        self.kind_edges = None
        self.edge_info = None
        self.type = None
        self.node_names = None
        self.node_nums = None
        self.final_func = None
        self.verbose = verbose

    def init_nodes(self, node_nums: int, node_names: list[str] = None):
        self.node_nums = node_nums
        if node_names:
            self.node_names = node_names
            for i in range(node_nums):
                self.add_node(i, name=node_names[i])
        else:
            for i in range(node_nums):
                self.add_node(i, name=str(i))

    def print(self, s: Any):
        if self.verbose:
            print(s)

    @staticmethod
    def stats(blocks: Union[str, list, dict], multiplier: int = 1):
        """
        >>> tp = TopoCreater()
        >>> print(tp.stats({'A':3, 'a':5}))
        Counter({'A': 8})
        """

        if isinstance(blocks, str):
            res = Counter(blocks.upper())
        elif isinstance(blocks, list):
            res = Counter(list(map(lambda block: block.upper(), blocks)))
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

    def add_di_edge(
        self,
        pairs: Tuple[int, int],
        fraction: Union[int, float, Symbol],
        name: str,
        **kwargs,
    ):
        self.add_edge(*pairs, fraction=fraction, name=name, **kwargs)
        self.add_edge(*pairs[::-1], fraction=fraction, name=name, **kwargs)

    def linear(
        self,
        blocks: Union[List[str], str, Dict],
        fractions: Optional[Sequence[Union[int, float, Symbol]]] = None,
        method: TopoBlockSet = TopoBlockSet.AUTO,
        **kwargs,
    ):
        if fractions is None:
            fractions = []
        self.clear()

        self.type = "linear"
        blocks_info = self.stats(blocks)
        if isinstance(blocks, dict):
            blocks = blocks_info.elements()

        self.init_nodes(sum(blocks_info.values()) + 1)
        if method == TopoBlockSet.AUTO and len(fractions) == 0:
            fraction = np.around(1 / sum(blocks_info.values()), 3)
            for i, b in enumerate(blocks):
                self.add_di_edge(
                    (i, i + 1),
                    fraction=fraction,
                    name="_".join([b, str(i + 1)]),
                    kind=b,
                )
        elif method == TopoBlockSet.MANUAL or len(fractions) != 0:
            if len(fractions) == 0 or len(fractions) != sum(blocks_info.values()):
                raise ValueError(
                    f"Fractions for blocks (length={len(fractions)}) is not right!"
                )
            if round(sum(fractions), 3) != 1:
                raise ValueError(f"Fractions do not sum to one, value={sum(fractions)}")
            for i, (b, f) in enumerate(zip(blocks, fractions)):
                self.add_di_edge(
                    (i, i + 1), fraction=f, name="_".join([b, str(i + 1)]), kind=b
                )
        else:
            raise NotImplementedError(method)
        self.get_info()

    def AmBn(
        self,
        blocks: Union[str, List[str], Dict],
        fractions: Optional[Sequence[Union[int, float, Symbol]]] = None,
        method: TopoBlockSet = TopoBlockSet.AUTO,
        **kwargs,
    ):

        if fractions is None:
            fractions = []
        self.clear()
        self.type = "AmBn"
        blocks_info = self.stats(blocks)
        if isinstance(blocks, dict):
            blocks = blocks_info.elements()

        self.init_nodes(sum(blocks_info.values()) + 1)
        if method == TopoBlockSet.AUTO and len(fractions) == 0:
            fraction = np.around(1 / sum(blocks_info.values()), 3)
            for i, b in enumerate(blocks):
                self.add_di_edge(
                    (0, i + 1),
                    fraction=fraction,
                    name="_".join([b, str(i + 1)]),
                    kind=b,
                )
        elif method == TopoBlockSet.MANUAL or len(fractions) != 0:
            if len(fractions) == 0 or len(fractions) != sum(blocks_info.values()):
                raise ValueError(
                    f"Fractions for blocks (length={len(fractions)}) is not right!"
                )
            if round(sum(fractions), 3) != 1:
                raise ValueError(f"Fractions do not sum to one, value={sum(fractions)}")
            for i, (b, f) in enumerate(zip(blocks, fractions)):
                self.add_di_edge(
                    (0, i + 1),
                    fraction=f,
                    name="_".join([b, str(i + 1)]),
                    kind=b,
                    minlen=f,
                )
        else:
            raise NotImplementedError(method)
        self.get_info()

    def star(
        self,
        blocks: Union[str, List[str]],
        fractions: Optional[Sequence[Union[int, float, Symbol]]] = None,
        arm: int = 5,
        head: bool = True,
        method: TopoBlockSet = TopoBlockSet.AUTO,
        **kwargs,
    ):

        if fractions is None:
            fractions = []
        self.clear()
        self.type = "star"
        blocks_info = self.stats(blocks, arm)
        if not head:
            blocks = blocks[::-1]

        self.init_nodes(sum(blocks_info.values()) + 1)
        if method == TopoBlockSet.AUTO and len(fractions) == 0:
            fraction = np.around(1 / sum(blocks_info.values()), 3)
            for a in range(arm):
                for i, b in enumerate(blocks):
                    idx_head = a * len(blocks) + i if i != 0 else 0
                    idx_tail = a * len(blocks) + i + 1
                    self.add_di_edge(
                        (idx_head, idx_tail),
                        fraction=fraction,
                        name="_".join([b, str(idx_tail)]),
                        kind=b,
                    )

        elif method == TopoBlockSet.MANUAL or len(fractions) != 0:
            if len(fractions) * 5 != sum(blocks_info.values()):
                raise ValueError(
                    f"Fractions for blocks (length={len(fractions)}*{arm}={len(fractions) * arm}) is not right!"
                )
            if round(sum(fractions) * arm, 3) != 1:
                raise ValueError(
                    f"Fractions do not sum to one, value={round(sum(fractions) * arm, 3)}"
                )
            for a in range(arm):
                for i, (b, f) in enumerate(zip(blocks, fractions)):
                    idx_head = a * len(blocks) + i if i != 0 else 0
                    idx_tail = a * len(blocks) + i + 1
                    self.add_di_edge(
                        (idx_head, idx_tail),
                        fraction=f,
                        name="_".join([b, str(idx_tail)]),
                        kind=b,
                        minlen=f,
                    )
        else:
            raise NotImplementedError(method)
        self.get_info()

    @classmethod
    def parseJson(cls, path: Union[Path, str, Mapping]) -> np.ndarray:
        if not isinstance(path, Mapping):
            with open(path, mode="r") as fp:
                data = json.load(fp, object_pairs_hook=OrderedDict)
        else:
            data = path
        topo_mat = []
        for block in data["Block"]:
            id1 = block["LeftVertexID"]
            id2 = block["RightVertexID"]
            mul = block["Multiplicity"]
            # if mul > 1:
            #     raise NotImplementedError(f"Mehthod for multiplicity larger thar {mul} is not supported.")
            if block["BranchDirection"] == "LEFT_BRANCH":
                direction = 0
                id_branch = id1
            elif block["BranchDirection"] == "RIGHT_BRANCH":
                direction = 1
                id_branch = id2
            else:
                raise ValueError
            id1, id2 = (id1, id2) if id1 < id2 else (id2, id1)
            fraction = block["ContourLength"]
            kind = block["ComponentName"]
            topo_mat.append([id1, id2, mul, direction, id_branch, fraction, kind])
        topo_mat = np.array(topo_mat, dtype=object)
        return topo_mat

    def fromJson(self, path: Union[Path, str, Mapping], **kwargs):
        topo_mat = self.parseJson(path)
        edges_info: dict[tuple, dict] = defaultdict(dict)
        count = 1
        for i, t in enumerate(topo_mat):
            core_id = t[t[3]]
            branch_id = t[1 - t[3]]
            branch_mul = t[2]
            edges_info[tuple(sorted((core_id, branch_id)))] = {
                "fraction": t[-2],
                "kind": t[-1],
                "name": "_".join([t[-1], str(count)]),
            }
            if i == 0:
                joint = Joint(_id=core_id, new=True)
            for j in range(branch_mul):
                joint.attach_by_id(_id=core_id, other_id=branch_id)
            count += 1
        connections = joint.stats_connection(verbose=self.verbose)
        node_num = joint.joint_num
        self.init_nodes(node_num)

        for k, v in connections.items():
            for e in v:
                self.add_di_edge(e, **edges_info[k])
        self.get_info()

    @staticmethod
    def count_nums(layers: int, branch: int):
        return sum([branch**i for i in range(layers)])

    def dendrimer(
        self,
        A_block_layer: int = 1,
        B_block_layer: int = 1,
        A_branch: int = 2,
        B_branch: int = 2,
        fractions: Optional[Sequence[Union[int, float, Symbol]]] = None,
        method: TopoBlockSet = TopoBlockSet.AUTO,
        **kwargs,
    ):

        # if fractions is None:
        #     fractions = []
        self.clear()
        self.type = "dendrimer"
        Ablock_num = self.count_nums(A_block_layer, A_branch)
        Ablock_out_num = Ablock_num - self.count_nums(A_block_layer - 1, A_branch)
        Bblock_num = (
            self.count_nums(B_block_layer, B_branch) * B_branch * Ablock_out_num
        )
        total_blocks = Ablock_num + Bblock_num
        total_nodes = total_blocks + 1
        self.init_nodes(total_nodes)

        if fractions is None:
            fractions = np.around(1 / total_nodes, 3) * np.ones(total_blocks)
        else:
            if len(fractions) != total_blocks:
                raise ValueError(
                    f"Fractions for blocks (length={len(fractions)}) is not right!"
                )
            if round(sum(fractions), 3) != 1:
                raise ValueError(
                    f"Fractions do not sum to one, value={round(sum(fractions), 3)}"
                )

        count = 0
        for i in range(A_block_layer):
            if i == 0:
                self.add_di_edge(
                    (0, 1),
                    fraction=fractions[count],
                    name="_".join(["A", str(count + 1)]),
                    kind="A",
                )
                count += 1
                last_layer_nodes = [count]
            else:
                this_layer_nodes = np.arange(
                    count + 1, count + A_branch**i + 1, 1
                )
                for j in last_layer_nodes:
                    tmp = 1
                    while tmp <= A_branch:
                        count += 1
                        self.add_di_edge(
                            (j, count),
                            fraction=fractions[count - 1],
                            name="_".join(["A", str(count + 1)]),
                            kind="A",
                        )
                        tmp += 1
                    last_layer_nodes = this_layer_nodes
        for i in range(A_block_layer, A_block_layer + B_block_layer, 1):
            beginning_node_num = len(last_layer_nodes)
            this_layer_nodes = np.arange(
                count + 1, count + 1 + beginning_node_num * B_branch, 1
            )
            for j in last_layer_nodes:
                tmp = 1
                while tmp <= B_branch:
                    count += 1
                    self.add_di_edge(
                        (j, count),
                        fraction=fractions[count - 1],
                        name="_".join(["B", str(count + 1)]),
                        kind="B",
                    )
                    tmp += 1
            last_layer_nodes = this_layer_nodes
        self.get_info()

    def get_info(self):

        self.edge_info = self.edges(data=True)
        self.kind_edges = {}
        for ix, iy, dic in self.edge_info:
            kind = dic["kind"]
            self.kind_edges[kind] = self.kind_edges.get(kind, []) + [(ix, iy)]

    def show_topo(
        self,
        colors: Optional[Sequence[str]] = None,
        node_size: int = 200,
        node_color: str = "gray",
        pos: Optional[dict] = None,
        figsize: Sequence[int] = (6, 6),
        interactive: bool = True,
        curve: bool = False,
        rad: float = 0.4,
        show_nodes: bool = True,
        show_edge_labels: bool = True,
        show_node_labels: bool = True,
        save: Optional[Union[Path, str, bool]] = False,
        **kwargs,
    ):

        if colors is None:
            colors = ["b", "r", "g", "blueviolet", "cyan"]
        fig, ax = plt.subplots(figsize=figsize)

        if pos is None:
            pos = nx.kamada_kawai_layout(self)

        kind_edges = list(self.kind_edges.items())
        kind_edges.sort(key=lambda x: x[0])
        kinds = list(self.kind_edges.keys())
        assert len(colors) >= len(kinds), "Not enough colors!"
        edge_kind = {}
        for k, v in kind_edges:
            for _e in v:
                edge_kind[_e] = k
        kind_color = dict(zip(kinds, colors[: len(kinds)]))

        if curve:
            rad_inverse_log = {}
            for edge in self.edges():
                source, target = edge
                if target < source:
                    continue
                if source not in rad_inverse_log:
                    flag = True
                else:
                    flag = not rad_inverse_log[source]
                rad_inverse_log[target] = flag
                _rad = rad if flag else -rad
                _kind = edge_kind[(source, target)]
                # nx.draw_networkx_edges(
                #     self,
                #     pos,
                #     edgelist=[edge],
                #     width=3,
                #     alpha=1,
                #     edge_color=kind_color[_kind],
                #     label=_kind,
                #     arrows=True,
                #     connectionstyle=f"arc3,rad={_rad}",
                #     arrowstyle="-",
                #     ax=ax,
                # )
                arrow_props = dict(
                    arrowstyle="-",
                    color=kind_color[_kind],
                    connectionstyle=f"arc3,rad={_rad}",
                    linestyle="-",
                    alpha=1.0,
                    linewidth=3,
                )
                ax.annotate(
                    "", xy=pos[source], xytext=pos[target], arrowprops=arrow_props
                )
            # print(rad_inverse_log)
        else:
            for i, (_kind, _edge) in enumerate(kind_edges):
                nx.draw_networkx_edges(
                    self,
                    pos,
                    edgelist=_edge,
                    width=3,
                    alpha=1,
                    edge_color=colors[i],
                    label=_kind,
                    arrows=False,
                    ax=ax,
                )

            plot_legend(**kwargs)

        node_size = node_size if show_nodes else 0
        node_color = node_color if show_nodes else None
        alpha = 1 if show_nodes else 0
        linewidths = 1 if show_nodes else 0
        nx.draw_networkx_nodes(
            self,
            pos,
            node_size=node_size,
            node_color=node_color,
            ax=ax,
            alpha=alpha,
            linewidths=linewidths,
        )

        if show_node_labels:
            nx.draw_networkx_labels(
                self,
                pos,
                labels=nx.get_node_attributes(self, "name"),
                font_size=12,
                font_family="sans-serif",
                font_color=kwargs.get("font_color", "white"),
                ax=ax,
            )

        if show_edge_labels:
            nx.draw_networkx_edge_labels(
                self, pos, edge_labels=nx.get_edge_attributes(self, "fraction"), ax=ax
            )
        plt.axis(False)
        if interactive:
            plt.show()
        return TopoShow(fig=fig, ax=ax, kind_color=kind_color, rad=rad)

    @classmethod
    def h(cls, f, x=x):
        return 2 / (x**2) * (f * x + sym.exp(-f * x) - 1)

    @classmethod
    def S(cls, f1, fdis, f2, x=x):
        result = (
            1
            / 2
            * (
                cls.h(f1 + fdis + f2, x)
                - cls.h(f1 + fdis, x)
                - cls.h(f2 + fdis, x)
                + cls.h(fdis, x)
            )
        )
        return result

    @classmethod
    def S_self(cls, f, x=x):
        return 1 / 2 * cls.h(f, x)

    def nodes_dis(self, start: int, end: int):
        path = nx.dijkstra_path(self, start, end)
        return sum(
            [
                self.get_edge_data(path[i], path[i + 1])["fraction"]
                for i in range(len(path) - 1)
            ]
        )

    def RPA(self, latex: bool = False):
        global x
        edgeNum = self.number_of_edges()
        edgeNum_half = int(edgeNum / 2)

        edges = []
        edge_fraction = []
        edge_kinds = []

        for i, j, dic in self.edge_info:
            if i < j:
                edges.append((i, j))
                edge_fraction.append(dic["fraction"])
                edge_kinds.append(dic["kind"])

        distance_matrix = np.zeros((edgeNum_half, edgeNum_half), dtype=object)
        for i in range(edgeNum_half):
            for j in range(edgeNum_half):
                if i == j:
                    distance_matrix[i][j] = 0
                    continue

                path_between_node = nx.dijkstra_path(self, edges[i][0], edges[j][1])
                if (
                    edges[i][1] in path_between_node
                    and edges[j][0] in path_between_node
                ):
                    tmpDis = self.nodes_dis(edges[i][1], edges[j][0])
                elif (
                    edges[i][1] in path_between_node
                    and edges[j][0] not in path_between_node
                ):
                    tmpDis = self.nodes_dis(edges[i][1], edges[j][1])
                elif (
                    edges[i][1] not in path_between_node
                    and edges[j][0] in path_between_node
                ):
                    tmpDis = self.nodes_dis(edges[i][0], edges[j][0])
                else:
                    tmpDis = self.nodes_dis(edges[i][0], edges[j][1])
                distance_matrix[i][j] = tmpDis

        Slist = np.zeros((edgeNum_half, edgeNum_half), dtype=object)
        for i in range(edgeNum_half):
            for j in range(i + 1):
                if j == i:
                    Slist[i][j] = self.S_self(edge_fraction[i], x)
                else:
                    Slist[i][j] = self.S(
                        edge_fraction[i], distance_matrix[i][j], edge_fraction[j], x
                    )
        self.SAA = 0
        self.SBB = 0
        self.SAB = 0

        for i in range(edgeNum_half):
            for j in range(i + 1):
                if edge_kinds[i] == edge_kinds[j] == "A":
                    self.SAA += 2 * Slist[i][j]
                elif edge_kinds[i] == edge_kinds[j] == "B":
                    self.SBB += 2 * Slist[i][j]
                else:
                    self.SAB += Slist[i][j]

        self.final_func = (
            1
            / 2
            * (self.SAA + self.SBB + 2 * self.SAB)
            / (self.SAA * self.SBB - self.SAB * self.SAB)
        )

        if latex is True:
            sym.print_latex(self.final_func)

    def ODT(
        self,
        fs: Union[List[float], np.ndarray] = np.arange(0.1, 1.0, 0.1),
        symbol: Symbol = fA,
        plot: bool = True,
        interactive: bool = True,
        xlabel: str = r"$f_{A}$",
        ylabel: str = r"$\chi {\rm N}$",
        save: Optional[Union[Path, str, bool]] = False,
        **kwargs,
    ):
        global x
        xN = []
        x_lim = kwargs.get("x_lim", (0, 1))
        y_lim = kwargs.get("y_lim", (0, 100))
        for i in fs:
            func = self.final_func.evalf(subs={symbol: i})
            func = sym.lambdify(x, func, "scipy")
            xN.append(opt.fminbound(func, *y_lim, full_output=True, maxfun=2000)[1])
        xN = np.array(xN)

        fig = ax = None
        if plot:
            fig, ax = plt.subplots(figsize=kwargs.get("figsize", (6, 6)))
            ax.plot(fs, xN, c="r", lw=2.5, marker="o", label="ODT")
            ax.set_xlim(x_lim)
            ax.set_ylim(y_lim)
            plot_locators(**kwargs)
            plot_legend(**kwargs)
            fontsize = kwargs.get("fontsize", 20)
            plt.xticks(fontsize=fontsize)
            plt.yticks(fontsize=fontsize)
            if xlabel:
                ax.set_xlabel(xlabel, fontsize=fontsize)
            if ylabel:
                ax.set_ylabel(ylabel, fontsize=fontsize)
            if interactive:
                plt.show()

        return ODTResult(
            f=fs,
            xN=xN,
            fig=fig,
            ax=ax,
            expression=self.final_func,
            xlabel=xlabel,
            ylabel=ylabel,
        )


if __name__ == "__main__":
    import doctest

    doctest.testmod()
