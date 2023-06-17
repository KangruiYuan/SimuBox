
from src import TopoCreater, fA
import networkx as nx

tc = TopoCreater()
tc.linear(blocks='AB', fractions=[fA, 1-fA])
tc.show_topo()
tc.RPA()
tc.ODT(fAs=[0.1 * i for i in range(1, 10)])