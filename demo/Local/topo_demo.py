
from src import TopoCreater
import networkx as nx

tc = TopoCreater()
tc.linear(blocks='AAABBB', fractions=[0.1,0.1,0.1,0.2,0.3,0.2])
print(tc.edge_kinds.items())
tc.show_topo()
