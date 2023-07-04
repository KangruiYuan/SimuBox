
from src import TopoCreater, fA
import os

# linear block
# tc = TopoCreater()
# tc.linear(blocks='AB', fractions=[fA, 1-fA])
# tc.show_topo()
# tc.RPA()
# tc.ODT(fAs=[0.1 * i for i in range(1, 10)])

# star block
# tc = TopoCreater()
# tc.star(blocks='BAB', fractions=[0.05, 0.12, 0.03])
# tc.show_topo(colorlist=['b','r'])
# tc.RPA()
# tc.ODT()

tc = TopoCreater()
tc.fromJson(os.path.join(os.path.dirname(__file__), '../datasets/Topo/input_star.json'))
tc.show_topo(colorlist=['b','r'])