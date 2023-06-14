from SimuBox import Scatter
import os

s = Scatter(os.path.join(os.path.dirname(__file__), '../datasets/scatter/'), inverse_flag=True)
s.collect()
s.sacttering_peak(s.phi0, cutoff=300)
x = s.PeakShow(w=0.01, cutoff=300)
print(x)