from collections import OrderedDict

from Agent import BABCB
from Utils import SCFTManager, Options, arange, WHICH

opts = Options()
opts.cell = True
opts.anderson = True
opts.ergodic = True
opts.function = BABCB
opts.name = "WORKING_DIR"
opts.combine_paradict = OrderedDict(
    fA=[0.12, "1"],
    tau=[arange(0.69, 0.6, 0.02), "1"],
    chiNAC=[100, "0"],
    phase=[["C4", "Crect"], "1"],
)
opts.init_phin = ["DG", "NaCl", "CsCl"]
opts.which = WHICH.cpuTOPS
SCFTManager.opts = opts

SCFTManager.ParamsLog()
res = SCFTManager.readParamsArray()
# SCFTManager.pushJob(res)
