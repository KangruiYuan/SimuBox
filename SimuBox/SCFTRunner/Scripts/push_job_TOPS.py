from collections import OrderedDict

try:
    from SCFTRunner import BABCB, SCFTManager, Options, arange, Servers
except ImportError:
    from SimuBox import BABCB, SCFTManager, Options, arange, Servers

import platform

agent = BABCB()

opts = Options()
# opts.json_name = "BABCB.json"
opts.cell = True
opts.anderson = True
opts.ergodic = True
opts.function = agent
opts.name = "WORKING_DIR"
opts.combine_paradict = OrderedDict(
    fA=[0.12, "1"],
    tau=[arange(0.69, 0.6, 0.02), "1"],
    chiNAC=[100, "0"],
    phase=[["C4", "Crect"], "1"],
)
opts.init_phin = ["DG", "NaCl", "CsCl"]
opts.which = Servers.cpuTOPS
SCFTManager.opts = opts

if __name__ == "__main__":
    SCFTManager.ParamsLog()
    res = SCFTManager.readParamsArray()
    if platform.system() == "Linux":
        SCFTManager.pushJob(res)
