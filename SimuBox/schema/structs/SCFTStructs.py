from pathlib import Path
from ..enums.SCFTEnums import Ensemble, Servers
from typing import Union, Callable, Any, Sequence
from collections import OrderedDict


class Options:
    filedir: Path = Path.cwd()
    name: str = "WORKING_DIR"

    json_name: str = "input.json"

    cell: bool = True
    anderson: bool = True
    ensemble: Ensemble = Ensemble.CANONICAL

    ergodic: bool = True

    function: Union[Callable, Any]

    clean: bool = True

    combine_paradict: OrderedDict[str, list[Any]] = OrderedDict()

    init_phin: Sequence[str] = []
    server: Servers = Servers.cpuTOPS
    gpuTOPS_require: Sequence[str] = []
    gpuSCFT_require: Sequence[str] = []

    phase_base: str = ""

    para_file: str = "param_list.txt"

    @property
    def workdir(self):
        return self.filedir / self.name

    @property
    def worker(self):
        return dict(
            cpuTOPS=f"srun --partition=intel_Xeon,amd_4090,amd_3090,intel_2080ti --cpus-per-task=2 /home/share/TOPS2020/TOPS2020 -j -i={self.json_name} >aa.txt 2>&1 &",
            gpuSCFT=f"srun --partition=amd_4090,amd_3090,intel_2080ti,intel_2080 --nodes=1 --gpus=1 /home/share/scft2024 -i={self.json_name} >aa.txt 2>&1 &",
            gpuTOPS=f"srun --gpus=rtx_3090:1 --cpus-per-gpu=1 --partition=amd_3090 --gpus=1 -w gpu04 /home/share/TOPS2020/TOPS_device -j -i={self.json_name} > aa.txt 2>&1 &",
        )
