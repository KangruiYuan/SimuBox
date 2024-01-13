import json
import os
import shutil
import subprocess as sp
import traceback
from collections import OrderedDict
from enum import Enum
from itertools import chain, product
from pathlib import Path
from typing import Any, Callable, Union, Sequence

import numpy as np


def arange(
    start: Union[int, float], end: Union[int, float], step: Union[int, float], accuracy: int = 6
):
    if start > end:
        start, end = end, start
    gap = end - start
    num = gap // abs(step) + 1
    asc = np.arange(num) * abs(step)

    res = np.around(asc + start, accuracy)
    if end not in res:
        res = np.append(res, end)
    return np.sort(res)


class Cells(int, Enum):
    lx = 0
    ly = 1
    lz = 2


class Servers(str, Enum):
    cpuTOPS = "cpuTOPS"
    gpuSCFT = "gpuSCFT"
    gpuTOPS = "gpuTOPS"


class Ensemble(str, Enum):
    CANONICAL = "CANONICAL"
    GRANDCANONICAL = "GRANDCANONICAL"


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

    init_phin: list[str] = []
    gpuTOPS_require: list[str] = []
    gpuSCFT_require: list[str] = []

    which: Servers = Servers.cpuTOPS

    phase_base: str = "/mnt/sdd/kryuan/bcbab/PhoutLib/"

    para_file: str = "param_list.txt"

    @property
    def workdir(self):
        return self.filedir / self.name

    @property
    def worker(self):
        return dict(
            cpuTOPS=f"srun --partition=intel_2080ti,amd_3090,intel_Xeon --cpus-per-task=2 /home/share/TOPS2020/TOPS2020 -j -i={self.json_name} >aa.txt 2>&1 &",
            gpuSCFT=f"srun --partition=intel_2080,intel_2080ti,amd_3090 --nodes=1 --gpus=1 /home/share/scft2022 -i={self.json_name} >aa.txt 2>&1 &",
            gpuTOPS=f"srun --gpus=rtx_3090:1 --cpus-per-gpu=1 --partition=amd_3090 --gpus=1 -w gpu04 /home/share/TOPS2020/TOPS_device -j -i={self.json_name} > aa.txt 2>&1 &",
        )


class SCFTManager:
    opts: Options

    @classmethod
    def collect_nodes(cls, blocks: list):
        nodes = set(
            chain(
                *[(block["LeftVertexID"], block["RightVertexID"]) for block in blocks]
            )
        )
        nodes = sorted(list(nodes))
        return nodes

    @classmethod
    def clean(cls, input_dict: dict):
        clean_block = []
        replace_pairs = dict()

        for block in input_dict["Block"]:
            if block["ContourLength"] == 0:
                # 将错误节点的左节点指向右节点
                replace_pairs[block["LeftVertexID"]] = block["RightVertexID"]
            else:
                # 只保留有效的嵌段
                clean_block.append(block.copy())

        # 修改有效嵌段中右节点的指向
        # TODO: 类似并查集，实现union的优化，目前需求不大，因为嵌段非常少
        for block in clean_block:
            while block["RightVertexID"] in replace_pairs:
                block["RightVertexID"] = replace_pairs[block["RightVertexID"]]

        nodes = cls.collect_nodes(clean_block)
        correct_nodes = list(range(len(nodes)))
        idx = 0
        # 将现有节点与正确节点对齐
        while idx < len(nodes):
            if nodes[idx] != correct_nodes[idx]:
                thresh = correct_nodes[idx]
                diff = nodes[idx] - correct_nodes[idx]
                for block in clean_block:
                    if block["LeftVertexID"] > thresh:
                        block["LeftVertexID"] -= diff
                    if block["RightVertexID"] > thresh:
                        block["RightVertexID"] -= diff
                nodes = cls.collect_nodes(clean_block)
            assert nodes[idx] == correct_nodes[idx]
            idx += 1

        input_dict["Block"] = clean_block

    @classmethod
    def editParams(cls, paramNameList: list[str], paramValueList: list[Any]):
        assert len(paramValueList) == len(paramValueList)

        with open(cls.opts.json_name, mode="r") as fp:
            input_dict = json.load(fp, object_pairs_hook=OrderedDict)

        if "Scripts" not in input_dict:
            input_dict["Scripts"] = {}

        for pn, pv in zip(paramNameList, paramValueList):
            cls.opts.function(pn, pv, input_dict, cls.opts)

        if cls.opts.clean:
            cls.clean(input_dict)

        if cls.opts.cell:
            input_dict["Iteration"]["VariableCell"]["Switch"] = "AUTO"
        else:
            input_dict["Iteration"]["VariableCell"]["Switch"] = "FORCED_OFF"

        if cls.opts.anderson:
            input_dict["Iteration"]["AndersonMixing"]["Switch"] = "AUTO"
        else:
            input_dict["Iteration"]["AndersonMixing"]["Switch"] = "FORCED_OFF"

        with open(cls.opts.json_name, "w") as f:
            json.dump(input_dict, f, indent=4, separators=(",", ": "))

    @classmethod
    def ParamsLog(cls, **kwargs):
        with open(cls.opts.para_file, "w") as para:
            para.write(" ".join(cls.opts.combine_paradict.keys()) + "\n")
            para.write(
                " ".join([i[-1] for i in cls.opts.combine_paradict.values()]) + "\n"
            )
            arr_len = np.inf
            ParaList = []
            for val in cls.opts.combine_paradict.values():
                if not isinstance(val[0], (np.ndarray, Sequence)):
                    ParaList.append([val[0]])
                    # arr_len = min(arr_len, 1)
                else:
                    arr_len = min(arr_len, len(val[0]))
                    ParaList.append(val[0])
            assert arr_len != np.inf
            if cls.opts.ergodic:
                paras = list(product(*ParaList))
            else:
                ParaAssem = []
                for pl in ParaList:
                    if len(pl) == 1:
                        ParaAssem.append(list(pl) * int(arr_len))
                    elif len(pl) > arr_len:
                        ParaAssem.append(pl[:arr_len])
                    else:
                        ParaAssem.append(pl)
                paras = list(zip(*ParaAssem))
            for p in paras:
                para.write(" ".join(list(map(str, p))) + "\n")

    @classmethod
    def IsFloatNum(cls, string: str):
        """
        输入参数 str 需要判断的字符串
        返回值  True：该字符串为浮点数；False：该字符串不是浮点数。
        :param string:
        :return:
        """
        s = string.split(".")
        if len(s) > 2:
            return False
        else:
            for si in s:
                if not si.isdigit():
                    return False
            return True

    @classmethod
    def readParamsArray(cls, acc: int = 6):
        if cls.opts.para_file not in os.listdir(cls.opts.filedir):
            print(f"ERROR: {cls.opts.para_file} not found.")
            return

        parasAll = np.loadtxt(cls.opts.para_file, dtype=object)
        nameList = parasAll[0]
        inOrNot = list(map(bool, map(int, parasAll[1])))
        paras = parasAll[2:]

        res = []
        for p in paras:
            pt = [round(float(i), acc) if cls.IsFloatNum(i) else i for i in p]
            res.append(
                {
                    "paramNameList": nameList,
                    "dirName": "_".join(
                        [i + j for i, j in zip(nameList[inOrNot], p[inOrNot])]
                    ),
                    "paramValueArray": pt,
                }
            )
        return res

    @classmethod
    def pushJob(cls, res):
        if cls.opts.json_name not in os.listdir(cls.opts.filedir):
            print(f"ERROR: {cls.opts.json_name} not found.")
            return

        if not cls.opts.workdir.exists():
            cls.opts.workdir.mkdir()

        for r in res:
            paramNameList = r["paramNameList"]
            dirName = r["dirName"]
            paramValueArray = r["paramValueArray"]

            if dirName in os.listdir(cls.opts.workdir):
                print("ERROR: " + dirName + " already existed.")
                continue
            else:
                dirPath = os.path.join(cls.opts.workdir, dirName)
                print(f"Dealing: {dirPath}")
                os.makedirs(dirPath)
                shutil.copy2(cls.opts.json_name, dirPath)

                idx = list(paramNameList).index("phase")
                phase = paramValueArray[idx]
                if phase in cls.opts.init_phin:
                    try:
                        shutil.copy2(
                            os.path.join(cls.opts.phase_base, str(phase) + "_phin.txt"),
                            os.path.join(dirPath, "phin.txt"),
                        )
                    except:
                        traceback.print_exc()

                os.chdir(dirPath)
                cls.editParams(paramNameList, paramValueArray)

                if phase in cls.opts.gpuSCFT_require:
                    _ = sp.Popen(
                        cls.opts.worker[Servers.gpuSCFT], shell=True, stdout=sp.PIPE
                    )
                elif phase in cls.opts.gpuTOPS_require:
                    _ = sp.Popen(
                        cls.opts.worker[Servers.gpuTOPS], shell=True, stdout=sp.PIPE
                    )
                else:
                    _ = sp.Popen(
                        cls.opts.worker[cls.opts.which], shell=True, stdout=sp.PIPE
                    )

            os.chdir(cls.opts.filedir)

        print("OK :)")
