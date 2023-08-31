import json
import os
import shutil
import subprocess as sp
import traceback
from collections import OrderedDict
from enum import Enum
from itertools import product
from pathlib import Path
from typing import Any, Callable, Union

import numpy as np


def arange(
    start: Union[int, float], end: Union[int, float], step: Union[int, float], **kwargs
):
    acc = kwargs.get("acc", 6)
    if start > end:
        start, end = end, start
    gap = end - start
    num = gap // abs(step) + 1
    asc = np.arange(num) * abs(step)

    res = np.around(asc + start, acc)
    if end not in res:
        res = np.append(res, end)
    return np.sort(res)


class Cells(int, Enum):
    lx = 0
    ly = 1
    lz = 2
    alpha = 3
    beta = 4
    gamma = 5


class WHICH(str, Enum):
    cpuTOPS = "cpuTOPS"
    gpuSCFT = "gpuSCFT"
    gpuTOPS = "gpuTOPS"


class Options:
    filedir: Path = Path.cwd()
    name: str = "WORKING_DIR"
    workdir: Path = filedir / name

    cell: bool = True
    anderson: bool = True
    ergodic: bool = True

    function: Callable

    combine_paradict: OrderedDict[str, list[Any]] = OrderedDict()

    init_phin: list[str] = []

    worker: dict[str, str] = dict(
        cpuTOPS=r"srun --partition=intel_2080ti,amd_3090,intel_Xeon --cpus-per-task=2 /home/share/TOPS2020/TOPS2020 -j -i=input.json >aa.txt 2>&1 &",
        gpuSCFT=r"srun --partition=intel_2080,intel_2080ti,amd_3090 --nodes=1 --gpus=1 /home/share/scft2022 >aa.txt 2>&1 &",
        gpuTOPS=r"srun --gpus=rtx_3090:1 --cpus-per-gpu=1 --partition=amd_3090 --gpus=1 -w gpu04 /home/share/TOPS2020/TOPS_device -j -i=input.json > aa.txt 2>&1 &",
    )

    which: WHICH = WHICH.cpuTOPS

    phase_base: str = "/mnt/sdd/kryuan/bcbab/PhoutLib/"

    para_file: str = "param_list.txt"


class SCFTManager:
    opts: Options

    @classmethod
    def editParams(cls, paramNameList: list[str], paramValueList: list[Any]):
        assert len(paramValueList) == len(paramValueList)

        with open("input.json", mode="r") as fp:
            input_dict = json.load(fp, object_pairs_hook=OrderedDict)

        if "Scripts" not in input_dict:
            input_dict["Scripts"] = {}

        for pn, pv in zip(paramNameList, paramValueList):
            cls.opts.function(pn, pv, input_dict, cls.opts)

        if cls.opts.cell:
            input_dict["Iteration"]["VariableCell"]["Switch"] = "AUTO"
        else:
            input_dict["Iteration"]["VariableCell"]["Switch"] = "FORCED_OFF"

        if cls.opts.anderson:
            input_dict["Iteration"]["AndersonMixing"]["Switch"] = "AUTO"
        else:
            input_dict["Iteration"]["AndersonMixing"]["Switch"] = "FORCED_OFF"

        with open("input.json", "w") as f:
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
                if not isinstance(val[0], (np.ndarray, list)):
                    ParaList.append([val[0]])
                else:
                    if 1 < len(val[0]) < arr_len:
                        arr_len = len(val[0])
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
        if "input.json" not in os.listdir(cls.opts.filedir):
            print("ERROR: input.json not found.")
            return

        if not os.path.exists(cls.opts.workdir):
            os.makedirs(cls.opts.workdir)

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
                shutil.copy2("input.json", dirPath)

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

                _ = sp.Popen(
                    cls.opts.worker[cls.opts.which], shell=True, stdout=sp.PIPE
                )

            os.chdir(cls.opts.filedir)

        print("OK :)")
