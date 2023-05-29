
import os
import json
import shutil
import subprocess as sp
import numpy as np
from collections import OrderedDict
from itertools import product
from .lib_tools import mypara
from .template import Mask_AB_A


class SCFTManager():

    cpuJob = r"srun --partition=intel_2080ti,amd_3090,intel_Xeon --cpus-per-task=2 /home/share/TOPS2020/TOPS2020 -j -i=input.json >aa.txt 2>&1 &"
    gpuJob = r"srun --partition=intel_2080,intel_2080ti,amd_3090 --nodes=1 --gpus=1 /home/share/scft2022 >aa.txt 2>&1 &"
    gputops = r"srun --gpus=rtx_3090:1 --cpus-per-gpu=1 --partition=amd_3090 --gpus=1 -w gpu04 /home/share/TOPS2020/TOPS_device -j -i=input.json > aa.txt 2>&1 &"
    phase_base = ""

    @classmethod
    def editParams(cls, paramNameList, paramValueList, cell, anderson, func):
        assert len(paramValueList) == len(paramValueList)
        with open('input.json', "r") as f:
            input_dict = json.load(f, object_pairs_hook=OrderedDict)

        for pn, pv in zip(paramNameList, paramValueList):
            func(pn, pv, input_dict)

        if cell:
            input_dict["Iteration"]["VariableCell"]["Switch"] = 'AUTO'
        else:
            input_dict["Iteration"]["VariableCell"]["Switch"] = 'FORCED_OFF'

        if anderson:
            input_dict["Iteration"]["AndersonMixing"]["Switch"] = 'AUTO'
        else:
            input_dict["Iteration"]["AndersonMixing"]["Switch"] = 'FORCED_OFF'
            

        with open('input.json', "w") as f:
            json.dump(input_dict, f, indent=4, separators=(",", ": "))

    @classmethod
    def ParamsLog(cls, paradict:dict, ergodic:bool=True, filename='param_list.txt', **kwargs):
        cls.filename = filename
        with open(filename, 'w') as para:
            para.write(" ".join(paradict.keys()) + '\n')
            para.write(" ".join([i[-1] for i in paradict.values()]) + '\n')
            arr_len = np.inf
            ParaList = []
            for val in paradict.values():
                if not isinstance(val[0], (np.ndarray, list)):
                    ParaList.append([val[0]])
                else:
                    if 1< len(val[0]) < arr_len:
                        arr_len = len(val[0])
                    ParaList.append(val[0])
            if ergodic:
                paras = list(product(*ParaList))
            else:
                ParaAssem = []
                for pl in ParaList:
                    if len(pl) == 1:
                        ParaAssem.append(list(pl)*arr_len)
                    elif len(pl) > arr_len:
                        ParaAssem.append(pl[:arr_len])
                    else:
                        ParaAssem.append(pl)
                paras = list(zip(*ParaAssem))
            for p in paras:
                    para.write(' '.join(list(map(str, p))) + '\n')
                    
    # 输入参数 str 需要判断的字符串
    # 返回值  True：该字符串为浮点数；False：该字符串不是浮点数。
    @classmethod
    def IsFloatNum(cls, str):
        s = str.split('.')
        if len(s) > 2:
            return False
        else:
            for si in s:
                if not si.isdigit():
                    return False
            return True
    
    @classmethod
    def readParamsArray(cls, acc=6):
        if cls.filename not in os.listdir(cls.fileDir):
            print(f"ERROR: {cls.filename} not found.")
            return
        
        parasAll = np.loadtxt(cls.filename, dtype=object)
        nameList = parasAll[0]
        inOrNot = list(map(bool, map(int, parasAll[1])))
        # inOrNot = list(map(bool, map(int, parasAll[1])))
        paras = parasAll[2:]
        
        res = []
        for p in paras:
            pt = [round(float(i), acc) if cls.IsFloatNum(i) else i for i in p]
            res.append(
                {
                    'paramNameList': nameList,
                    'dirName':'_'.join([i+j for i,j in zip(nameList[inOrNot], p[inOrNot])]),
                    'paramValueArray':pt,
                }
            )
        return res
    
    @classmethod
    def pushJob(cls, res):
        global cell
        global anderson
        global func
                
        if "input.json" not in os.listdir(cls.fileDir):
            print("ERROR: input.json not found.")
            return
        
        if not os.path.exists(cls.workDir):
            os.makedirs(cls.workDir)

        for r in res:
            paramNameList = r['paramNameList']
            dirName = r['dirName']
            paramValueArray = r['paramValueArray']
            
            if dirName in os.listdir(cls.workDir):
                print("ERROR: " + dirName + " already existed.")
                continue
            else:
                
                dirPath = os.path.join(cls.workDir, dirName)
                print(f"Dealing: {dirPath}")
                os.makedirs(dirPath)
                shutil.copy2("input.json", dirPath)
                
                
                idx = list(paramNameList).index('phase')
                phase = paramValueArray[idx]
                if phase in ['peanut']:
                    try:
                        shutil.copy2(os.path.join(
                            cls.phase_base, str(phase) + '_phin.txt'), 
                            os.path.join(dirPath, 'phin.txt'))
                    except:
                        pass

                os.chdir(dirPath)
                cls.editParams(paramNameList, paramValueArray, cell=cell, anderson=anderson, func=func)
                
                command = ' '.join([
                    'python3', '/mnt/sdd/kryuan/DSA/now/genMask_2d_read_alphashape_curvature.py', '-p', str(phase)])
                job = sp.Popen(
                    command, shell=True, stdout=sp.PIPE)
                job.wait()

                _ = sp.Popen(cls.gpuJob, shell=True, stdout=sp.PIPE)

            os.chdir(cls.fileDir)
            
        print("OK :)")
        


if __name__ == '__main__':

    workDir = 'WORKING_DIR'
    cell = False
    anderson = True
    ergodic = True
    func = Mask_AB_A

    combine_paradict = OrderedDict(

        fA=[0.3, '0'],
        gamma_B=[1, '0'],
        phi_AB=[1, '0'],
        cylDis=[3.0, '1'],
        xN=[25, '1'],
        xAW=[20, '0'],
        xBW=[0, '0'],
        r=[3.0, '0'],
        cw=[0, '0'],
        phase=['C2', '1'],
        shapeIdx=[7, '0'],
        ratio=[mypara(2.2, 2.9, 0.1), '1']
    )

    SCFTManager.fileDir = os.getcwd()
    SCFTManager.workDir = os.path.join(SCFTManager.fileDir, workDir)
    SCFTManager.ParamsLog(combine_paradict, ergodic, filename='param_list.txt')
    res = SCFTManager.readParamsArray()
    SCFTManager.pushJob(res)


