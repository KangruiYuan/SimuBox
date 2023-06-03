#! /home/kryuan/miniconda3/envs/gputensor/bin/python

import os
import json
import subprocess as sp
import argparse
from collections import OrderedDict
import re

lxlylz_re = re.compile('[.0-9]+(?!e)')
freeE_re = re.compile('[.0-9e+-]+')

parser = argparse.ArgumentParser(description="Parameters for data extraction")
parser.add_argument("-n", "--name", default='EXTRACTED_DATA',
                    type=str, help="the File name")
parser.add_argument("-d", "--dir", default=False, action='store_true')
parser.add_argument("-t", "--terminal", default='WORKING_DIR',
                    type=str, help="the Work Dir")
args = parser.parse_args()
main_dir = os.getcwd()

cmd_lxlylz = "tail -3 printout.txt | head -1"
cmd_freeE_MaxC = "tail -1 printout.txt | head -1"
cpuJob = r"srun --partition=intel_2080ti,amd_3090,intel_Xeon --cpus-per-task=2 /home/share/TOPS2020/TOPS2020 -j -i=input.json >aa.txt 2>&1 &"
gpuTOPs = r"srun --gpus=rtx_3090:1 --cpus-per-gpu=1 --partition=amd_3090 --gpus=1 -w gpu04 /home/share/TOPS2020/TOPS_device -j -i=input.json > aa.txt 2>&1 &"
gpuJob = r"srun --partition=intel_2080,intel_2080ti,amd_3090 --nodes=1 --gpus=1 /home/share/scft2022 >aa.txt 2>&1 &"

def getSubDirs(workdir, subList=[]):
    subdir = os.listdir(workdir)
    for ex in ['zmin', 'cros.csv']:
        if ex in subdir:
            subdir.remove(ex)
    subdir = [os.path.join(workdir, i) for i in subdir]

    for i in subdir:
        if os.path.isdir(i) and os.path.isfile(os.path.join(i, 'input.json')):
            subList.append(i)
        else:
            getSubDirs(i, subList=subList)
            
    return subList

subList = getSubDirs(workdir=args.terminal)

def readJson(i):
    try:
        with open("./input.json", mode='r') as fp:
            json_base = json.load(fp, object_pairs_hook=OrderedDict)
            return json_base
    except json.decoder.JSONDecodeError as e:
        print(repr(e), " : ", i)
        return None

def FindWrongData(subList):
    WrongList = []
    
    for i in subList:
        tmp_path = os.path.join(main_dir, i)
        os.chdir(tmp_path)

        json_base = readJson(i)
        if json_base is None: continue
        # phase_log = json_base['_phase_log']
        which_type = json_base.get('_which_type', 'cpu')
        if which_type == 'gpu' or os.path.isfile('fet.dat'):
            with open('fet.dat', 'r') as fp:
                cont = fp.readlines()
            dataDict = [line.strip().split()[1:] for line in cont]
            dataDict = {line[0]:float(line[1]) for line in dataDict if len(line) == 2}
            # print(dataDict)
            if json_base['Iteration']['IncompressibilityTarget'] < dataDict["inCompMax"]:
                WrongList.append({
                    'path': tmp_path,
                    'lxlylz': [dataDict['lx'], dataDict['ly'], dataDict['lz']],
                    'step': 3000,
                    'TYPE':'gpu'
                })
                print(tmp_path)
                
        elif which_type == 'cpu':
            popen_freeE_MaxC = sp.Popen(cmd_freeE_MaxC, shell=True, stdout=sp.PIPE, stderr=sp.PIPE)
            popen_freeE_MaxC.wait()
            line_freeE_MaxC = popen_freeE_MaxC.stdout.readline()
            uws = list(map(float,
                        freeE_re.findall(str(line_freeE_MaxC, encoding="UTF-8").strip())))
            
            popen_lxlylz = sp.Popen(
                cmd_lxlylz, shell=True, stdout=sp.PIPE, stderr=sp.PIPE)
            popen_lxlylz.wait()
            line_lxlylz = popen_lxlylz.stdout.readline()
            try:
                lxlylz = list(
                    map(float, str(line_lxlylz, encoding="UTF-8").strip().split(" ")))
            except:
                print('Wrong:', tmp_path)
                continue
        
            if json_base['Iteration']['IncompressibilityTarget'] < uws[-1] or uws[2] < 0:
                WrongList.append({
                    'path': tmp_path,
                    'lxlylz': lxlylz,
                    'step':int(uws[0] + 1000)
                    })
                print(tmp_path)
    return WrongList

WrongList = FindWrongData(subList)

def RepushOrDelete(wrongList):
    if len(wrongList) == 0:
        return
    mode = input('Delete(d) or Repush(r)?(d/r/[pass])') or "pass"
    if mode == 'r':
        repush = input('Continue(c) or Start over again(s)?([c]/s)') or "c"
        for i in wrongList:
            tmp_path = i['path']
            os.chdir(tmp_path)
            if repush == 'c':
                json_base = readJson(tmp_path)
                json_base["Iteration"]["MaxStep"] = i['step']
                json_base['Initializer']["UnitCell"]['Length'] = i['lxlylz']
                if os.path.isfile(os.path.join(tmp_path, 'phout.txt')):
                    sp.Popen("mv phout.txt phin.txt", shell=True, stdout=sp.PIPE)
                    json_base["Initializer"]["Mode"] = "FILE"
                    json_base["Initializer"]["FileInitializer"] = {
                    "Mode": "OMEGA",
                    "Path": "phin.txt",
                    "SkipLineNumber": 1 if i['TYPE'] == 'cpu' else 2}
                    json_base["Iteration"]["AndersonMixing"]["Switch"] = 'AUTO'
                    
                    with open('input.json', "w") as f:
                        json.dump(json_base, f, indent=4,
                                  separators=(",", ": "))
                    if i['TYPE'] == 'cpu':
                        job = sp.Popen(cpuJob, shell=True, stdout=sp.PIPE)
                    else:
                        job = sp.Popen(gpuJob, shell=True, stdout=sp.PIPE)
                else:
                    with open('input.json', "w") as f:
                        json.dump(json_base, f, indent=4,
                                  separators=(",", ": "))
                    job = sp.Popen(cpuJob, shell=True, stdout=sp.PIPE)
                
                print("{pid}:{path}".format(pid=job.pid, path=tmp_path))
            elif repush == 's':
                pass
    elif mode == 'd':
        for i in wrongList:
            tmp_path = i['path']
            sp.Popen('rm -r ' + tmp_path, shell=True, stdout=sp.PIPE)
        print('Delete finished!')
        
RepushOrDelete(WrongList)

# cmd1 = "tail ./printout.txt -n 1 -q | awk {'print $3'}"
# cmd2 = "tail ./printout.txt -n 1 -q | awk {'print $8'}"
# cmd3 = "tail -3 printout.txt | head -1"
# cmd4 = "tail -2 printout.txt | head -1"
# cmd5 = "tail -1 printout.txt | head -1"
# cpuJob = r"srun --partition=intel_2080ti,amd_3090 --cpus-per-task=2 /home/share/TOPS2020/TOPS2020 -j -i=input.json >aa.txt 2>&1 &"
# gpuJob = r"srun --partition=intel_2080,intel_2080ti,amd_3090 --nodes=1 --gpus=1 /home/share/scft2021 >aa.txt 2>&1 &"

# main_dir = os.getcwd()
# main_dir = os.path.join(main_dir, args.terminal)
# sub_dirs = os.listdir(main_dir)
# sub_dirs = sorted(sub_dirs)
# Wrong_list = list()
# if not args.dir:
#     extract_name = args.name.strip('.csv') + '.csv'
# else:
#     temp_name = os.getcwd()
#     temp_name = temp_name.split('/')[-1]
#     extract_name = temp_name + '.csv'


# with open(extract_name, 'w', newline='') as csvfile:
#     writer = csv.writer(csvfile)
#     writer.writerow(['lx', 'ly', 'lz', 'alpha', 'beta',
#                     'gamma', 'freeE', 'freeAB', 'freeWS', 'MaxComp', 'phase', 'ksi', 'tau', 'cal_type', 'bridge', 'width', 'width_var', 'surface', 'surface_rela', 'AB1', 'AB2', 'AB3'])
#     for subdir in sub_dirs:
#         if subdir == 'old':
#             continue
#         temp_path = os.path.join(main_dir, subdir)
#         if os.path.isdir(temp_path):
#             os.chdir(temp_path)
#             if not os.path.isfile('printout.txt'):
#                 continue
#             try:
#                 with open("./input.json", mode='r') as fp:
#                     json_base = json.load(fp, object_pairs_hook=OrderedDict)
#             except json.decoder.JSONDecodeError as e:
#                 print(repr(e), " : ", temp_path)
#                 continue
#                 # [lx, ly, lz, alpha, beta,
#                 #     gamma] = json_base['Initializer']["UnitCell"]['Length'][:]
#             phase_log = json_base['_phase_log']
#             ksi_log = json_base['_ksi_log']
#             tau_log = json_base['_tau_log']
#             try:
#                 which_type = json_base['_which_type']
#             except KeyError:
#                 which_type = 'cpu'

#             if which_type == 'cpu':
#                 popen1 = sp.Popen(cmd1, shell=True,
#                                   stdout=sp.PIPE, stderr=sp.PIPE)
#                 a = popen1.wait()
#                 popen2 = sp.Popen(cmd2, shell=True,
#                                   stdout=sp.PIPE, stderr=sp.PIPE)
#                 b = popen2.wait()
#                 popen3 = sp.Popen(cmd3, shell=True,
#                                   stdout=sp.PIPE, stderr=sp.PIPE)
#                 c = popen3.wait()
#                 line1 = popen1.stdout.readline()
#                 line2 = popen2.stdout.readline()
#                 line3 = popen3.stdout.readline()

#                 popen5 = sp.Popen(cmd5, shell=True,
#                                   stdout=sp.PIPE, stderr=sp.PIPE)
#                 d = popen5.wait()
#                 line4 = popen5.stdout.readline()
#                 try:
#                     freeE1 = float(eval(str(line1, encoding="UTF-8").strip()))
#                 except (SyntaxError, TypeError) as e:
#                     print(repr(e), " : ", temp_path)
#                     continue

#                 MaxComp2 = float(eval(str(line2, encoding="UTF-8").strip()))
#                 temp_str = str(line3, encoding="UTF-8").strip()
#                 temp_str = temp_str.split(" ")
#                 # print(temp_str)
#                 value_list = list(map(float, temp_str))

#                 uws = list(
#                     map(float,
#                         freeE_re.findall(str(line4, encoding="UTF-8").strip())
#                         )
#                 )
#                 freeAB = uws[2]
#                 freeWS = uws[3] + uws[4]

#                 if os.path.isfile('log_bridge.txt'):
#                     bridge_ratio = open(
#                         'log_bridge.txt', 'r').readline().strip()
#                 else:
#                     # print(temp_path, " :No log_bridge.txt")
#                     bridge_ratio = 0

#                 if os.path.isfile('width_log.txt'):
#                     width = open(
#                         'width_log.txt', 'r').readline().strip().split(' ')
#                 else:
#                     width = [0, 0]

#                 if os.path.isfile('surface.txt'):
#                     surface = open(
#                         'surface.txt', 'r').readline().strip().split(' ')
#                 else:
#                     surface = [0, 0]

#                 if os.path.isfile('freeAB_log.txt'):
#                     freeAB_split = open(
#                         'freeAB_log.txt', 'r').readline().strip().split(' ')
#                 else:
#                     freeAB_split = [0, 0, 0]

#                 # print(temp_path, round(value_list[1]/value_list[2], 6))
#                 if phase_log == 'Crect' and round(value_list[1]/value_list[2], 3) == 1:
#                     # phase_log = 'C4'
#                     print(temp_path)
#                     Wrong_list.append(temp_path)
#                     continue
#                 elif phase_log == 'C4' and round(value_list[1]/value_list[2], 3) != 1:
#                     # phase_log = 'Crect'
#                     print(temp_path)
#                     Wrong_list.append(temp_path)
#                     continue

#                 if MaxComp2 > 1e-8:
#                     print('{} has something wrong'.format(temp_path))
#                     Wrong_list.append(temp_path)
#                     continue
#                 else:
#                     writer.writerow(
#                         [*value_list[:6], freeE1, freeAB, freeWS, MaxComp2, phase_log, ksi_log, tau_log, 'cpu', bridge_ratio, *width, *surface, *freeAB_split])
#             elif which_type == 'gpu':
#                 popen4 = sp.Popen(cmd4, shell=True,
#                                   stdout=sp.PIPE, stderr=sp.PIPE)
#                 d = popen4.wait()
#                 line4 = popen4.stdout.readline()
#                 popen5 = sp.Popen(cmd5, shell=True,
#                                   stdout=sp.PIPE, stderr=sp.PIPE)
#                 f = popen5.wait()
#                 line5 = popen5.stdout.readline()
#                 all_num = lxlylz_re.findall(
#                     str(line4, encoding="UTF-8").strip())
#                 try:
#                     lx, ly, lz = list(map(float, all_num[-3:]))
#                 except ValueError as e:
#                     print(repr(e), " : ", temp_path)
#                     continue
#                 all_str = str(line5, encoding="UTF-8").strip().split(',')
#                 freeE1 = float(all_str[0])
#                 MaxComp2 = float(all_str[-1])
#                 if MaxComp2 > 1e-6:
#                     print('{} has something wrong'.format(temp_path))
#                     Wrong_list.append(temp_path)
#                     continue
#                 else:
#                     writer.writerow(
#                         [lx, ly, lz, 0, 0, 0, freeE1, MaxComp2, phase_log, ksi_log, tau_log, 'gpu'])


# if len(Wrong_list) >= 1:
#     rp1 = input('Do you want to re-Calculate these above?(yes/[no])') or "no"
#     if rp1 == 'yes' or rp1 == 'y':
#         for i in Wrong_list:
#             os.chdir(i)
#             _ = sp.Popen(cpuJob, shell=True, stdout=sp.PIPE)
#         print('Repush finished!')
#     elif rp1 == 'no' or rp1 == 'n':
#         rp2 = input('Do you Delete these above?(yes/[no])') or "no"
#         if rp2 == 'yes' or rp2 == 'y':
#             for i in Wrong_list:
#                 _ = sp.Popen('rm -r ' + i, shell=True, stdout=sp.PIPE)
#             print('Delete finished!')
# else:
#     print('Everything is ok~ Have a good time!')
