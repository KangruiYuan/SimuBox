
import os
import json
import subprocess as sp
import csv
import argparse
from collections import OrderedDict
import re

parser = argparse.ArgumentParser(description="Define the name of output file~")
parser.add_argument("-n", "--name", default='EXTRACTED_DATA',
                    type=str, help="the File name")
parser.add_argument("-d", "--dir", default=False, action='store_true')
parser.add_argument("-t", "--terminal", default='WORKING_DIR',
                    type=str, help="the Work Dir")
parser.add_argument("-p", "--phase", default='rand',
                    type=str, help="phase")
# parser.add_argument("-h", "--help", action="help", help="查看帮助信息")
args = parser.parse_args()


cmd1 = "tail ./printout.txt -n 1 -q | awk {'print $3'}"
cmd2 = "tail ./printout.txt -n 1 -q | awk {'print $8'}"
cmd3 = "tail -3 printout.txt | head -1"
cmd4 = "tail -2 printout.txt | head -1"
cmd5 = "tail -1 printout.txt | head -1"
cmd_lxlylz = "tail -3 printout.txt | head -1"
cmd_freeE_MaxC = "tail -1 printout.txt | head -1"
lxlylz_re = re.compile('[.0-9]+(?!e)')
freeE_re = re.compile('[.0-9e+-]+')

cpuJob = r"srun --partition=intel_2080ti,amd_3090,intel_Xeon --cpus-per-task=2 /home/share/TOPS2020/TOPS2020 -j -i=input.json >aa.txt 2>&1 &"
gpuJob = r"srun --partition=intel_2080,intel_2080ti,amd_3090 --nodes=1 --gpus=1 /home/share/scft2021 >aa.txt 2>&1 &"
gputops = r"srun --gpus=rtx_3090:1 --cpus-per-gpu=1 --partition=amd_3090 --gpus=1 -w gpu04 /home/share/TOPS2020/TOPS_device -j -i=input.json > aa.txt 2>&1 &"

main_dir = os.getcwd()
main_dir = os.path.join(main_dir, args.terminal)
sub_dirs = os.listdir(main_dir)
sub_dirs = sorted(sub_dirs)
Wrong_list = list()
if not args.dir:
    if '.csv' not in args.name:
        extract_name =  args.name.strip('/') + '.csv'
    else:
        extract_name =  args.name
else:
    temp_name = os.getcwd()
    temp_name = temp_name.split('/')[-1]
    extract_name = temp_name + '.csv'


def readJson(path=None):
    try:
        with open("./input.json", mode='r') as fp:
            json_base = json.load(fp, object_pairs_hook=OrderedDict)
            return json_base
    except json.decoder.JSONDecodeError as e:
        print(repr(e), " : ", path)
        return None
    
with open(os.path.join('CSV_Col/', extract_name), 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    flag = True
    for subdir in sub_dirs:
        temp_path = os.path.join(main_dir, subdir)
        if os.path.isdir(temp_path):
            os.chdir(temp_path)
            if not os.path.isfile('printout.txt'):
                continue
            json_base = readJson(temp_path)
            if json_base is None:
                continue

            if flag:
                writer.writerow(['lx', 'ly', 'lz', 'alpha', 'beta', 'gamma',
                                 'freeE', 'MaxComp',
                                 *json_base['Scripts'].keys(), 'path'])
                flag = False

            scripts_vals = json_base['Scripts'].values()
            which_type = json_base['Scripts'].get('cal_type', 'gpu')

            if which_type == 'cpu':
                
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
                    print('Wrong:', temp_path)
                    continue
                
                CompMax = uws[-1]
                freeE = uws[2]
                
                if json_base['Iteration']['IncompressibilityTarget'] < CompMax or freeE < 0:
                    print('{} WRONG'.format(temp_path))
                    Wrong_list.append(temp_path)
                    continue
            elif which_type == 'gpu':
                try:
                    with open('fet.dat') as fet:
                        cont = fet.readlines()
                except:
                    print('{} WRONG'.format(temp_path))
                    Wrong_list.append(temp_path)
                    continue
                cont = [i.strip().split(' ')[1:] for i in cont if i.split(' ')[1:]]
                # dataDict = dict([line[0]: float(line[1]) if (len(line) == 2 and line[1] != 'nan') else line[0]:line[1] for line in dataDict])
                cont = {line[0]: float(line[1])
                            for line in cont if len(line) == 2}
                
                CompMax = cont["inCompMax"]
                freeE = cont["freeEnergy"]
                lxlylz = [cont['lx'], cont['ly'], cont['lz']] + [0]*3
                
                if json_base['Iteration']['IncompressibilityTarget'] < CompMax or cont['lx'] == 'nan':
                    print('{} has something wrong'.format(temp_path))
                    Wrong_list.append(temp_path)
                    continue
        
        writer.writerow([*lxlylz, freeE, CompMax, *scripts_vals, temp_path])


if len(Wrong_list) >= 1:
    rp1 = input('Do you want to re-Calculate these above?(yes/[no])') or "no"
    if rp1 == 'yes' or rp1 == 'y':
        for i in Wrong_list:
            os.chdir(i)
            _ = sp.Popen(cpuJob, shell=True, stdout=sp.PIPE)
        print('Repush finished!')
    elif rp1 == 'no' or rp1 == 'n':
        rp2 = input('Do you Delete these above?(yes/[no])') or "no"
        if rp2 == 'yes' or rp2 == 'y':
            for i in Wrong_list:
                _ = sp.Popen('rm -r ' + i, shell=True, stdout=sp.PIPE)
            print('Delete finished!')
else:
    print('Everything is ok~ Have a good time!')
