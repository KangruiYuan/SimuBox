import argparse
import csv
import json
import os
import re
import subprocess as sp
from collections import ChainMap, OrderedDict, defaultdict
from itertools import combinations
from pathlib import Path
from typing import Any

from push_job_TOPS import opts

parser = argparse.ArgumentParser()
parser.add_argument("-n", "--name", default="Default", type=str)
parser.add_argument("-t", "--terminal", default="WORKING_DIR", type=str)
parser.add_argument("-w", "--where", default="", type=str)
parser.add_argument("-a", "--all", action="store_true", default=False)
args = parser.parse_args()

cmd_lxlylz = "tail -3 printout.txt | head -1"
cmd_freeE_MaxC = "tail -1 printout.txt | head -1"
lxlylz_re = re.compile("[.0-9]+(?!e)")
freeE_re = re.compile("[.0-9e+-]+")

parent_folder = Path.cwd()
working_directory = parent_folder / args.terminal
subdirectories = [item for item in working_directory.iterdir() if item.is_dir()]


if args.name == "Default":
    extract_name = parent_folder / parent_folder.name
    extract_name = str(extract_name) + ".csv"
else:
    extract_name = Path.cwd() / args.name
    if not extract_name.name.endswith(".csv"):
        extract_name = str(extract_name) + ".csv"


def check_files(files: list[str, Path]):
    for file in files:
        if not os.path.isfile(file):
            print(f"Absent: {file}")
            return False
    return True


def check_result(res: dict):
    if res["phase"] == "C4" and round(res["ly"] / res["lz"], 3) != 1.0:
        res["phase"] = "Crect"
    elif res["phase"] == "Crect" and round(res["ly"] / res["lz"], 3) == 1.0:
        res["phase"] = "C4"

    if round(res["freeE_dis"] - res["freeE"], 3) == 0:
        res["phase"] = "Disorder"

    if (res["which_type"] == "cpu" and res["target_comp"] < res["CompMax"]) or (
        res["which_type"] == "gpu" and res["target_comp"] * 1e3 < res["CompMax"]
    ):
        print("CompMax not satisfied")
        return False
    else:
        return True


def stats_component(json_dict: dict):
    comp_stats = defaultdict(float)
    for i in json_dict["Block"]:
        comp_stats[i["ComponentName"]] += i["ContourLength"]

    xN_stats = {}
    for i in json_dict["Component"]["FloryHugginsInteraction"]:
        xN_stats[f"x{i['FirstComponentName']}{i['SecondComponentName']}N"] = i[
            "FloryHugginsParameter"
        ]
        xN_stats[f"x{i['SecondComponentName']}{i['FirstComponentName']}N"] = i[
            "FloryHugginsParameter"
        ]

    freeUComp = {}
    for one, two in combinations(comp_stats.keys(), 2):
        freeUComp[f"freeU{one}{two}"] = (
            comp_stats[one] * comp_stats[two] * xN_stats[f"x{one}{two}N"]
        )

    total_stats = {}
    total_stats.update(comp_stats)
    total_stats.update(xN_stats)
    # total_stats.update(freeUComp)
    total_stats["freeE_dis"] = sum(freeUComp.values())
    return total_stats


columns: set[str] = set()
datas: list[dict[str, Any]] = []
wrong_list = list()

desp = "REPUSH" if args.all else "WRONG"

for subdir in subdirectories:
    os.chdir(subdir)
    if not check_files(files=["printout.txt", opts.json_name]):
        wrong_list.append({"path": subdir})
        print(str(subdir).center(50, "*"))
        continue
    with open(opts.json_name, mode="r") as fp:
        json_data = json.load(fp, object_pairs_hook=OrderedDict)

    stats_res = stats_component(json_data)

    scripts = json_data["Scripts"]
    which_type = json_data["Scripts"].get("cal_type", "cpu")
    target_comp = json_data["Iteration"]["IncompressibilityTarget"]
    data = {}
    data.update(scripts)
    data.update(stats_res)
    data["which_type"] = which_type
    data["target_comp"] = target_comp
    if which_type == "cpu":
        popen_freeE_MaxC = sp.Popen(
            cmd_freeE_MaxC, shell=True, stdout=sp.PIPE, stderr=sp.PIPE
        )
        popen_freeE_MaxC.wait()
        line_freeE_MaxC = popen_freeE_MaxC.stdout.readline()

        uws = list(
            map(
                float,
                freeE_re.findall(str(line_freeE_MaxC, encoding="UTF-8").strip()),
            )
        )

        data.update(
            dict(
                step=int(uws[0]),
                freeE=uws[1],
                freeU=uws[2],
                freeWS=uws[3] + uws[4],
                CompMax=uws[-1],
            )
        )

        popen_lxlylz = sp.Popen(cmd_lxlylz, shell=True, stdout=sp.PIPE, stderr=sp.PIPE)
        popen_lxlylz.wait()
        line_lxlylz = popen_lxlylz.stdout.readline()

        lxlylz = list(
            map(
                lambda x: round(float(x), 6),
                str(line_lxlylz, encoding="UTF-8").strip().split(" "),
            )
        )
        data.update(
            {
                _k: _v
                for _k, _v in zip(["lx", "ly", "lz", "alpha", "beta", "gamma"], lxlylz)
            }
        )

    else:
        if not check_files(files=["fet.dat"]):
            print(str(subdir).center(50, "*"))
            continue
        cont = open("fet.dat", "r").readlines()
        cont = [i.strip().split(" ")[1:] for i in cont if i.split(" ")[1:]]
        cont = {line[0]: float(line[1]) for line in cont if len(line) == 2}

        data.update(
            dict(
                freeE=cont["freeEnergy"],
                freeU=cont["freeAB_sum"],
                freeWS=cont["freeW"] + cont["freeS"],
                CompMax=cont["inCompMax"],
            )
        )
        data.update(cont)
    columns = columns.union(set(data.keys()))

    if check_result(data) and not args.all:
        datas.append(data.copy())
    else:
        if not args.all:
            print(f"{desp}: {subdir}")
        wrong_list.append(
            {
                "path": subdir,
                "lxlylz": [
                    data.get(n, 0) for n in ["lx", "ly", "lz", "alpha", "beta", "gamma"]
                ],
                "step": data.get("step", 2000),
                "type": which_type,
            }
        )

with open(extract_name, "w", newline="") as csvfile:
    writer = csv.writer(csvfile)
    columns_list: list = sorted(list(columns))
    writer.writerow(columns_list)
    for data in datas:
        writer.writerow([data.get(c, 0) for c in columns_list])


def RepushOrDelete(wrongList):
    if len(wrongList) == 0:
        print("Everything is ok~ Have a good time!")
        return
    mode = input("Delete(d) / Continue(c) / Repush(r)?(d/c/r/[pass])") or "pass"
    if mode == "c":
        for i in wrongList:
            tmp_path = i["path"]
            os.chdir(tmp_path)
            with open(opts.json_name, mode="r") as fp:
                json_base = json.load(fp, object_pairs_hook=OrderedDict)
            new_type = args.where or i["type"]
            json_base["Scripts"]["cal_type"] = (
                new_type if os.path.isfile("phout.txt") else "cpu"
            )
            json_base["Iteration"]["MaxStep"] = i["step"]
            json_base["Initializer"]["UnitCell"]["Length"] = i["lxlylz"]
            if os.path.isfile("phout.txt"):
                sp.call("cp phout.txt phin.txt", shell=True, stdout=sp.PIPE)
                json_base["Initializer"]["Mode"] = "FILE"
                json_base["Initializer"]["FileInitializer"] = {
                    "Mode": "OMEGA",
                    "Path": "phin.txt",
                    "SkipLineNumber": 1 if i["type"] == "cpu" else 2,
                }
                json_base["Iteration"]["AndersonMixing"]["Switch"] = "AUTO"

                with open(opts.json_name, "w") as f:
                    json.dump(json_base, f, indent=4, separators=(",", ": "))

            if json_base["Scripts"]["cal_type"] == "gpu":
                job = sp.Popen(opts.worker["gpuSCFT"], shell=True, stdout=sp.PIPE)
            else:
                job = sp.Popen(opts.worker["cpuTOPS"], shell=True, stdout=sp.PIPE)
            print("{pid}:{path}".format(pid=job.pid, path=tmp_path))
    elif mode == "r":
        for i in wrongList:
            tmp_path = i["path"]
            os.chdir(tmp_path)
            new_type = args.where or i["type"]
            if new_type == "gpu":
                job = sp.Popen(opts.worker["gpuSCFT"], shell=True, stdout=sp.PIPE)
            else:
                job = sp.Popen(opts.worker["cpuTOPS"], shell=True, stdout=sp.PIPE)
            print("{pid}:{path}".format(pid=job.pid, path=tmp_path))
    elif mode == "d":
        for i in wrongList:
            tmp_path = str(i["path"])
            sp.Popen("rm -r " + tmp_path, shell=True, stdout=sp.PIPE)
        print("Delete finished!")


RepushOrDelete(wrong_list)
