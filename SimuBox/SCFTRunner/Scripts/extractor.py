import argparse
from argparse import Namespace
import csv
import json
import os
import re
import subprocess as sp
from collections import OrderedDict, defaultdict
from itertools import combinations
from pathlib import Path
from typing import Any, Union, Sequence
from tqdm import tqdm

from push_job_TOPS import opts


def check_files(files: list[Union[str, Path]]):
    for file in files:
        if not os.path.isfile(file):
            print(f"Absent: {file}".center(50, "*"))
            return False
    return True


def check_result(res: dict):
    if res["phase"] == "C4" and round(res["ly"] / res["lz"], 3) != 1.0:
        res["phase"] = "Crect"
    elif res["phase"] == "Crect" and round(res["ly"] / res["lz"], 3) == 1.0:
        res["phase"] = "C4"

    if round(res["freeE_dis"] - res["freeE"], 3) == 0:
        res["phase"] = "Disorder"

    if (res["server"] == "cpu" and res["target_comp"] < res["CompMax"]) or (
        res["server"] == "gpu" and res["target_comp"] * 1e3 < res["CompMax"]
    ):
        print("CompMax not satisfied")
        return False
    else:
        return True


def stats_component(inputs: dict):
    total_stats = {}

    comp_stats = defaultdict(float)
    for i in inputs["Block"]:
        comp_stats[i["ComponentName"]] += i["ContourLength"]

    xN_stats = {}
    for i in inputs["Component"]["FloryHugginsInteraction"]:
        xN_stats[f"x{i['FirstComponentName']}{i['SecondComponentName']}N"] = i[
            "FloryHugginsParameter"
        ]
        xN_stats[f"x{i['SecondComponentName']}{i['FirstComponentName']}N"] = i[
            "FloryHugginsParameter"
        ]

    freeU_stats = {}
    for kind1, kind2 in combinations(comp_stats.keys(), 2):
        freeU_stats[f"freeU{kind1}{kind2}"] = (
            comp_stats[kind1] * comp_stats[kind2] * xN_stats[f"x{kind1}{kind2}N"]
        )

    total_stats.update(comp_stats)
    total_stats.update(xN_stats)
    total_stats["freeE_dis"] = sum(freeU_stats.values())
    return total_stats


def collect(
    subdirectories: list[Path],
    args: Namespace,
):
    """
    :param subdirectories: 需要进行信息收集
    :param args: 命令行参数，用于操控信息收集的行为
    :return:
        datas
            每个文件夹对应的数据信息
        columns
            数据表的列名
        missions_list
            需要进行下一步处理的数据信息

    """
    missions_list = list()
    columns: set[str] = set()
    datas: list[dict[str, Any]] = []
    dsp = "COLLECTING" if args.all else "WRONG"

    for subdir in tqdm(subdirectories):
        os.chdir(subdir)
        if not check_files(files=["printout.txt", opts.json_name]):
            continue
        with open(opts.json_name, mode="r") as fp:
            json_data = json.load(fp, object_pairs_hook=OrderedDict)

        data = {}
        scripts = json_data["Scripts"]
        data.update(scripts)
        stats_res = stats_component(json_data)
        data.update(stats_res)
        server_before = json_data["Scripts"].get("cal_type", "cpu")
        data["server"] = server_before
        data["target_comp"] = json_data["Iteration"]["IncompressibilityTarget"]
        if server_before == "cpu":
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

            popen_lxlylz = sp.Popen(
                cmd_lxlylz, shell=True, stdout=sp.PIPE, stderr=sp.PIPE
            )
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
                    for _k, _v in zip(
                        ["lx", "ly", "lz", "alpha", "beta", "gamma"], lxlylz
                    )
                }
            )

        else:
            if not check_files(files=["fet.dat"]):
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
                print(f"{dsp}: {subdir}")
            missions_list.append(
                {
                    "path": subdir,
                    "lxlylz": [
                        data.get(n, 0)
                        for n in ["lx", "ly", "lz", "alpha", "beta", "gamma"]
                    ],
                    "step": max(data.get("step", 2000), 5000),
                    "server": server_before,
                }
            )
    return datas, columns, missions_list


def dispatch(missions: list):
    if len(missions) == 0:
        print("所有数据无显著错误，祝您生活愉快。")
        return
    mode = input("删除(d) / 续跑(c) / 重投(r) / 跳过(其他任意键)? (d/c/r/[pass])") or "pass"
    if mode == "c":
        for mission in missions:
            mission_folder = mission["path"]
            os.chdir(mission_folder)
            with open(opts.json_name, mode="r") as fp:
                json_base = json.load(fp, object_pairs_hook=OrderedDict)
            server = args.server or mission["server"]
            json_base["Scripts"]["cal_type"] = (
                server if os.path.isfile("phout.txt") else "cpu"
            )
            json_base["Iteration"]["MaxStep"] = mission["step"]
            json_base["Initializer"]["UnitCell"]["Length"] = mission["lxlylz"]
            if os.path.isfile("phout.txt"):
                sp.call("cp phout.txt phin.txt", shell=True, stdout=sp.PIPE)
                json_base["Initializer"]["Mode"] = "FILE"
                json_base["Initializer"]["FileInitializer"] = {
                    "Mode": "OMEGA",
                    "Path": "phin.txt",
                    "SkipLineNumber": 1 if mission["server"] == "cpu" else 2,
                }
                json_base["Iteration"]["AndersonMixing"]["Switch"] = "AUTO"

                with open(opts.json_name, "w") as f:
                    json.dump(json_base, f, indent=4, separators=(",", ": "))

            if json_base["Scripts"]["cal_type"] == "gpu":
                job = sp.Popen(opts.worker["gpuSCFT"], shell=True, stdout=sp.PIPE)
            else:
                job = sp.Popen(opts.worker["cpuTOPS"], shell=True, stdout=sp.PIPE)
            print(f"续跑任务({job.pid}):{mission_folder}")
    elif mode == "r":
        for mission in missions:
            mission_folder = mission["path"]
            os.chdir(mission_folder)
            server = args.server or mission["server"]
            if server == "gpu":
                job = sp.Popen(opts.worker["gpuSCFT"], shell=True, stdout=sp.PIPE)
            else:
                job = sp.Popen(opts.worker["cpuTOPS"], shell=True, stdout=sp.PIPE)
            print(f"重投任务({job.pid}):{mission_folder}")
    elif mode == "d":
        for mission in missions:
            mission_folder = str(mission["path"])
            sp.Popen("rm -r " + mission_folder, shell=True, stdout=sp.PIPE)
        print("全部有误文件夹均已删除。")


def write_to_csv(
    path: Union[Path, str], datas: list[dict[str, Any]], columns: Union[Sequence, set]
):
    with open(path, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        columns_list: list = sorted(list(columns))
        writer.writerow(columns_list)
        for data in datas:
            writer.writerow([data.get(c, 0) for c in columns_list])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-n", "--name", default="", type=str)
    parser.add_argument("-t", "--terminal", default="WORKING_DIR", type=str)
    parser.add_argument("-s", "--server", default="", type=str)
    parser.add_argument("-a", "--all", action="store_true", default=False)
    args = parser.parse_args()

    cmd_lxlylz = "tail -3 printout.txt | head -1"
    cmd_freeE_MaxC = "tail -1 printout.txt | head -1"
    lxlylz_re = re.compile("[.0-9]+(?!e)")
    freeE_re = re.compile("[.0-9e+-]+")

    parent_folder = Path.cwd()
    working_directory = parent_folder / args.terminal
    subdirectories = [item for item in working_directory.iterdir() if item.is_dir()]

    output_csv_name = args.name or parent_folder.name
    output_csv_path = parent_folder / output_csv_name
    if output_csv_path.suffix != ".csv":
        output_csv_path = str(output_csv_path) + ".csv"

    datas, columns, missions_list = collect(subdirectories, args)

    print("数据收集完成，写入中。")

    write_to_csv(output_csv_path, datas, columns)

    dispatch(missions_list)
