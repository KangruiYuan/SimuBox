import os
import subprocess
from pathlib import Path
import platform

if __name__ == "__main__":

    abs_current_dir = Path(__file__).parent.absolute()
    json_files = (abs_current_dir / "JSON").iterdir()

    alias_list = [
        'alias web="python -m SimuBox.script.web"',
        'alias ext="python -m SimuBox.script.extractor"',
        'alias pj="ls | grep push_job | xargs -i python3 {}"',
        'alias pjs="ls | grep push_job | xargs -i python3 {} >aa.txt 2>&1 &"',
        f'alias prep="cp {abs_current_dir / "push_job.py"}" ./',
        # f'alias prep="cp {abs_current_dir / "extractor.py"} ./;cp {abs_current_dir / "push_job.py"}"',
    ]

    for json_file in json_files:
        name = json_file.stem.lower()
        alias_list.append(f'alias {name}="cp {json_file.absolute()} ./input.json"')
        alias_list.append(f'alias prep_{name}="prep;{name}"')


    if platform.system() == "Linux":
        bashrc_path = os.path.expanduser("~/.bashrc")
        with open(bashrc_path, "r") as bashrc_file:
            cont = bashrc_file.read()

        with open(bashrc_path, "a") as bashrc_append_file:
            for element in alias_list:
                # 检查.bashrc是否已经包含该alias，如果不存在则添加
                if element not in cont:
                    bashrc_append_file.write(f"{element}\n")
                    print(f"{element} added to .bashrc")
                else:
                    print(f"{element} already exists in .bashrc")

        # 执行 source ~/.bashrc 命令
        subprocess.run("source ~/.bashrc", shell=True)
    else:
        for element in alias_list:
            print(element)
