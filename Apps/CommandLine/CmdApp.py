import tkinter as tk
from tkinter import filedialog
from SimuBox import read_density, iso2D, iso3D
import pyfiglet


def get_directory_path():

    # 弹出文件夹选择对话框
    folder_path = filedialog.askdirectory()
    # 打印所选路径
    if folder_path:
        print("选择的路径是:", folder_path)
    else:
        print("用户取消了选择。")
    return folder_path


def print_help():
    description = [
        "欢迎使用SimuBox。",
        "-" * 20,
        "(1) 绘制相结构(2D)",
        "(2) 绘制相结构(3D)",
        "-" * 20,
        "请输入相应功能序号：",
    ]
    for desc in description:
        print(desc)

def plot_iso2D():
    folder_path = get_directory_path()
    if folder_path:
        density = read_density(folder_path)
        print(f"当前可绘制的图像列号为: {list(range(len(density.data.columns)))}")
        targets = list(map(int, input("请输入绘制目标为（以空格间隔）：").split(" ")))
        permute = list(range(len(density.shape)))
        print(f"当前可调换的坐标轴为: {permute}")
        try:
            permute = list(map(int, input("请输入调换后的坐标轴顺序（以空格间隔，默认为原始序列）：").split(" ")))
        except:
            pass
        iso2D(density, target=targets, permute=permute)

def plot_iso3D():
    folder_path = get_directory_path()
    if folder_path:
        density = read_density(folder_path)
        print(f"当前可绘制的图像列号为: {list(range(len(density.data.columns)))}")
        targets = int(input("请输入绘制目标为（仅能输入单列序号）："))
        permute = list(range(len(density.shape)))
        print(f"当前可调换的坐标轴为: {permute}")
        try:
            permute = list(map(int, input("请输入调换后的坐标轴顺序（以空格间隔，默认为原始序列）：").split(" ")))
        except:
            pass
        iso3D(density, target=targets, permute=permute)


if __name__ == "__main__":
    root = tk.Tk()
    root.withdraw()
    # 将弹窗设置为最顶层窗口
    root.attributes("-topmost", 1)
    print(pyfiglet.figlet_format("S i m u", font="alligator"))
    print(pyfiglet.figlet_format("B o x", font="alligator"))
    while True:
        print_help()
        cmd = input()
        try:
            if cmd == "x":
                print("程序退出，祝您愉快。")
                break
            elif cmd == "1":
                plot_iso2D()
            elif cmd == "2":
                plot_iso3D()
            else:
                print("无效命令。")
        except:
            continue
