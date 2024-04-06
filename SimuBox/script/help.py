
from pathlib import Path

if __name__ == "__main__":

    abs_current_dir = Path(__file__).parent.absolute()
    width = 40
    print("使用/拓展本库功能")
    print("-" * width)
    print("添加不同聚合物体系的抽象对象：")
    print(abs_current_dir.parent / "runner" / "agents")
    print("-" * width)
    print("批量化递交任务的脚本：")
    print(abs_current_dir / "push_job.py")
    print("-" * width)
    print("处理计算结果（收集/删除/重投/续跑）的脚本：")
    print(abs_current_dir / "extractor.py")
    print("-" * width)
    print("启动网页端服务的脚本：")
    print(abs_current_dir / "web.py")
    print("-" * width)
    print("向环境变量中添加快捷命令的脚本：")
    print(abs_current_dir / "setenv.py")