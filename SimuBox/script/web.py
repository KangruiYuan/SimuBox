from pathlib import Path
import argparse
import os
import platform


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="启动SimuBox的网页端服务")
    parser.add_argument("-a", "--app", type=str, default="Homepage.py", help="主页面文件")
    parser.add_argument("-p", "--port", type=int, default=9998)
    parser.add_argument(
        "-l",
        "--log_level",
        type=str,
        default="error",
        choices=["error", "warning", "info", "debug"],
    )
    args = parser.parse_args()
    app_path = Path(__file__).parents[1] / "web" / args.app

    if platform.system() == "Windows":
        os.environ["NUMEXPR_MAX_THREADS"] = "16"
    else:
        os.environ["NUMEXPR_MAX_THREADS"] = "4"
    os.system(f"python -m streamlit run {app_path} --server.port {args.port}")
