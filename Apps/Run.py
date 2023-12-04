from pathlib import Path
import argparse
import os
os.environ['NUMEXPR_MAX_THREADS'] = '16'

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="启动SimuBox的网页端服务")
    parser.add_argument("-a", "--app", type=str, default="主页.py", help="主页面文件")
    parser.add_argument("-p", "--port", type=int, default=9998)
    args = parser.parse_args()
    app_path = Path(__file__).parent / "Web" / args.app
    os.system(f"python -m streamlit run {app_path} --server.port {args.port}")