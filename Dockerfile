FROM python:3.11
LABEL authors="Alkaid"


COPY SimuBox ./SimuBox
COPY README.md .
COPY pyproject.toml .
COPY requirements.txt .
COPY LICENSE .

RUN pip config set global.index-url https://mirrors.aliyun.com/pypi/simple/
RUN pip install --no-cache-dir -r requirements.txt
# RUN rm -rf /root/.cache

ENTRYPOINT ["python3"]

CMD ["-m", "streamlit", "run", "./SimuBox/Web/主页.py"]