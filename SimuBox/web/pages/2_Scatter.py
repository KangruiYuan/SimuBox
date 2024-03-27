import tempfile
import warnings
from pathlib import Path

import numpy as np
import pyecharts.options as opts
import streamlit as st

from SimuBox import (
    read_density,
    read_printout,
    Scatter,
    SCATTER_PLOT_CONFIG,
    init_plot_config,
)

from pyecharts.charts import Line
from streamlit_echarts import st_pyecharts

warnings.filterwarnings("ignore")

st.set_page_config(layout="wide")
st.title(":blue[SimuBox] :red[Visual] : 散射图绘制")

init_plot_config(SCATTER_PLOT_CONFIG)



with st.expander("散射图绘制使用说明"):
    st.markdown(
        """
        ### 散射图绘制
        - 通用文件类型：phout.txt, block.txt, joint.txt 或者其他符合相同格式的文本文件
        - 参数类型：
            - target: 对密度的第几列进行散射，仅能选择一列。
            - permute: 调换后的坐标轴顺序。
            - expand: 向外延申的比例（原点为几何中心）。
            - slices: 切片信息(index, axis), 意为取axis轴上索引为index的片层
        """
    )

left_upload_col, middle_upload_col, right_upload_col = st.columns(3)
uploaded_phout = left_upload_col.file_uploader(
    "请选择密度类文件", accept_multiple_files=False, type=[".txt"]
)
uploaded_printout = middle_upload_col.file_uploader(
    "请选择TOPS标准输出文件", accept_multiple_files=False, type=[".txt"]
)
uploaded_fet = right_upload_col.file_uploader(
    "请选择SCFT输出文件", accept_multiple_files=False, type=[".txt", ".dat"]
)
st.divider()
left_sub_cols = left_upload_col.columns(2)
parse_N = left_sub_cols[0].toggle("从密度解析格点数", value=True, key="parse_N")
# colorbar = left_sub_cols[0].toggle("显示数值条", value=True, key="colorbar")

parse_L = left_sub_cols[1].toggle("从密度解析周期", value=False, key="parse_L")
use_lxlylz = left_sub_cols[1].toggle("使用手动输入的周期", value=False, key="use_lxlylz")

lxlylz = middle_upload_col.text_input("请输入绘制结构的周期（空格间隔）", value="")
lxlylz = np.array(list(map(float, lxlylz.split()))) if lxlylz else None

right_sub_cols = right_upload_col.columns(2)
repair_from_printout = right_sub_cols[0].toggle(
    "从TOPS输出补充信息", value=False, key="repair_from_printout"
)
repair_from_fet = right_sub_cols[1].toggle(
    "从SCFT输出补充信息", value=False, key="repair_from_fet"
)

if uploaded_phout:
    with tempfile.NamedTemporaryFile(
        delete=False, mode="w+", encoding="utf-8", dir=st.session_state.cache_dir
    ) as temp_file:
        # 将 BytesIO 中的数据写入临时文件
        temp_file.write(uploaded_phout.read().decode("utf-8"))
        temp_name = Path(temp_file.name)
        save = st.session_state.cache_dir / uploaded_phout.name
    density = read_density(temp_name, parse_N=parse_N, parse_L=parse_L)
    density.path = save
    temp_name.unlink()
    # targets = left_upload_col.text_input(label="请选择需要绘制的列号（从0开始，以空格间隔）", value="0")
    if use_lxlylz and lxlylz is not None:
        density.lxlylz = lxlylz
    elif repair_from_printout and uploaded_printout:
        with tempfile.NamedTemporaryFile(
            delete=False, mode="w+", encoding="utf-8", dir=st.session_state.cache_dir
        ) as temp_file:
            # 将 BytesIO 中的数据写入临时文件
            temp_file.write(uploaded_printout.read().decode("utf-8"))
            temp_name = Path(temp_file.name)
        printout = read_printout(temp_name)
        density.repair_from_printout(printout)
        temp_name.unlink()
    elif repair_from_fet and uploaded_fet:
        with tempfile.NamedTemporaryFile(
            delete=False, mode="w+", encoding="utf-8", dir=st.session_state.cache_dir
        ) as temp_file:
            # 将 BytesIO 中的数据写入临时文件
            temp_file.write(uploaded_fet.read().decode("utf-8"))
            temp_name = Path(temp_file.name)

    multi_select_cols = st.columns(4)
    targets = multi_select_cols[0].selectbox(
        label="请选择需要绘制的列号", options=list(range(density.data.shape[1])), index=0
    )
    targets = int(targets)

    base_permute = list(range(len(density.shape)))
    permute = multi_select_cols[1].text_input(
        label=f"请指定坐标轴{base_permute}的顺序，以空格为间隔", value=""
    )
    permute = np.array(list(map(int, permute.split()))) if permute else base_permute

    slices = multi_select_cols[2].text_input(label="请输入切片信息，格式为: index ais", value=None)
    slices = list(map(int, slices.split())) if slices else None

    expand = multi_select_cols[3].number_input(
        label="延拓信息", value=1, min_value=1, max_value=3
    )

    sc = Scatter.sacttering_peak(
        density, target=targets, expand=expand, slices=slices, permute=permute
    )

    plot_col, info_col = st.columns([0.75, 0.25])

    with info_col:
        width = st.number_input("高斯展开峰宽", value=0.5, min_value=0.0)
        step = st.number_input(
            "离散化参数（越大越光滑）", value=2000, min_value=1000, max_value=5000
        )
        cut_off = st.number_input("截断参数", value=300.0, min_value=0.0)
        min_height = st.number_input("最低峰高", value=1.0, min_value=0.0)
        interactive = st.toggle("启用交互式绘图", value=False)

    with plot_col:
        sc_plots = Scatter.show_peak(
            res=sc,
            width=width,
            step=step,
            cutoff=cut_off,
            min_height=min_height,
            save=st.session_state.save_auto,
            dpi=st.session_state.dpi,
        )
        if not interactive:
            st.pyplot(sc_plots.fig)
        else:
            c = (
                Line(
                    init_opts=opts.InitOpts(
                        # height="450px",
                    )
                )
                .add_xaxis(sc_plots.plot_x)
                .add_yaxis(
                    "散射峰",
                    sc_plots.plot_y,
                    is_symbol_show=True,
                    symbol=None,
                    symbol_size=1,
                    areastyle_opts=opts.AreaStyleOpts(opacity=0.5),
                )
                .set_global_opts(
                    title_opts=opts.TitleOpts(title="结构散射结果图"),
                    yaxis_opts=opts.AxisOpts(
                        name="Intensity",
                        name_location="middle",
                        name_gap=35,
                        name_textstyle_opts=opts.TextStyleOpts(
                            font_weight="bold", font_size=20
                        ),
                    ),
                    xaxis_opts=opts.AxisOpts(
                        name=r"q/Rg",
                        name_location="end",
                        name_gap=20,
                        name_textstyle_opts=opts.TextStyleOpts(
                            font_weight="bold", font_size=20
                        ),
                    ),
                    datazoom_opts=opts.DataZoomOpts(
                        type_="slider",
                        is_show=True,
                        # start_value=min(sc_plots.plot_x),
                        # end_value=max(sc_plots.plot_x),
                        range_start=0,
                        range_end=100,
                        orient="horizontal",
                        pos_bottom="15px",
                    ),
                )
                .set_series_opts(
                    label_opts=opts.LabelOpts(is_show=False),
                )
            )
            st_pyecharts(c, height="500px")

    peaks_loc = np.around(sc_plots.peaks_location, 3)
    peaks_loc_str = " : ".join([str(x) for x in peaks_loc])
    peaks_loc_2 = np.around(sc_plots.peaks_location**2).astype(int)
    peaks_loc_2_str = " : ".join([rf"\sqrt{str(x)}" for x in peaks_loc_2])

    info_col.markdown(
        f"""
    #### 峰位比值: \n
    ${peaks_loc_str}$ \n
    ${peaks_loc_2_str}$
    """
    )
