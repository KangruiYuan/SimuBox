import tempfile
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
from SimuBox import (
    check_state,
    add_logo,
    read_density,
    read_printout,
    ColorMode,
    WeightedMode,
    AnalyzeMode,
    VoronoiCell,
)

warnings.filterwarnings("ignore")

st.set_page_config(layout="wide")
st.title(":blue[SimuBox] :red[Visual] : Voronoi图绘制")
check_state()
add_logo()

with st.expander("Voronoi图绘制使用说明"):
    st.markdown(
        """
        ### Voronoi图绘制
        - 通用文件类型：phout.txt, block.txt, joint.txt 或者其他符合相同格式的文本文件
        - 参数类型：
            - target: 对密度的第几列进行剖分，仅能选择一列。
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

    expand = multi_select_cols[3].number_input(label="延拓信息", value=5, min_value=3)


plot_col, info_col = st.columns([0.6, 0.4])
main_mode = info_col.selectbox(
    "请选择分析模式", options=AnalyzeMode.values(), index=0, key="main mode"
)
info_col.divider()

if uploaded_phout and main_mode != AnalyzeMode.WEIGHTED:

    with info_col:
        point_color = st.text_input("请输入节点颜色", value="b")
        line_color = st.text_input("请输入边线颜色", value="k")
        point_xlim = st.slider(
            "节点横向范围",
            min_value=-expand / 2 + 0.5,
            max_value=expand / 2 + 0.5,
            value=(-0.1, 1.1),
            key="point_xlim",
        )
        point_ylim = st.slider(
            "节点纵向范围",
            min_value=-expand / 2 + 0.5,
            max_value=expand / 2 + 0.5,
            value=(-0.1, 1.1),
            key="point_ylim",
        )
        axis_xlim = st.slider(
            "图幅横向范围",
            min_value=-expand / 2 + 0.5,
            max_value=expand / 2 + 0.5,
            value=(-0.5, 1.5),
            key="axis_xlim",
        )
        axis_ylim = st.slider(
            "图幅纵向范围",
            min_value=-expand / 2 + 0.5,
            max_value=expand / 2 + 0.5,
            value=(-0.5, 1.5),
            key="axis_ylim",
        )
    with plot_col:
        if len(density.shape) == 3 and slices is None:
            st.warning("现无法支持三维剖分，请选择二维密度文件或输入切片信息")
        else:
            vc = VoronoiCell.Analyze(
                density,
                mode=main_mode,
                pc=point_color,
                lc=line_color,
                target=targets,
                expand=expand,
                slices=slices,
                permute=permute,
                save=st.session_state.save_auto,
                dpi=st.session_state.dpi,
                interactive=False,
                point_xlim=point_xlim,
                point_ylim=point_ylim,
                axis_xlim=axis_xlim,
                axis_ylim=axis_ylim,
                # figsize=(4, 4),
            )
            st.pyplot(vc.fig)

elif main_mode == AnalyzeMode.WEIGHTED:
    with info_col:
        sub_mode = st.selectbox(
            "请选择加权方式", options=WeightedMode.values(), index=0, key="sub_mode"
        )
        color_mode = st.selectbox(
            "请选择颜色模式",
            options=[ColorMode.L.value, ColorMode.RGB.value],
            index=0,
            key="color_mode",
        )
        linear = st.toggle("颜色线性变化 (目前仅对灰度图有效)", value=True)
        init_weight = 100 if sub_mode == WeightedMode.additive else 1e4
        df = pd.DataFrame(
            [
                {"x": 100, "y": 350, "weight": init_weight},
                {"x": 100, "y": 100, "weight": 0},
                {"x": 350, "y": 350, "weight": 0},
                {"x": 350, "y": 100, "weight": 0},
            ]
        )
        edited_df = st.data_editor(
            df,
            num_rows="dynamic",
            hide_index=False,
            use_container_width=False,
            width=290,
        )
        sub_size_cols = st.columns(2)

        width = sub_size_cols[0].number_input(
            "图幅宽度", value=500, min_value=edited_df["x"].max() + 10
        )
        height = sub_size_cols[1].number_input(
            "图幅高度", value=500, min_value=edited_df["y"].max() + 10
        )

        plot = st.button("开始绘制", use_container_width=True)
    with plot_col:
        if plot:
            sub_plot_cols = st.columns([0.15, 0.7, 0.15])
            with st.spinner("In progress..."):
                fig, ax = VoronoiCell.weighted_voronoi_diagrams(
                    edited_df[["x", "y"]].values,
                    weights=edited_df["weight"].values,
                    plot="imshow",
                    method=sub_mode,
                    color_mode=color_mode,
                    linear=linear,
                    interactive=False,
                    size=(width, height),
                )
                sub_plot_cols[1].pyplot(fig)
