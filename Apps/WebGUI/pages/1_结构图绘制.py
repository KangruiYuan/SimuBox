import tempfile
import warnings
from pathlib import Path

import numpy as np
import streamlit as st
from SimuBox import read_density, read_printout, iso2D, iso3D, check_state
from stpyvista import stpyvista

warnings.filterwarnings("ignore")


st.set_page_config(layout="wide")
st.title(":blue[SimuBox] :red[Visual] : 结构图绘制")

check_state(Path(__file__).parents[1])

with st.expander("结构图绘制使用说明"):
    st.markdown(
        """
            ### 密度文件可视化
            - 通用文件类型：phout.txt, block.txt, joint.txt 或者其他符合相同格式的文本文件
            - 参数类型：
                - target: 绘制图像的第几列（从0开始编号），可以选择多列。
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
parse_N = left_sub_cols[0].checkbox("从密度解析格点数", value=True, key="parse_N")
colorbar = left_sub_cols[0].checkbox("显示数值条", value=True, key="colorbar")

parse_L = left_sub_cols[1].checkbox("从密度解析周期", value=False, key="parse_L")
use_lxlylz = left_sub_cols[1].checkbox("使用手动输入的周期", value=False, key="use_lxlylz")

lxlylz = middle_upload_col.text_input("请输入绘制结构的周期（空格间隔）", value="")
lxlylz = np.array(list(map(float, lxlylz.split()))) if lxlylz else None

right_sub_cols = right_upload_col.columns(2)
repair_from_printout = right_sub_cols[0].checkbox(
    "从TOPS标准输出补充信息", value=False, key="repair_from_printout"
)
repair_from_fet = right_sub_cols[1].checkbox(
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
    targets = multi_select_cols[0].multiselect(
        label="请选择需要绘制的列号", options=list(range(density.data.shape[1])), default=[0]
    )
    targets = targets if targets else [0]
    base_permute = list(range(len(density.shape)))
    permute = multi_select_cols[1].text_input(
        label=f"请指定坐标轴{base_permute}的顺序，以空格为间隔", value=""
    )
    permute = np.array(list(map(int, permute.split()))) if permute else base_permute

    slices = multi_select_cols[2].text_input(label="请输入切片信息，格式为: index ais", value=None)
    slices = list(map(int, slices.split())) if slices else None

    expand = multi_select_cols[3].number_input(
        label="延拓信息", value=1.0, min_value=1.0, max_value=3.0
    )

    if len(density.shape) == 2 or slices:
        if len(targets) == 1:
            sub_cols = st.columns(3)
            fig, axes = iso2D(
                density,
                target=targets,
                permute=permute,
                slices=slices,
                colorbar=colorbar,
                expand=expand,
                save=st.session_state.save_auto,
                dpi=st.session_state.dpi,
            )
            sub_cols[1].pyplot(fig, use_container_width=True)
        elif len(targets) == 2:
            sub_cols = st.columns(4)
            for target, sub_col in zip(targets, sub_cols[1:3]):
                fig, axes = iso2D(
                    density,
                    target=target,
                    permute=permute,
                    slices=slices,
                    colorbar=colorbar,
                    expand=expand,
                    save=st.session_state.save_auto,
                    dpi=st.session_state.dpi,
                )
                sub_col.pyplot(fig, use_container_width=True)
        else:
            sub_cols = st.columns(len(targets))
            for target, sub_col in zip(targets, sub_cols):
                fig, axes = iso2D(
                    density,
                    target=target,
                    permute=permute,
                    slices=slices,
                    colorbar=colorbar,
                    expand=expand,
                    save=st.session_state.save_auto,
                    dpi=st.session_state.dpi,
                )
                sub_col.pyplot(fig, use_container_width=True)
    elif len(density.shape) == 3:
        sub_cols = st.columns(spec=[0.75, 0.25])
        with sub_cols[1]:
            front_color = st.multiselect(
                "请输入物体颜色",
                options=("blue", "red", "green", "yellow"),
                default=("blue", "red"),
            )
            background_color = st.text_input("请输入背景颜色", value="white")
            opacity = st.number_input(
                "请输入透明度（0-1）", value=0.8, min_value=0.0, max_value=1.0
            )
            frame = st.checkbox("是否需要外部框体", value=True)
            style = st.selectbox(
                "选择绘图模式",
                options=["surface", "wireframe", "points", "points_gaussian"],
                index=0,
            )

        with sub_cols[0]:
            plotter = iso3D(
                density,
                target=targets,
                backend="vista",
                interactive=False,
                permute=permute,
                bk=background_color,
                colors=front_color,
                opacity=opacity,
                frame=frame,
                style=style,
                expand=expand,
                save=save.parent
                / ("_".join(["iso3d", save.stem, str(targets[0])]) + ".svg"),
            )
            stpyvista(plotter)
