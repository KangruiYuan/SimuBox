import tempfile
import warnings
from pathlib import Path

import numpy as np
import streamlit as st
import streamlit_nested_layout

from SimuBox import check_state, read_density, read_printout, iso2D, iso3D, add_logo

from stpyvista import stpyvista

warnings.filterwarnings("ignore")


st.set_page_config(layout="wide")
st.title(":blue[SimuBox] :red[Visual] : 结构图绘制")

check_state()
add_logo()

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

plot_col, file_upload_col = st.columns([0.7, 0.3])

with file_upload_col:
    uploaded_phout = st.file_uploader(
        "请选择密度类文件", accept_multiple_files=False, type=[".txt", ".bin"]
    )

    phout_sub_cols = st.columns(2)
    parse_N = phout_sub_cols[0].toggle("从密度解析格点数", value=True, key="parse_N")
    parse_L = phout_sub_cols[1].toggle("从密度解析周期", value=False, key="parse_L")

    lxlylz = st.text_input("请输入绘制结构的周期（空格间隔）", value="")
    lxlylz = np.array(list(map(float, lxlylz.split()))) if lxlylz else None
    use_lxlylz = st.toggle("使用手动输入的周期", value=False, key="use_lxlylz")

    uploaded_printout = st.file_uploader(
        "请选择TOPS标准输出文件", accept_multiple_files=False, type=[".txt"]
    )
    repair_from_printout = st.toggle(
            "从TOPS输出补充信息", value=False, key="repair_from_printout"
    )

    uploaded_fet = st.file_uploader(
        "请选择SCFT输出文件", accept_multiple_files=False, type=[".txt", ".dat"]
    )
    repair_from_fet = st.toggle(
            "从SCFT输出补充信息", value=False, key="repair_from_fet"
    )

with plot_col:

    if uploaded_phout:
        save = st.session_state.cache_dir / uploaded_phout.name
        if uploaded_phout.name.endswith(".txt"):
            with tempfile.NamedTemporaryFile(
                delete=False, mode="w+", encoding="utf-8", dir=st.session_state.cache_dir
            ) as temp_file:
                # 将 BytesIO 中的数据写入临时文件
                temp_file.write(uploaded_phout.read().decode("utf-8"))
                temp_name = Path(temp_file.name)
            density = read_density(temp_name, parse_N=parse_N, parse_L=parse_L)
            density.path = save
            temp_name.unlink()
        elif uploaded_phout.name.endswith(".bin"):
            binary_data = uploaded_phout.read()
            density = read_density(binary_data, parse_N=parse_N, parse_L=parse_L)
            density.path = save
            # density.data = density.data.div(density.data.sum(axis=0))
            # density.data = density.data.div(density.data.sum(axis=1), axis=0)

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



        sub_info_cols = st.columns(2)
        targets = sub_info_cols[0].multiselect(
            label="请选择需要绘制的列号", options=list(range(density.data.shape[1])), default=[0]
        )
        targets = targets if targets else [0]
        base_permute = list(range(len(density.shape)))
        permute = sub_info_cols[1].text_input(
            label=f"请指定坐标轴{base_permute}的顺序，以空格为间隔", value=""
        )
        permute = np.array(list(map(int, permute.split()))) if permute else base_permute

        slices = sub_info_cols[0].text_input(label="请输入切片信息，格式为: index ais", value=None)
        slices = list(map(int, slices.split())) if slices else None

        right_sub_cols = sub_info_cols[1].columns(2)
        expand = right_sub_cols[0].number_input(
            label="延拓信息", value=1.0, min_value=1.0, max_value=3.0
        )
        colorbar = right_sub_cols[1].toggle("显示数值条", value=True, key="colorbar")

        if len(density.shape) == 2 or slices:
            sub_cols = st.columns(3)
            for _idx, target in enumerate(targets):
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
                sub_cols[_idx % 3].pyplot(fig, use_container_width=True)
        elif len(density.shape) == 3:

            sub_fig_cols = st.columns(4)
            levels = sub_fig_cols[0].number_input(
                "输入界面密度",
                min_value=0.0000,
                max_value=1.0000,
                value=0.5,
                step=0.00001,
                format="%.5f",
            )

            background_color = sub_fig_cols[1].text_input("请输入背景颜色", value="white")

            # sub_right_fig_cols = sub_fig_cols[2].columns(2)
            opacity = sub_fig_cols[2].number_input(
                "请输入透明度（0-1）", value=0.8, min_value=0.0, max_value=1.0
            )
            frame = sub_fig_cols[3].toggle("是否需要外部框体", value=True)

            front_color = st.multiselect(
                    "请输入物体颜色",
                    options=("blue", "red", "green", "yellow"),
                    default=("blue", "red"),
            )
            plotter = iso3D(
                density,
                level=levels,
                target=targets,
                backend="vista",
                interactive=False,
                permute=permute,
                bk=background_color,
                colors=front_color,
                opacity=opacity,
                frame=frame,
                style="surface",
                expand=expand,
                save=save.parent
                / ("_".join(["iso3d", save.stem, str(targets[0])]) + ".svg"),
            )
            stpyvista(plotter)
