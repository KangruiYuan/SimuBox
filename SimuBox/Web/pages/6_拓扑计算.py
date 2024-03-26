import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import json
from SimuBox import (
    check_state,
    add_logo,
    TopoCreater,
    fA,
    fB,
    COMPARE_PLOT_CONFIG,
    init_plot_config,
    TopoCreateMode,
)


warnings.filterwarnings("ignore")

init_plot_config(COMPARE_PLOT_CONFIG)
st.set_page_config(layout="wide")
st.title(":blue[SimuBox] :red[Visual] : 拓扑计算")

check_state()
add_logo()


with st.expander("拓扑计算使用说明"):
    st.markdown(
        """
        ### 曲线比较
        - 通用文件类型：json
        - 参数类型
        """
    )


plot_col, param_col = st.columns([0.75, 0.25])
tc = TopoCreater(verbose=False)

with param_col:
    colors = st.multiselect(
        "请指定配色顺序",
        options=["b", "r", "g", "blueviolet", "cyan"],
        default=["b", "r", "g"],
    )
    curve_rad = st.number_input("输入弧度 (仅对曲线模式起效)", value=0.4, min_value=0.0)
    sub_cols = st.columns(2)

    with sub_cols[1]:
        odt = st.toggle("计算ODT", value=True)
        show_nodes = st.toggle("节点信息", value=True)
        x_lim_max = st.number_input("X轴截止值", value=0.9, min_value=0.0, max_value=1.0)
        y_lim_max = st.number_input("Y轴截止值", value=100.0)
        xlabel = st.text_input("X标签", value=r"$f_{A}$")

    with sub_cols[0]:
        curve = st.toggle("曲线形式", value=False)
        show_edge_labels = st.toggle("边信息", value=True)
        x_lim_min = st.number_input(
            "X轴起始值", value=0.1, min_value=0.0, max_value=x_lim_max - 0.1
        )
        y_lim_min = st.number_input(
            "Y轴起始值", value=0.0, min_value=0.0, max_value=y_lim_max
        )
        ylabel = st.text_input("Y标签", value=r"$\chi {\rm N}$")

    plot_button = st.button("绘制拓扑图/ODT", use_container_width=True)


with plot_col:
    select_mode = st.selectbox("请选择模型", options=TopoCreateMode.values(), index=0)
    if select_mode == TopoCreateMode.JSON:
        uploaded_json = st.file_uploader("请上传JSON文件", type=[".json"])
        if uploaded_json:
            content = json.load(uploaded_json)
            with st.expander("文件读取结果"):
                st.json(content)
        st.divider()
    elif select_mode != TopoCreateMode.DENDRIMER:
        sub_input_cols = st.columns([0.7, 0.3])
        with sub_input_cols[1]:
            arm = st.number_input("请输入臂数", value=1, min_value=1)
            st.caption("以上仅对star模式有效")
            check_button = st.button("校验数据合法性", use_container_width=True)

        with sub_input_cols[0]:
            data = pd.DataFrame(
                data=[["A", "fA"], ["B", "1 - fA"]], columns=["kind", "fraction"]
            )
            edit_data = st.data_editor(
                data,
                hide_index=True,
                num_rows="dynamic",
                width=int(1200 * 0.75 * 0.7),
                height=220,
            )
            infos = edit_data.dropna(how="any", axis=0)
            blocks = infos.kind.tolist()
            fractions = [eval(i) for i in infos.fraction]
            if check_button:
                if round(sum(fractions) * arm, 3) != 1:
                    sub_input_cols[1].warning("fraction 和不为1")
                else:
                    sub_input_cols[1].success("数据合法，请继续。")
    elif select_mode == TopoCreateMode.DENDRIMER:
        sub_dendri_cols = st.columns(4)
        A_block_layer = sub_dendri_cols[0].number_input(
            "A嵌段层数", value=1, min_value=0, key="A_block_layer"
        )
        A_block_branch = sub_dendri_cols[1].number_input(
            "A嵌段支化度", value=1, min_value=1, key="A_block_branch"
        )
        B_block_layer = sub_dendri_cols[2].number_input(
            "B嵌段层数", value=1, min_value=0, key="B_block_layer"
        )
        B_block_branch = sub_dendri_cols[3].number_input(
            "B嵌段支化度", value=1, min_value=1, key="B_block_branch"
        )
        fractions = st.text_input("嵌段分数，默认从第一个A嵌段开始，以英文逗号，其余位置不要使用空格", value="")
        fractions = list(map(eval, fractions.split(","))) if fractions else None

    if plot_button:
        if select_mode == TopoCreateMode.LINEAR:
            # print(list(tc.stats(blocks).elements()))
            tc.linear(blocks=blocks, fractions=fractions)
        elif select_mode == TopoCreateMode.JSON and uploaded_json:
            tc.fromJson(content)
        elif select_mode == TopoCreateMode.AMBN:
            tc.AmBn(blocks=blocks, fractions=fractions)
        elif select_mode == TopoCreateMode.STAR:
            tc.star(blocks=blocks, fractions=fractions, arm=arm)
        elif select_mode == TopoCreateMode.DENDRIMER:
            tc.dendrimer(
                A_block_layer=A_block_layer,
                B_block_layer=B_block_layer,
                A_branch=A_block_branch,
                B_branch=B_block_branch,
                fractions=fractions,
            )

        frac = 0.48
        sub_plot_cols = st.columns([frac, 1 - frac])
        topo_save_path = (
            st.session_state.cache_dir / f"{tc.type}.png"
            if st.session_state.save_auto
            else False
        )
        topo_plot = tc.show_topo(
            colors=colors,
            curve=curve,
            interactive=False,
            save=topo_save_path,
            show_nodes=show_nodes,
            show_edge_labels=show_edge_labels,
        )
        sub_plot_cols[0].pyplot(topo_plot.fig)
        if odt:
            odt_save_path = (
                st.session_state.cache_dir / f"{tc.type}.png"
                if st.session_state.save_auto
                else False
            )
            tc.RPA()
            odt_plot = tc.ODT(
                fs=np.linspace(x_lim_min, x_lim_max, 51),
                interactive=False,
                plot=True,
                figsize=(8, 8),
                xlabel=xlabel,
                ylabel=ylabel,
                x_lim=(max(x_lim_min - 0.1, 0.0), min(1.0, x_lim_max + 0.1)),
                y_lim=(y_lim_min, y_lim_max),
                save=odt_save_path,
            )
            if len(np.unique(odt_plot.xN)) != 1:
                odt_xN_idx = np.argmin(odt_plot.xN)
                sub_plot_cols[1].subheader(
                        r"$f_{\rm A}=$" + str(round(odt_plot.f[odt_xN_idx], 3))
                )
                sub_plot_cols[1].subheader(
                        r"$\chi N_{\rm ODT}=$" + str(round(odt_plot.xN[odt_xN_idx], 3))
                )
                sub_plot_cols[1].pyplot(odt_plot.fig)
            else:
                sub_plot_cols[1].subheader(
                    r"$\chi N_{\rm ODT}=$" + str(round(odt_plot.xN[0], 3))
                )
