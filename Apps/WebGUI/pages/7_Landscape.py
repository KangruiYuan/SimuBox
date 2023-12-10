import warnings

import pandas as pd
import streamlit as st
from SimuBox import Landscaper, LAND_PLOT_CONFIG, init_plot_config

warnings.filterwarnings("ignore")

init_plot_config(LAND_PLOT_CONFIG)

st.set_page_config(layout="wide")
st.title(":blue[SimuBox] :red[Visual] : Landscape")

with st.expander("Landscape绘制使用说明"):
    st.markdown(
        """
        ### 曲线比较
        - 通用文件类型：csv, xlsx
        - 参数类型
        """
    )

upload_col, param_col = st.columns([0.75, 0.25])
with upload_col:

    uploaded_csv = st.file_uploader("请上传数据文件", type=[".csv", ".xlsx"])


with param_col:
    sub_para_cols = st.columns(2)
    skip_rows = sub_para_cols[0].number_input(
        "跳过的首行数目", min_value=0, value=0, key="skip_rows"
    )
    sheet_name = sub_para_cols[1].number_input(
        "Sheet索引", min_value=0, value=0, key="sheet_name"
    )

if uploaded_csv:
    path = st.session_state.cache_dir / uploaded_csv.name
    if uploaded_csv.name.endswith(".csv"):
        data = pd.read_csv(uploaded_csv, skiprows=skip_rows)
    elif uploaded_csv.name.endswith(".xlsx"):
        data = pd.read_excel(uploaded_csv, skiprows=skip_rows, sheet_name=sheet_name)
    else:
        raise NotImplementedError
    edit_data = upload_col.data_editor(data, hide_index=False, height=300)
    upload_col.divider()

    with param_col:
        options = list(edit_data.columns)

        sub_opt_cols = st.columns(2)
        with sub_opt_cols[0]:
            x_axis = st.selectbox(
                "选择X轴属性",
                options=options,
                index=options.index("ly") if "ly" in options else 0,
            )
            target = st.selectbox(
                "选择目标属性",
                options=options,
                index=options.index("freeE") if "freeE" in options else 0,
            )
            xmajor = st.number_input("X轴主刻度间隔", value=0.05, min_value=0.0)
        with sub_opt_cols[1]:
            y_axis = st.selectbox(
                "选择Y轴属性",
                options=options,
                index=options.index("lz") if "lz" in options else 0,
            )
            precision = st.number_input("插值精度", value=-2)
            ymajor = st.number_input("Y轴主刻度间隔", value=0.05, min_value=0.0)

        levels = st.text_input(
            "输入等高线(列表格式)", value=str([0, 0.0001, 0.001, 0.01, 0.015])
        )
        levels = eval(levels)

        manual_mode = st.selectbox(
            "选择等高线标注模式", options=["默认", "手动指定", "自动检测（仅限对角线）"], index=0
        )
        if manual_mode == "自动检测（仅限对角线）":
            manual = "auto"
        elif manual_mode == "手动指定":
            raw_manual_data = pd.DataFrame(
                data=[[0, 0]],
                columns=["x", "y"],
            )
            manual_data = st.data_editor(
                raw_manual_data, num_rows="dynamic", hide_index=False, width=320
            )
            manual = manual_data.values.tolist()
        elif manual_mode == "默认":
            manual = ()

        exclude = st.text_input("输入不标注的等高线(列表格式)", value=str([0, 0.015, 0.01]))
        exclude = eval(exclude)

        plot_button = st.button("绘制图像", use_container_width=True)

    with upload_col:
        if plot_button:
            land = Landscaper(path=path)
            land_res = land.prospect(
                data=edit_data,
                precision=precision,
                levels=levels,
                xmajor=xmajor,
                ymajor=ymajor,
                interactive=False,
                save=st.session_state.save_auto,
                exclude=exclude,
                manual=manual,
            )
            st.pyplot(land_res.fig)
