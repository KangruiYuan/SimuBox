import warnings
from pathlib import Path

import pandas as pd
import streamlit as st
from SimuBox import (
    PeakData,
    peak_fit,
    SCATTER_PLOT_CONFIG,
    init_plot_config,
    check_state,
)

init_plot_config(SCATTER_PLOT_CONFIG)

warnings.filterwarnings("ignore")
check_state(Path(__file__).parents[1])

st.set_page_config(layout="wide")
st.title(":blue[SimuBox] :red[Visual] : 曲线分峰")

with st.expander("曲线分峰使用说明"):
    st.markdown(
        """
        ### 曲线比较
        - 通用文件类型：csv, xlsx。需确保第一列为峰位，第二列为峰高。Excel可有多个sheet。
        - 参数类型
            
        """
    )

upload_col, param_col = st.columns([0.75, 0.25])
with upload_col:

    uploaded_csv = st.file_uploader("请上传数据文件", type=[".csv", ".xlsx"])

    peaks = pd.DataFrame(
        [
            {
                "amplitude": None,
                "center": 1640.0,
                "width": None,
                "fix_amplitude": False,
                "fix_center": True,
                "fix_width": False,
            },
            {
                "amplitude": None,
                "center": 1660.0,
                "width": None,
                "fix_amplitude": False,
                "fix_center": True,
                "fix_width": False,
            },
            {
                "amplitude": None,
                "center": 1680.0,
                "width": None,
                "fix_amplitude": False,
                "fix_center": True,
                "fix_width": False,
            },
            {
                "amplitude": None,
                "center": 1695.0,
                "width": None,
                "fix_amplitude": False,
                "fix_center": True,
                "fix_width": False,
            },
            {
                "amplitude": None,
                "center": 1713.0,
                "width": None,
                "fix_amplitude": False,
                "fix_center": True,
                "fix_width": False,
            },
            {
                "amplitude": None,
                "center": 1740.0,
                "width": None,
                "fix_amplitude": False,
                "fix_center": True,
                "fix_width": False,
            },
            {
                "amplitude": None,
                "center": None,
                "width": None,
                "fix_amplitude": False,
                "fix_center": False,
                "fix_width": False,
            },
        ]
    )
    # left_input_col, right_option_col = st.columns([0.75, 0.25])
    upload_col.markdown("**编辑初始峰及约束信息**")
    edited_peaks = upload_col.data_editor(
        peaks,
        num_rows="dynamic",
        use_container_width=False,
        hide_index=False,
        width=int(1200 * 0.75),
        height=360,
    )
    edited_peaks = edited_peaks.dropna(subset=["center"])
    edited_peaks = edited_peaks.sort_values(by="center")
    if len(edited_peaks) == 0:
        fit_min_x = 0
        fit_max_x = 100
    else:
        fit_min_x = min(edited_peaks["center"]) * 0.995
        fit_max_x = max(edited_peaks["center"]) * 1.005

with param_col:
    sub_para_cols = st.columns(2)
    skip_rows = sub_para_cols[0].number_input(
        "跳过的首行数目", min_value=0, value=0, key="skip_rows"
    )
    sheet_name = sub_para_cols[1].number_input(
        "Sheet索引", min_value=0, value=0, key="sheet_name"
    )

min_x = 0.0
max_x = 1e5
if uploaded_csv:
    if uploaded_csv.name.endswith(".csv"):
        data = pd.read_csv(uploaded_csv, skiprows=skip_rows)
    elif uploaded_csv.name.endswith(".xlsx"):
        data = pd.read_excel(uploaded_csv, skiprows=skip_rows, sheet_name=sheet_name)
    else:
        raise NotImplementedError
    save = st.session_state.cache_dir / uploaded_csv.name
    x = data.iloc[:, 0].values
    y = data.iloc[:, 1].values
    min_x = min(x)
    max_x = max(x)

with param_col:

    xlabel = st.text_input("请输入横(X)坐标名称", value=r"Wavenumbers/${\rm cm}^{-1}$")
    ylabel = st.text_input("请输入纵(Y)坐标名称", value=r"Absorbance (a.u.)")
    constraint_scale = st.slider(
        "请输入约束强度", min_value=0.9, max_value=1.1, value=(0.99, 1.01), format="%.3f"
    )
    sub_input_cols = st.columns(2)
    peaks_scale_min = sub_input_cols[0].number_input(
        "起始值", value=max(fit_min_x, min_x), key="peaks_scale_min", format="%.2f"
    )
    peaks_scale_max = sub_input_cols[1].number_input(
        "截止值", value=min(fit_max_x, max_x), key="peaks_scale_max", format="%.2f"
    )
    # peaks_scale = st.slider(
    #     "请输入拟合范围",
    #     min_value=min_x,
    #     max_value=max_x,
    #     value=(max(fit_min_x, min_x), min(fit_max_x, max_x)),
    # )
    interactive = st.checkbox("交互式绘图", value=False)
    plot_button = st.button("绘制图像", use_container_width=True)

st.divider()

plot_col, info_col = st.columns([0.6, 0.4])

with plot_col:
    peaks_res = None
    if plot_button and uploaded_csv:

        peaks = []
        for idx, row in edited_peaks.iterrows():
            row = row.tolist()
            # print(row[1], type(row[1]))
            peaks.append(
                PeakData(
                    amplitude=row[0],
                    center=row[1],
                    width=row[2],
                    fix=row[3:],
                )
            )

        mask = (x >= peaks_scale_min) & (x <= peaks_scale_max)
        plot_x = x[mask]
        plot_y = y[mask]
        # print(peaks)
        peaks_res = peak_fit(plot_x, plot_y, peaks, scale=constraint_scale)

        st.pyplot(peaks_res.fig)

with info_col:
    if plot_button and uploaded_csv and peaks_res is not None:

        sub_write_cols = st.columns(2)
        sub_write_cols[0].markdown(f"### $R^2$={peaks_res.r2: .4f}")
        sub_write_cols[1].markdown(f"### $Adj.R^2$={peaks_res.adj_r2: .4f}")

        fitted_peaks = peaks_res.peaks
        fitted_peaks.sort(key=lambda x: x.center)
        peaks_print_cols = st.columns(3)

        collect_data_output = []
        for i in range(len(fitted_peaks)):
            temp_peak = fitted_peaks[i]
            peaks_print_cols[i % 3].metric(
                f"Peak {i}",
                round(temp_peak.center, 4),
                round(temp_peak.center - peaks[i].center, 4),
            )
            peaks_print_cols[i % 3].markdown(
                f"""
                **amplitude**={temp_peak.amplitude: .4f}\n
                **width**={temp_peak.width: .4f}\n
                **area%**={temp_peak.area / peaks_res.area: .4%}
                """
            )
            collect_data_output.append(
                {
                    "amplitude": temp_peak.amplitude,
                    "center": temp_peak.center,
                    "width": temp_peak.width,
                    "background": temp_peak.background,
                    "area": temp_peak.area / peaks_res.area * 100,
                }
            )
        collect_output_df = pd.DataFrame(collect_data_output)
        st.dataframe(collect_output_df, use_container_width=True)
        # ** center **= {temp_peak.center: .2f}\n
