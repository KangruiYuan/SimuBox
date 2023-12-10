import warnings
from copy import deepcopy

import pandas as pd
import plotly.graph_objs as go
import streamlit as st
from SimuBox import CompareJudger, CompareMode, COMPARE_PLOT_CONFIG, init_plot_config

warnings.filterwarnings("ignore")

init_plot_config(COMPARE_PLOT_CONFIG)

st.set_page_config(layout="wide")
st.title(":blue[SimuBox] :red[Visual] : 曲线比较")

with st.expander("曲线比较使用说明"):
    st.markdown(
        """
        ### 曲线比较
        - 通用文件类型：csv
        - 参数类型
        """
    )

uploaded_csv = st.file_uploader("请上传数据CSV", type=[".csv"])

if uploaded_csv:
    data = pd.read_csv(uploaded_csv)
    edit_data = st.data_editor(data, height=250)
    phase = set(edit_data.phase)
    backup_phase = deepcopy(phase)


st.divider()

if uploaded_csv:
    save = st.session_state.cache_dir / uploaded_csv.name
    compare = CompareJudger(path=save)

    plot_col, info_col = st.columns([0.75, 0.25])

    with info_col:
        mode = st.selectbox(
            "请选择比较模式", options=[CompareMode.DIFF.value, CompareMode.ABS.value], index=0
        )
        plot_button = st.button("绘制图像")
        interactive = st.checkbox("开启交互式绘图", value=False)

        st.divider()
        xlabel = st.selectbox("请选择横坐标", options=edit_data.columns, index=0)
        ylabel = st.selectbox("请选择纵坐标", options=edit_data.columns, index=0)

        if mode == CompareMode.DIFF:
            base = st.selectbox("请选择基准相", options=phase, index=0)
            backup_phase.remove(base)
            others = st.multiselect(
                "请选择其他相", options=backup_phase, default=backup_phase
            )
        elif mode == CompareMode.ABS:
            selected_phase = st.multiselect("请选择所需相", options=phase, default=phase)

    with plot_col:
        if mode == CompareMode.ABS and plot_button:
            with st.spinner("绘图中"):
                plot_result = compare.abs_comparison(
                    phases=selected_phase,
                    xlabel=xlabel,
                    ylabel=ylabel,
                    interactive=False,
                    data=edit_data,
                    save=st.session_state.save_auto,
                )
        elif mode == CompareMode.DIFF and plot_button:
            with st.spinner("绘图中"):
                plot_result = compare.diff_comparison(
                    base=base,
                    others=others,
                    xlabel=xlabel,
                    ylabel=ylabel,
                    interactive=False,
                    data=edit_data,
                    save=st.session_state.save_auto,
                )

        if plot_button:
            if not interactive:
                st.pyplot(plot_result.fig)
            else:
                x_min = min(min(line.x) for line in plot_result.lines)
                x_max = max(max(line.x) for line in plot_result.lines)
                fig = go.Figure()
                for line in plot_result.lines:
                    fig.add_trace(
                        go.Scatter(
                            x=line.x,
                            y=line.y,
                            name=line.label,
                            mode="lines+markers"
                            # marker=go.scatter.Marker.angle
                        )
                    )
                fig.update_layout(
                    autosize=False,
                    width=800,
                    height=600,
                    xaxis_title=plot_result.xlabel,
                    yaxis_title=plot_result.ylabel,
                )
                # st.components.v1.html(fig.to_html(include_mathjax='cdn'), height=500)
                st.plotly_chart(fig, use_container_width=True)

                # line_charts = Line().set_global_opts(
                #     title_opts=opts.TitleOpts(title="相对值比较结果图"),
                #     yaxis_opts=opts.AxisOpts(
                #         name=plot_result.ylabel,
                #         name_location="middle",
                #         name_gap=40,
                #         name_textstyle_opts=opts.TextStyleOpts(
                #             font_weight="bold", font_size=20
                #         ),
                #     ),
                #     xaxis_opts=opts.AxisOpts(
                #         name=plot_result.xlabel,
                #         name_location="end",
                #         name_gap=20,
                #         name_textstyle_opts=opts.TextStyleOpts(
                #             font_weight="bold", font_size=20
                #         ),
                #         min_=x_min,
                #         max_=x_max
                #     ),
                #     datazoom_opts=opts.DataZoomOpts(
                #         type_="slider",
                #         is_show=True,
                #         # start_value="0",
                #         # end_value="8",
                #         range_start=0,
                #         range_end=100,
                #         orient="horizontal",
                #         pos_bottom="15px",
                #     ),
                # )
                #
                # for line in plot_result.lines:
                #     line_charts.add_xaxis(line.x).add_yaxis(
                #         line.label,
                #         line.y,
                #         is_symbol_show=True,
                #         symbol=None,
                #         symbol_size=6,
                #         # areastyle_opts=opts.AreaStyleOpts(opacity=0.5),
                #     ).set_series_opts(
                #         label_opts=opts.LabelOpts(is_show=False),
                #     )
                #
                # st_pyecharts(line_charts, height="500px")
