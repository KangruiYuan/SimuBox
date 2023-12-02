import streamlit as st
from datetime import datetime
from pathlib import Path

st.set_page_config(
    page_title="SimuBox Visual",
    page_icon="🤞",
    layout="wide",
    initial_sidebar_state="auto",
    menu_items={
        "About": "https://github.com/KangruiYuan/SimuBox",
        "Report a bug": "https://github.com/KangruiYuan/SimuBox/issues",
    },
)

st.title(":blue[SimuBox] :red[Visual] :sunglasses:")
st.header('该项目由复旦大学李卫华教授课题组全体开发。', divider='rainbow')
st.caption(f'当前时间为: {datetime.now()}')

st.subheader("全局参数设置")

cols = st.columns(3)

cache = cols[0].text_input("缓存文件夹", value="cache")
st.session_state.current_dir = Path(__file__).parent
st.session_state.cache_dir = st.session_state.current_dir / cache
if not st.session_state.cache_dir.exists():
    st.session_state.cache_dir.mkdir()
cols[0].caption(st.session_state.cache_dir.resolve())

dpi = cols[1].number_input("保存图像dpi", value=150, min_value=50, max_value=300)
st.session_state.dpi = dpi

save_auto = cols[2].checkbox("自动保存相应图像", value=True)
st.session_state.save_auto = save_auto

save_name_auto = cols[2].checkbox("自动重命名", value=True)
st.session_state.save_name_auto = save_name_auto

st.markdown(
    f"""
    - 项目主页: [Github pages](https://github.com/KangruiYuan/SimuBox)
    - 报告错误: [Issue](https://github.com/KangruiYuan/SimuBox/issues)
    """
)
