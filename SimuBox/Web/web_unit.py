from pathlib import Path
import streamlit as st
import base64

logo = f"url(data:image/png;base64,{base64.b64encode((st.session_state.current_dir / 'SimuBox.png').read_bytes()).decode()})"


def check_state(**kwargs):

    if "save_auto" not in st.session_state:
        st.session_state.save_auto = kwargs.get("save_auto", False)

    if "current_dir" not in st.session_state:
        st.session_state.current_dir = Path(__file__).parent

    if "cache_dir" not in st.session_state:
        st.session_state.cache_dir = st.session_state.current_dir / kwargs.get(
            "cache", "cache"
        )

        if not st.session_state.cache_dir.exists():
            st.session_state.cache_dir.mkdir()

    if "dpi" not in st.session_state:
        st.session_state.dpi = kwargs.get("dpi", 150)


def add_logo(**kwargs):

    st.markdown(
        f"""
        <style>
            [data-testid="stSidebarNav"] {{
                background-image: {logo};
                background-repeat: no-repeat;
                padding-top: 0px;
                background-position: 20px 50px;
                background-size: 60%;
            }}
        </style>
        """,
        unsafe_allow_html=True,
    )
    # [data-testid="stSidebarNav"]::before {{
    #             content: "SimuBox";
    #             margin-left: 20px;
    #             margin-top: 20px;
    #             font-size: 30px;
    #             position: relative;
    #             top: 100px;
    #         }}
