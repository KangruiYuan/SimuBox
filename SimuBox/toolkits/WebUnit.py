from pathlib import Path
import streamlit as st
import base64

__all__ = ["check_state", "add_logo"]

def check_state(**kwargs):

    if "save_auto" not in st.session_state:
        st.session_state.save_auto = kwargs.get("save_auto", False)

    if "current_dir" not in st.session_state:
        st.session_state["current_dir"] = Path(__file__).parent

    if "cache_dir" not in st.session_state:
        st.session_state.cache_dir = st.session_state.current_dir / kwargs.get(
            "cache", "cache"
        )

        if not st.session_state.cache_dir.exists():
            st.session_state.cache_dir.mkdir()

    if "dpi" not in st.session_state:
        st.session_state.dpi = kwargs.get("dpi", 150)

    # if "logo" not in st.session_state:
    #     logo_path = st.session_state.current_dir / 'SimuBox.png'
    #     st.session_state.logo = f"url(data:image/png;base64,{base64.b64encode(logo_path.read_bytes()).decode()})"


def add_logo(**kwargs):

    st.markdown(
        f"""
        <style>
            [data-testid="stSidebarNav"] {{
                background-image: {st.session_state.logo};
                background-repeat: no-repeat;
                padding-top: 0px;
                background-position: 20px 50px;
                background-size: 60%;
            }}
        </style>
        """,
        unsafe_allow_html=True,
    )