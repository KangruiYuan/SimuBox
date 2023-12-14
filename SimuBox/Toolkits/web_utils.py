
import streamlit as st
from pathlib import Path

def check_state(parent_path: Path, **kwargs):

    if "save_auto" not in st.session_state:
        st.session_state.save_auto = kwargs.get("save_auto", False)

    if "cache_dir" not in st.session_state:
        st.session_state.cache_dir = parent_path / kwargs.get("cache", "cache")

        if not st.session_state.cache_dir.exists():
            st.session_state.cache_dir.mkdir()

    if "dpi" not in st.session_state:
        st.session_state.dpi = kwargs.get("dpi", 150)