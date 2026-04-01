import os
import streamlit as st

st.set_page_config(
    page_title="Portfolio Project",
    page_icon="🪐",
    layout="wide",
    initial_sidebar_state="expanded",
)

if "run_tag" not in st.session_state:
    st.session_state.run_tag = os.getenv("EXOPLANET_STREAMLIT_RUN_TAG", "")

if "service_module" not in st.session_state:
    st.session_state.service_module = os.getenv(
        "EXOPLANET_STREAMLIT_SERVICE_MODULE",
        "your.service.module",
    )

overview_page = st.Page(
    "pages/overview_page.py",
    title="Overview & Evaluation",
    icon="📊",
    default=True,
)
predictor_page = st.Page(
    "pages/predictor_page.py",
    title="Predictor",
    icon="🔭",
)

navigation = st.navigation([overview_page, predictor_page], position="sidebar")
navigation.run()