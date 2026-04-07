import base64
import os
from pathlib import Path

import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parent
ICON_DIR = PROJECT_ROOT / "assets" / "icons"
CV_PATH = PROJECT_ROOT / "assets" / "cv.pdf"

PROFILE_LINKS = (
    {
        "label": "CV",
        "path": CV_PATH,
        "download_name": "Jan_Matusiak_CV.pdf",
        "icon": ICON_DIR / "cv.svg",
    },
    {
        "label": "LinkedIn",
        "url": "https://www.linkedin.com/in/janmatusiak/",
        "icon": ICON_DIR / "linkedin.svg",
    },
    {
        "label": "GitHub",
        "url": "https://github.com/JanMatusiak",
        "icon": ICON_DIR / "github.svg",
    },
    {
        "label": "Email",
        "url": "mailto:jan.matusiak.2006@gmail.com",
        "icon": ICON_DIR / "email.svg",
    },
)


@st.cache_data
def icon_data_uri(icon_path: str) -> str:
    path = Path(icon_path)
    mime = "image/svg+xml" if path.suffix.lower() == ".svg" else "image/png"
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{mime};base64,{encoded}"


@st.cache_data
def file_data_uri(file_path: str, mime: str) -> str:
    path = Path(file_path)
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{mime};base64,{encoded}"


def render_connect_link(label: str, url: str, icon_path: Path) -> None:
    icon_uri = icon_data_uri(str(icon_path))
    st.markdown(
        f"""
        <a href="{url}" target="_blank" rel="noopener noreferrer" style="
            display:flex;
            align-items:center;
            gap:0.5rem;
            padding:0.35rem 0;
            text-decoration:none;
            color:inherit;
        ">
            <img src="{icon_uri}" alt="{label} icon" width="18" height="18" />
            <span>{label}</span>
        </a>
        """,
        unsafe_allow_html=True,
    )


def render_connect_download(
    label: str,
    file_path: Path,
    icon_path: Path,
    *,
    download_name: str | None = None,
) -> None:
    icon_uri = icon_data_uri(str(icon_path))
    file_uri = file_data_uri(str(file_path), "application/pdf")
    filename = download_name or file_path.name
    st.markdown(
        f"""
        <a href="{file_uri}" download="{filename}" style="
            display:flex;
            align-items:center;
            gap:0.5rem;
            padding:0.35rem 0;
            text-decoration:none;
            color:inherit;
        ">
            <img src="{icon_uri}" alt="{label} icon" width="18" height="18" />
            <span>{label}</span>
        </a>
        """,
        unsafe_allow_html=True,
    )


st.set_page_config(
    page_title="Portfolio Project",
    page_icon="🪐",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    section[data-testid="stSidebar"] {
        min-width: 15rem !important;
        width: 15rem !important;
        max-width: 15rem !important;
    }

    button[data-testid="stSidebarCollapseButton"] {
        display: none !important;
    }

    button[data-testid="collapsedControl"] {
        display: none !important;
    }

    section[data-testid="stSidebar"] [data-testid="stSidebarHeader"] button {
        display: none !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
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
with st.sidebar:
    st.divider()
    st.markdown(
        """
        ### About Me
        #### Jan Matusiak
        Data Engineering Student at Gdansk University of Technology and a Developer at 
        Student Research Group of Measurement Systems and Automation.
        Passionate about data and ML.
        """
    )

    cv_link = next((link for link in PROFILE_LINKS if link["label"] == "CV"), None)
    if cv_link is not None:
        if "url" in cv_link:
            render_connect_link(cv_link["label"], str(cv_link["url"]), cv_link["icon"])
        else:
            render_connect_download(
                cv_link["label"],
                Path(cv_link["path"]),
                cv_link["icon"],
                download_name=cv_link.get("download_name"),
            )

    st.markdown("### Connect")
    for link in PROFILE_LINKS:
        if link["label"] == "CV":
            continue
        render_connect_link(link["label"], str(link["url"]), link["icon"])
    email_address = next(
        (link["url"].replace("mailto:", "") for link in PROFILE_LINKS if link["label"] == "Email"),
        "",
    )
    if email_address:
        st.caption(f"`{email_address}`")
navigation.run()
