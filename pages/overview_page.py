import importlib
import os
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st


SERVICE_MODULE = os.getenv(
    "EXOPLANET_STREAMLIT_SERVICE_MODULE",
    "exoplanet_detector.app.services",
)


@st.cache_resource
def load_service():
    return importlib.import_module(SERVICE_MODULE)


@st.cache_resource
def load_context(run_tag: str):
    service = load_service()
    kwargs = {"run_tag": run_tag} if str(run_tag).strip() else {}
    return service.get_run_context(**kwargs)


def render_header(title: str, subtitle: str, links: dict[str, str]) -> None:
    left, *right_cols = st.columns([7, 1, 1, 1])
    with left:
        st.title(title)
        st.caption(subtitle)

    for col, (label, url) in zip(right_cols, links.items()):
        with col:
            st.link_button(label, url, use_container_width=True)


def safe_dataframe(obj: Any) -> pd.DataFrame:
    return obj if isinstance(obj, pd.DataFrame) else pd.DataFrame()


def resolve_model_label(model_options: list[dict[str, Any]], deploy_id: str) -> str:
    for option in model_options:
        if option["deploy_id"] == deploy_id:
            return option["label"]
    return deploy_id


def render_plot_section(service, context) -> None:
    st.subheader("Interactive evaluation explorer")

    model_options = service.list_models(context)
    datasets = service.list_datasets_for_plots(context)
    plot_types = service.list_plot_types()

    if not model_options:
        st.info("No deployed models were found in the manifest.")
        return
    if not datasets:
        st.info("No saved plot manifest was found.")
        return

    col1, col2, col3 = st.columns(3)
    with col1:
        selected_model_label = st.selectbox(
            "Model",
            options=[m["label"] for m in model_options],
            key="overview_model_label",
        )
    with col2:
        selected_dataset = st.selectbox("Dataset", options=datasets, key="overview_dataset")
    with col3:
        selected_plot_type = st.selectbox("Visualization", options=plot_types, key="overview_plot")

    selected_model = next(m for m in model_options if m["label"] == selected_model_label)

    try:
        plot_path = service.resolve_plot(
            context,
            deploy_id=selected_model["deploy_id"],
            dataset=selected_dataset,
            plot_type=selected_plot_type,
        )
        st.image(str(plot_path), use_container_width=True)
        st.caption(f"Loaded from: {plot_path}")
    except Exception as exc:
        st.warning(f"Unable to load the selected plot: {exc}")


def render_feature_importance_section(service, context) -> None:
    st.subheader("Feature importance")

    model_options = service.list_models(context)
    datasets = service.list_datasets_for_plots(context)

    if not model_options or not datasets:
        st.info("Feature-importance view becomes available when evaluation artifacts exist.")
        return

    col1, col2, col3 = st.columns([2, 2, 1])
    with col1:
        selected_model_label = st.selectbox(
            "Model for importance",
            options=[m["label"] for m in model_options],
            key="importance_model_label",
        )
    with col2:
        selected_dataset = st.selectbox(
            "Dataset for importance",
            options=datasets,
            key="importance_dataset",
        )
    with col3:
        top_n = st.slider("Top features", min_value=5, max_value=30, value=15)

    selected_model = next(m for m in model_options if m["label"] == selected_model_label)
    importance_df = service.get_feature_importance(
        context,
        deploy_id=selected_model["deploy_id"],
        dataset=selected_dataset,
        top_n=top_n,
    )

    if importance_df.empty:
        st.info("No feature-importance rows were found for that combination.")
        return

    st.dataframe(importance_df, use_container_width=True, hide_index=True)

    if {"feature", "importance_mean"}.issubset(importance_df.columns):
        chart_df = importance_df.loc[:, ["feature", "importance_mean"]].set_index("feature")
        st.bar_chart(chart_df)


def main() -> None:
    try:
        service = load_service()
        context = load_context(st.session_state.get("run_tag", ""))
    except Exception as exc:
        st.error(
            "The application could not load the service layer or run context. "
            "Update EXOPLANET_STREAMLIT_SERVICE_MODULE and verify that the artifacts exist."
        )
        st.exception(exc)
        return

    links = service.get_profile_links()
    render_header(
        title="Portfolio Project",
        subtitle="Overview, datasets, evaluation results and interpretability artifacts.",
        links=links,
    )

    st.markdown(
        '''
        This page is intended to introduce the project and let the user explore
        the available evaluation artifacts. Replace this text with your final
        project description, dataset summary, model rationale and conclusions.
        '''
    )

    comparison_df = safe_dataframe(context.get("comparison_df"))
    manifest_df = safe_dataframe(context.get("deploy_manifest_df"))

    left, right = st.columns([1.25, 1.75])

    with left:
        st.subheader("Run summary")
        st.write(f"Run tag: `{context.get('run_tag', '')}`")
        st.write(f"Example dataset: `{context.get('example_dataset_name', '')}`")
        st.write(f"Background dataset: `{context.get('background_dataset_name', '')}`")
        st.write(f"Feature count: `{len(context.get('feature_columns', []))}`")

        st.subheader("Available models")
        if manifest_df.empty:
            st.info("No deploy manifest available.")
        else:
            display_cols = [
                col for col in ["deploy_id", "model", "model_name", "profile", "threshold"]
                if col in manifest_df.columns
            ]
            st.dataframe(manifest_df.loc[:, display_cols], use_container_width=True, hide_index=True)

    with right:
        st.subheader("Model comparison")
        if comparison_df.empty:
            st.info("No comparison table available.")
        else:
            st.dataframe(comparison_df, use_container_width=True, hide_index=True)

    st.divider()
    render_plot_section(service, context)

    st.divider()
    render_feature_importance_section(service, context)

    with st.expander("Raw artifact tables"):
        st.write("Use this section only if you want to inspect the loaded metadata directly.")
        st.markdown("**Plot manifest**")
        st.dataframe(safe_dataframe(context.get("plot_manifest_df")), use_container_width=True)
        st.markdown("**Permutation importance**")
        st.dataframe(safe_dataframe(context.get("permutation_importance_df")), use_container_width=True)
        st.markdown("**Feature importance matrix**")
        st.dataframe(safe_dataframe(context.get("feature_importance_matrix_df")), use_container_width=True)


main()