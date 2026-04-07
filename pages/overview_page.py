import importlib
import os
from typing import Any

import pandas as pd
import streamlit as st


SERVICE_MODULE = os.getenv(
    "EXOPLANET_STREAMLIT_SERVICE_MODULE",
    "exoplanet_detector.app.services",
)

# Optional: specify exactly which columns to show in the Feature importance table.
# Leave empty (`[]`) to display all available columns.
FEATURE_IMPORTANCE_DISPLAY_COLUMNS = ["importance_rank", "feature", "importance_mean", "importance_std"]


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

    st.markdown(
        """
        Select the model, dataset and plot type to evaluate against.
        
        **confusion_matrix**: Shows proportions of true positives, true negatives, false positives, and false negatives at the selected decision threshold. Use it to understand what kinds of errors the model is making in relative terms.

        **roc_curve**: Plots true positive rate against false positive rate across all thresholds. Use it to evaluate how well the model separates classes independently of one fixed threshold.

        **pr_curve**: Plots precision against recall across thresholds, focusing on positive-class performance. It is especially informative when classes are imbalanced and false positives/false negatives have different costs.
        """
    )

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
        left_col, center_col, right_col = st.columns([1, 2, 1])
        with center_col:
            st.image(str(plot_path), use_container_width=True)
        st.caption(f"Loaded from: {plot_path}")
    except Exception as exc:
        st.warning(f"Unable to load the selected plot: {exc}")


def render_feature_importance_section(service, context) -> None:
    st.subheader("Feature importance")

    st.markdown(
        """
        This table shows **permutation feature importance**, i.e. how much model performance changes when one feature is randomly shuffled while all other features stay unchanged.  
        Importance is computed repeatedly (20 times) for each feature on the selected dataset and model profile.  
        **importance_mean** is the average score impact across repeats (higher usually means the feature matters more), and **importance_std** is the variability of that impact (lower means more stable importance estimates).  
        A **negative** importance means shuffling that feature slightly improved the metric, which usually indicates noise, redundancy, or sampling variability.  
        Values are below 1 because importance was calculated as a change in optimized metric, which are in range [0-1].
        """
    )

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
        top_n = st.slider("Top features", min_value=5, max_value=11, value=11)

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

    if FEATURE_IMPORTANCE_DISPLAY_COLUMNS:
        display_columns = [
            col for col in FEATURE_IMPORTANCE_DISPLAY_COLUMNS if col in importance_df.columns
        ]
        if not display_columns:
            st.warning("None of the configured feature-importance columns were found.")
            st.caption(f"Available columns: {', '.join(importance_df.columns)}")
            return
        table_df = importance_df.loc[:, display_columns]
    else:
        table_df = importance_df
    st.dataframe(table_df, use_container_width=True, hide_index=True)

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
        """
    ### Exoplanet Candidate Detection

    The goal of this project is to classify exoplanet candidates as planet-like (1) or non-planet-like (0) from tabular 
    astrophysical measurements, and then expose the best model configurations in an interactive app for evaluation and prediction.
    
    The pipeline is built on two source datasets: KOI and K2P.
    KOI (Kepler Objects of Interest) provides records with known dispositions, which makes it suitable for supervised 
    training and model benchmarking. K2P represents a different candidate pool from the K2 context, with different 
    distribution characteristics and less reliable/limited labeling for training purposes. 
    Operationally, this is handled as a split into labeled data (used for fitting and validation) and unlabeled-style 
    data (used for realistic inference and explanation workflows).
    
    Feature selection was intentionally restricted to physical, interpretable variables only. 
    After removing non-physical identifiers, leakage-prone fields, and weak / redundant columns, 
    the final model input includes 11 features.
    
    Training is performed across five model families under a common evaluation setup. 
    From these, three deployable variants are selected as final profiles:
    * Balanced F2 profile (for best overall performance, where recall is valued slightly more than precision),
    * Recall-priority profile (maximum sensitivity, minimizes the risk of omitting a true planet),
    * Precision-priority profile (stricter positive calls, minimizes the risk of false alarms).
    This profile-based selection allows the same project to support different decision priorities without retraining the entire system.

    Use the controls below to explore model comparison tables, plots (confusion/ROC/PR), and feature-importance outputs.
    """
    )

    comparison_df = safe_dataframe(context.get("comparison_df"))
    manifest_df = safe_dataframe(context.get("deploy_manifest_df"))

    left, right = st.columns([1.25, 1.75])

    with left:
        st.subheader("Final models")
        if manifest_df.empty:
            st.info("No deploy manifest available.")
        else:
            display_cols = [
                col for col in ["deploy_id", "model", "model_name", "profile", "threshold"]
                if col in manifest_df.columns
            ]
            st.dataframe(manifest_df.loc[:, display_cols], use_container_width=True, hide_index=True)

    with right:
        st.subheader("All models")
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
