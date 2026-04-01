import importlib
import math
import os
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st


SERVICE_MODULE = os.getenv(
    "EXOPLANET_STREAMLIT_SERVICE_MODULE",
    "your.service.module",
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


def choose_example_record(service, context) -> tuple[pd.DataFrame, dict[str, float]]:
    example_index_table = service.list_example_records(context)
    if example_index_table.empty:
        st.info("No example records are available.")
        return example_index_table, {}

    selector_col = "example_row_id"
    if "group_id" in example_index_table.columns:
        selector_col = "group_id"

    selection = st.selectbox(
        "Choose an example record",
        options=example_index_table[selector_col].tolist(),
        key="predictor_example_selector",
    )

    if selector_col == "group_id":
        matched = example_index_table[example_index_table["group_id"] == selection].iloc[0]
        row_idx = int(matched["example_row_id"])
    else:
        row_idx = int(selection)

    record = service.get_example_record(context, row_idx=row_idx)
    preview_cols = [col for col in ["example_row_id", "group_id", "label"] if col in example_index_table.columns]
    st.dataframe(
        example_index_table[example_index_table["example_row_id"] == row_idx][preview_cols],
        use_container_width=True,
        hide_index=True,
    )
    return example_index_table, record


def default_feature_value(feature_schema: dict[str, Any], loaded_value: float | None = None) -> float:
    if loaded_value is not None and not math.isnan(float(loaded_value)):
        return float(loaded_value)

    lower = feature_schema.get("min_value")
    upper = feature_schema.get("max_value")

    if lower is not None and upper is not None:
        return float((lower + upper) / 2.0)
    if lower is not None:
        return float(lower)
    if upper is not None:
        return float(upper)
    return 0.0


def build_feature_form(service, schema: list[dict[str, Any]], initial_values: dict[str, float]) -> dict[str, float]:
    values: dict[str, float] = {}
    columns = st.columns(2)

    for idx, feature in enumerate(schema):
        col = columns[idx % 2]
        with col:
            kwargs: dict[str, Any] = {
                "label": feature["label"],
                "value": default_feature_value(feature, initial_values.get(feature["name"])),
                "format": "%.6f",
                "help": f"Allowed range: {feature['allowed_range_text']}",
                "key": f"input_{feature['name']}",
            }
            if feature["min_value"] is not None:
                kwargs["min_value"] = float(feature["min_value"])
            if feature["max_value"] is not None:
                kwargs["max_value"] = float(feature["max_value"])

            values[feature["name"]] = st.number_input(**kwargs)
            st.caption(f"Allowed range: {feature['allowed_range_text']}")

    return values


def render_prediction_output(prediction: dict[str, Any]) -> None:
    st.subheader("Prediction result")

    score = float(prediction["score"])
    threshold = float(prediction["threshold"])
    decision = prediction["prediction"]
    margin = score - threshold

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Positive score", f"{score:.4f}")
    col2.metric("Threshold", f"{threshold:.4f}")
    col3.metric("Decision", str(decision))
    col4.metric("Margin vs threshold", f"{margin:+.4f}")

    with st.expander("Prediction payload"):
        st.json(
            {
                "deploy_id": prediction.get("deploy_id"),
                "model_name": prediction.get("model_name"),
                "profile": prediction.get("profile"),
                "positive_label": str(prediction.get("positive_label")),
                "negative_label": str(prediction.get("negative_label")),
                "probability_positive": prediction.get("probability_positive"),
                "probability_negative": prediction.get("probability_negative"),
            }
        )


def render_explanation_output(explanation: dict[str, Any]) -> None:
    st.subheader("Explanation")

    if not isinstance(explanation, dict) or not explanation:
        st.info("No explanation payload was returned.")
        return

    shown_anything = False

    for key, value in explanation.items():
        if isinstance(value, pd.DataFrame):
            st.markdown(f"**{key}**")
            st.dataframe(value, use_container_width=True, hide_index=True)
            shown_anything = True
            continue

        if isinstance(value, str):
            lower_key = key.lower()
            looks_like_path = "path" in lower_key or "plot" in lower_key
            if looks_like_path and Path(value).exists():
                st.markdown(f"**{key}**")
                st.image(value, use_container_width=True)
                shown_anything = True
                continue

    scalar_payload = {}
    for key, value in explanation.items():
        if isinstance(value, (str, int, float, bool)) or value is None:
            if isinstance(value, str):
                lower_key = key.lower()
                looks_like_path = "path" in lower_key or "plot" in lower_key
                if looks_like_path and Path(value).exists():
                    continue
            scalar_payload[key] = value

    if scalar_payload:
        st.markdown("**Explanation payload**")
        st.json(scalar_payload)
        shown_anything = True

    if not shown_anything:
        st.info("Explanation was returned, but no directly renderable fields were detected.")


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
        subtitle="Choose a model, provide feature values and inspect the prediction.",
        links=links,
    )

    model_options = service.list_models(context)
    if not model_options:
        st.error("No deployed models are available.")
        return

    selected_model_label = st.selectbox(
        "Choose a deployed model",
        options=[m["label"] for m in model_options],
        key="predictor_model_label",
    )
    selected_model = next(m for m in model_options if m["label"] == selected_model_label)

    input_mode = st.radio(
        "Input mode",
        options=["Manual values", "Load example record"],
        horizontal=True,
    )

    initial_values: dict[str, float] = {}
    if input_mode == "Load example record":
        _, initial_values = choose_example_record(service, context)

    schema = service.get_input_schema()

    with st.form("prediction_form"):
        feature_values = build_feature_form(service, schema, initial_values)
        submitted = st.form_submit_button("Run prediction", use_container_width=True)

    if not submitted:
        st.info("Provide inputs and submit the form to run the selected model.")
        return

    try:
        cleaned_values, validation_errors = service.validate_inputs(feature_values, strict=False)
        if validation_errors:
            for error in validation_errors:
                st.warning(error)
            st.stop()

        prediction = service.predict(
            context,
            deploy_id=selected_model["deploy_id"],
            feature_values=cleaned_values,
        )
        render_prediction_output(prediction)

        explanation = service.explain(
            context,
            deploy_id=selected_model["deploy_id"],
            feature_values=cleaned_values,
        )
        render_explanation_output(explanation)

    except Exception as exc:
        st.error("Prediction failed.")
        st.exception(exc)


main()
