"""Streamlit-facing service layer for loading artifacts and serving predictions."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from exoplanet_detector.config import (
    ARTIFACTS_DIR,
    DEFAULT_RUN_TAG,
    K2P_FILE,
    KOI_TEST_FILE,
    get_run_artifact_dirs,
)
from exoplanet_detector.features.feature_selection import FINAL_FEATURE_COLUMNS, PHYSICAL_INTERVALS
from exoplanet_detector.models.feature_analysis import load_deployed_models, predict_and_explain_single_row

RunContext = dict[str, Any]


def _dataset_path_by_name(dataset_name: str) -> Path:
    name = dataset_name.strip()
    if name == "K2P":
        return K2P_FILE
    if name == "KOI_test":
        return KOI_TEST_FILE
    raise ValueError(f"Unsupported dataset name: {dataset_name!r}. Use 'K2P' or 'KOI_test'.")


def _read_optional_csv(path: Path) -> pd.DataFrame:
    if path.exists():
        return pd.read_csv(path)
    return pd.DataFrame()


@lru_cache(maxsize=8)
def get_run_context(
    run_tag: str = DEFAULT_RUN_TAG,
    *,
    example_dataset: str = "K2P",
    background_dataset: str = "KOI_test",
) -> RunContext:
    """
    Load and cache app artifacts for a run tag.

    The returned context is read-mostly and intended to be reused by Streamlit pages.
    """
    run_dirs = get_run_artifact_dirs(run_tag, create=False)
    deployment_dir = run_dirs["deployment"]
    evaluation_dir = run_dirs["evaluation"]

    deployment_models_path = deployment_dir / "deploy_models.joblib"
    deployment_manifest_path = deployment_dir / "deploy_manifest.csv"
    comparison_path = evaluation_dir / "comparison_df.csv"

    visualization_dir = ARTIFACTS_DIR / "visualization" / run_tag
    plot_manifest_path = visualization_dir / "plot_manifest.csv"

    feature_analysis_dir = ARTIFACTS_DIR / "feature_analysis" / run_tag
    permutation_importance_path = feature_analysis_dir / "permutation_importance.csv"
    feature_importance_matrix_path = feature_analysis_dir / "feature_importance_matrix.csv"

    if not deployment_models_path.exists():
        raise FileNotFoundError(
            f"Deployment bundle not found: {deployment_models_path}. "
            "Run notebook 04 to generate deployment artifacts."
        )

    deployed_models, deployed_model_registry_df = load_deployed_models(deployment_models_path)
    deploy_manifest_df = _read_optional_csv(deployment_manifest_path)
    comparison_df = _read_optional_csv(comparison_path)
    plot_manifest_df = _read_optional_csv(plot_manifest_path)
    permutation_importance_df = _read_optional_csv(permutation_importance_path)
    feature_importance_matrix_df = _read_optional_csv(feature_importance_matrix_path)

    example_df = pd.read_csv(_dataset_path_by_name(example_dataset))
    background_df = pd.read_csv(_dataset_path_by_name(background_dataset))

    selected_features = list(FINAL_FEATURE_COLUMNS)
    background_x = background_df.loc[:, selected_features].copy()

    return {
        "run_tag": run_tag,
        "run_dirs": run_dirs,
        "deployed_models": deployed_models,
        "deployed_model_registry_df": deployed_model_registry_df,
        "deploy_manifest_df": deploy_manifest_df,
        "comparison_df": comparison_df,
        "plot_manifest_df": plot_manifest_df,
        "permutation_importance_df": permutation_importance_df,
        "feature_importance_matrix_df": feature_importance_matrix_df,
        "example_dataset_name": example_dataset,
        "example_df": example_df,
        "background_dataset_name": background_dataset,
        "background_df": background_x,
        "feature_columns": tuple(selected_features),
    }


def get_profile_links(overrides: dict[str, str] | None = None) -> dict[str, str]:
    """Return external profile/repository links for the app header."""
    default_links = {
        "GitHub": "",
        "LinkedIn": "",
        "Repository": "",
    }
    if overrides:
        default_links.update(overrides)
    return {label: url for label, url in default_links.items() if str(url).strip()}


def list_models(context: RunContext) -> list[dict[str, Any]]:
    """Return deployed model options for UI selectors."""
    manifest_df = context["deploy_manifest_df"]
    if isinstance(manifest_df, pd.DataFrame) and not manifest_df.empty:
        rows = []
        for _, row in manifest_df.iterrows():
            model_name = row["model"] if "model" in manifest_df.columns else row.get("model_name")
            profile = row.get("profile", "")
            deploy_id = row["deploy_id"]
            threshold = float(row["threshold"])
            rows.append(
                {
                    "deploy_id": deploy_id,
                    "model_name": model_name,
                    "profile": profile,
                    "threshold": threshold,
                    "label": f"{deploy_id} | {model_name} | {profile}",
                }
            )
        return sorted(rows, key=lambda item: item["deploy_id"])

    registry_df = context["deployed_model_registry_df"]
    rows = []
    for _, row in registry_df.iterrows():
        rows.append(
            {
                "deploy_id": row["deploy_id"],
                "model_name": row["model_name"],
                "profile": row["profile"],
                "threshold": float(row["threshold"]),
                "label": f"{row['deploy_id']} | {row['model_name']} | {row['profile']}",
            }
        )
    return sorted(rows, key=lambda item: item["deploy_id"])


def list_datasets_for_plots(context: RunContext) -> list[str]:
    """Return available dataset names from plot manifest."""
    plot_manifest_df = context["plot_manifest_df"]
    if not isinstance(plot_manifest_df, pd.DataFrame) or plot_manifest_df.empty:
        return []
    return sorted(plot_manifest_df["dataset"].dropna().astype(str).unique().tolist())


def list_plot_types() -> list[str]:
    """Return standardized visualization selector options."""
    return ["confusion_matrix", "roc_curve", "pr_curve"]


def _plot_type_to_column(plot_type: str) -> str:
    aliases = {
        "confusion_matrix": "confusion_matrix_path",
        "cm": "confusion_matrix_path",
        "roc_curve": "roc_curve_path",
        "roc": "roc_curve_path",
        "pr_curve": "pr_curve_path",
        "pr": "pr_curve_path",
        "precision_recall": "pr_curve_path",
    }
    key = plot_type.strip().lower()
    if key not in aliases:
        allowed = ", ".join(sorted(aliases))
        raise ValueError(f"Unsupported plot type: {plot_type!r}. Allowed: {allowed}")
    return aliases[key]


def resolve_plot(
    context: RunContext,
    *,
    deploy_id: str,
    dataset: str,
    plot_type: str,
) -> Path:
    """Resolve selected deploy/dataset/plot-type to a saved plot file path."""
    plot_manifest_df = context["plot_manifest_df"]
    if not isinstance(plot_manifest_df, pd.DataFrame) or plot_manifest_df.empty:
        raise FileNotFoundError("Plot manifest is empty. Run notebook 06 to generate visual artifacts.")

    column = _plot_type_to_column(plot_type)
    matched = plot_manifest_df[
        (plot_manifest_df["deploy_id"] == deploy_id) & (plot_manifest_df["dataset"] == dataset)
    ]
    if matched.empty:
        raise KeyError(
            f"No plot entry for deploy_id={deploy_id!r}, dataset={dataset!r}."
        )

    path = Path(matched.iloc[0][column])
    if not path.exists():
        raise FileNotFoundError(f"Plot file not found: {path}")
    return path


def get_feature_importance(
    context: RunContext,
    *,
    deploy_id: str,
    dataset: str,
    top_n: int = 15,
) -> pd.DataFrame:
    """Return sorted permutation-importance rows for a selected deploy/dataset."""
    permutation_importance_df = context["permutation_importance_df"]
    if not isinstance(permutation_importance_df, pd.DataFrame) or permutation_importance_df.empty:
        return pd.DataFrame(
            columns=["feature", "importance_mean", "importance_std", "importance_rank"]
        )

    filtered = permutation_importance_df[
        (permutation_importance_df["deploy_id"] == deploy_id)
        & (permutation_importance_df["dataset"] == dataset)
    ].copy()
    if filtered.empty:
        return filtered

    filtered = filtered.sort_values("importance_mean", ascending=False).reset_index(drop=True)
    if top_n > 0:
        filtered = filtered.head(top_n).reset_index(drop=True)
    return filtered


def _range_text(lower: float | None, upper: float | None) -> str:
    if lower is None and upper is None:
        return "No explicit bounds"
    if lower is None:
        return f"<= {upper:g}"
    if upper is None:
        return f">= {lower:g}"
    return f"{lower:g} to {upper:g}"


def get_input_schema() -> list[dict[str, Any]]:
    """Build feature form schema from final feature list and physical intervals."""
    schema: list[dict[str, Any]] = []
    for feature in FINAL_FEATURE_COLUMNS:
        lower, upper = PHYSICAL_INTERVALS.get(feature, (None, None))
        schema.append(
            {
                "name": feature,
                "label": feature.replace("_", " ").title(),
                "min_value": lower,
                "max_value": upper,
                "allowed_range_text": _range_text(lower, upper),
                "required": True,
            }
        )
    return schema


def list_example_records(
    context: RunContext,
    *,
    id_column: str = "group_id",
    max_rows: int | None = 500,
) -> pd.DataFrame:
    """Return a lightweight table of selectable example records."""
    df = context["example_df"].reset_index(drop=True).copy()
    if max_rows is not None and max_rows > 0:
        df = df.head(max_rows).copy()

    columns = ["example_row_id"]
    df.insert(0, "example_row_id", df.index.astype(int))

    if id_column in df.columns:
        columns.append(id_column)
    if "label" in df.columns:
        columns.append("label")

    return df.loc[:, columns].copy()


def get_example_record(context: RunContext, *, row_idx: int) -> dict[str, float]:
    """Return one record (final features only) as a feature-value dictionary."""
    df = context["example_df"].reset_index(drop=True)
    if not 0 <= int(row_idx) < len(df):
        raise IndexError(f"row_idx out of bounds: {row_idx}. Valid range is [0, {len(df) - 1}].")

    row = df.loc[int(row_idx), list(FINAL_FEATURE_COLUMNS)]
    record: dict[str, float] = {}
    for feature in FINAL_FEATURE_COLUMNS:
        value = row[feature]
        record[feature] = float(value) if pd.notna(value) else float("nan")
    return record


def validate_inputs(
    feature_values: dict[str, Any],
    *,
    strict: bool = True,
) -> tuple[dict[str, float], list[str]]:
    """Validate and normalize user-provided feature values."""
    errors: list[str] = []
    cleaned: dict[str, float] = {}

    for feature in FINAL_FEATURE_COLUMNS:
        if feature not in feature_values:
            errors.append(f"Missing required feature: {feature}")
            continue

        raw_value = feature_values[feature]
        if raw_value is None:
            errors.append(f"{feature}: value is required")
            continue

        try:
            value = float(raw_value)
        except (TypeError, ValueError):
            errors.append(f"{feature}: expected numeric value, got {raw_value!r}")
            continue

        lower, upper = PHYSICAL_INTERVALS.get(feature, (None, None))
        if np.isfinite(value):
            if lower is not None and value < lower:
                errors.append(f"{feature}: {value:g} is below minimum {lower:g}")
            if upper is not None and value > upper:
                errors.append(f"{feature}: {value:g} is above maximum {upper:g}")
        cleaned[feature] = value

    if strict and errors:
        raise ValueError("; ".join(errors))
    return cleaned, errors


def _resolve_binary_labels(classes: np.ndarray) -> tuple[Any, Any]:
    if classes.shape[0] != 2:
        raise ValueError(f"Expected binary classes, got {classes.tolist()!r}")
    positive_label = 1 if 1 in classes else classes[-1]
    negative_label = [label for label in classes if label != positive_label][0]
    return positive_label, negative_label


def predict(
    context: RunContext,
    *,
    deploy_id: str,
    feature_values: dict[str, Any],
) -> dict[str, Any]:
    """Run one-row prediction and thresholded classification for a selected deployed model."""
    cleaned, _ = validate_inputs(feature_values, strict=True)
    x_row = pd.DataFrame([cleaned], columns=list(FINAL_FEATURE_COLUMNS))

    deployed_models = context["deployed_models"]
    if deploy_id not in deployed_models:
        available = ", ".join(sorted(deployed_models.keys()))
        raise KeyError(f"Unknown deploy_id {deploy_id!r}. Available: {available}")

    spec = deployed_models[deploy_id]
    tuned_model = spec["model"]
    base_estimator = tuned_model.estimator_
    if not hasattr(base_estimator, "predict_proba"):
        raise ValueError("Selected model does not support predict_proba.")

    tuned_classes = np.asarray(tuned_model.classes_)
    positive_label, negative_label = _resolve_binary_labels(tuned_classes)

    estimator_classes = np.asarray(base_estimator.classes_)
    pos_idx = int(np.where(estimator_classes == positive_label)[0][0])
    neg_idx = int(np.where(estimator_classes == negative_label)[0][0])

    score_matrix = base_estimator.predict_proba(x_row)
    if score_matrix.ndim != 2 or score_matrix.shape[0] != 1:
        raise ValueError(f"Unexpected predict_proba output shape: {score_matrix.shape!r}")

    probability_positive = float(score_matrix[0, pos_idx])
    probability_negative = float(score_matrix[0, neg_idx])
    threshold = float(spec["threshold"])
    prediction = positive_label if probability_positive >= threshold else negative_label

    return {
        "deploy_id": deploy_id,
        "model_name": spec["model_name"],
        "profile": spec["profile"],
        "threshold": threshold,
        "score": probability_positive,
        "probability_positive": probability_positive,
        "probability_negative": probability_negative,
        "prediction": prediction,
        "positive_label": positive_label,
        "negative_label": negative_label,
        "feature_values": cleaned,
    }


def explain(
    context: RunContext,
    *,
    deploy_id: str,
    feature_values: dict[str, Any],
    max_background_rows: int = 200,
    random_state: int = 42,
    kernel_nsamples: int = 200,
    make_waterfall_plot: bool = True,
    waterfall_max_display: int = 12,
    show_waterfall: bool = False,
    explainer_cache: dict[tuple[str, str], Any] | None = None,
) -> dict[str, Any]:
    """Return SHAP-based explanation payload for one user-provided record."""
    cleaned, _ = validate_inputs(feature_values, strict=True)
    return predict_and_explain_single_row(
        context["deployed_models"],
        deploy_id=deploy_id,
        row=cleaned,
        background_data=context["background_df"],
        feature_columns=FINAL_FEATURE_COLUMNS,
        max_background_rows=max_background_rows,
        random_state=random_state,
        kernel_nsamples=kernel_nsamples,
        make_waterfall_plot=make_waterfall_plot,
        waterfall_max_display=waterfall_max_display,
        show_waterfall=show_waterfall,
        explainer_cache=explainer_cache,
    )


__all__ = [
    "explain",
    "get_example_record",
    "get_feature_importance",
    "get_input_schema",
    "get_profile_links",
    "get_run_context",
    "list_datasets_for_plots",
    "list_example_records",
    "list_models",
    "list_plot_types",
    "predict",
    "resolve_plot",
    "validate_inputs",
]
