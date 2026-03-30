"""Feature-importance analysis helpers for deployed models."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance

from exoplanet_detector.config import ARTIFACTS_DIR, DEFAULT_RUN_TAG, TARGET_COLUMN
from exoplanet_detector.features.feature_selection import FINAL_FEATURE_COLUMNS

EvaluationSets = dict[str, tuple[pd.DataFrame, pd.Series]]
DeployedModels = dict[str, dict[str, Any]]


def get_feature_analysis_paths(
    run_tag: str = DEFAULT_RUN_TAG,
    *,
    create: bool = True,
) -> dict[str, Path]:
    """Return feature-analysis artifact paths for a run tag."""
    artifact_dir = ARTIFACTS_DIR / "feature_analysis" / run_tag
    if create:
        artifact_dir.mkdir(parents=True, exist_ok=True)
    return {
        "artifact_dir": artifact_dir,
        "permutation_importance_path": artifact_dir / "permutation_importance.csv",
        "feature_analysis_meta_path": artifact_dir / "feature_analysis_meta.json",
        "feature_importance_matrix_path": artifact_dir / "feature_importance_matrix.csv",
    }


def load_deployed_models(deploy_models_path: str | Path) -> tuple[DeployedModels, pd.DataFrame]:
    """Load deployment bundle and return normalized model specs + registry table."""
    deploy_bundle = joblib.load(Path(deploy_models_path))
    if not isinstance(deploy_bundle, dict):
        raise TypeError(
            f"Expected deployment bundle dict, got {type(deploy_bundle)} at {deploy_models_path}."
        )

    deployed_models: DeployedModels = {
        deploy_id: {
            "model": payload["model"],
            "model_name": payload["model_name"],
            "profile": payload["profile"],
            "threshold": float(payload["threshold"]),
        }
        for deploy_id, payload in deploy_bundle.items()
    }

    registry_df = pd.DataFrame(
        [
            {
                "deploy_id": deploy_id,
                "model_name": spec["model_name"],
                "profile": spec["profile"],
                "threshold": spec["threshold"],
            }
            for deploy_id, spec in deployed_models.items()
        ]
    ).sort_values("deploy_id").reset_index(drop=True)

    return deployed_models, registry_df


def build_evaluation_sets(
    koi_test_set: pd.DataFrame,
    k2p_set: pd.DataFrame,
    *,
    feature_columns: Sequence[str] = FINAL_FEATURE_COLUMNS,
    label_column: str = TARGET_COLUMN,
    include_combined: bool = True,
) -> EvaluationSets:
    """Build evaluation datasets used for permutation importance."""
    selected_columns = list(feature_columns)
    missing_final_features = [
        column
        for column in selected_columns
        if column not in koi_test_set.columns or column not in k2p_set.columns
    ]
    if missing_final_features:
        missing_csv = ", ".join(missing_final_features)
        raise KeyError(f"Missing final feature columns in test datasets: {missing_csv}")

    for dataset_name, dataset_df in (("KOI_test", koi_test_set), ("K2P", k2p_set)):
        if label_column not in dataset_df.columns:
            raise KeyError(f"Missing `{label_column}` in {dataset_name} dataframe.")

    x_koi_test = koi_test_set.loc[:, selected_columns].copy()
    y_koi_test = koi_test_set[label_column].copy()
    x_k2p = k2p_set.loc[:, selected_columns].copy()
    y_k2p = k2p_set[label_column].copy()

    evaluation_sets: EvaluationSets = {
        "KOI_test": (x_koi_test, y_koi_test),
        "K2P": (x_k2p, y_k2p),
    }
    if include_combined:
        x_combined = pd.concat([x_koi_test, x_k2p], axis=0, ignore_index=True)
        y_combined = pd.concat([y_koi_test, y_k2p], axis=0, ignore_index=True)
        evaluation_sets["KOI_test_plus_K2P"] = (x_combined, y_combined)
    return evaluation_sets


def _cache_config_matches(
    feature_analysis_meta: Mapping[str, Any],
    *,
    run_tag: str,
    scoring_name: str,
    n_repeats: int,
    random_state: int,
    n_jobs: int,
    dataset_names: Sequence[str],
) -> bool:
    return (
        feature_analysis_meta.get("run_tag") == run_tag
        and feature_analysis_meta.get("scoring") == scoring_name
        and feature_analysis_meta.get("n_repeats") == n_repeats
        and feature_analysis_meta.get("random_state") == random_state
        and feature_analysis_meta.get("n_jobs") == n_jobs
        and set(feature_analysis_meta.get("datasets", [])) == set(dataset_names)
    )


def _compute_permutation_importance(
    deployed_models: Mapping[str, Mapping[str, Any]],
    evaluation_sets: Mapping[str, tuple[pd.DataFrame, pd.Series]],
    *,
    scorer: Any,
    scoring_name: str,
    n_repeats: int,
    random_state: int,
    n_jobs: int,
) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []

    for deploy_id, spec in deployed_models.items():
        model = spec["model"]
        model_name = spec["model_name"]
        profile = spec["profile"]
        threshold = float(spec["threshold"])

        for dataset_name, (x_eval, y_eval) in evaluation_sets.items():
            result = permutation_importance(
                model,
                x_eval,
                y_eval,
                scoring=scorer,
                n_repeats=n_repeats,
                random_state=random_state,
                n_jobs=n_jobs,
            )
            dataset_df = pd.DataFrame(
                {
                    "deploy_id": deploy_id,
                    "model_name": model_name,
                    "profile": profile,
                    "threshold": threshold,
                    "dataset": dataset_name,
                    "feature": list(x_eval.columns),
                    "importance_mean": result.importances_mean,
                    "importance_std": result.importances_std,
                    "n_repeats": n_repeats,
                    "scoring": scoring_name,
                }
            )
            dataset_df["importance_rank"] = (
                dataset_df["importance_mean"].rank(method="dense", ascending=False).astype(int)
            )
            rows.append(dataset_df)

    permutation_importance_df = pd.concat(rows, axis=0, ignore_index=True)
    return permutation_importance_df.sort_values(
        ["deploy_id", "dataset", "importance_rank", "feature"],
        ascending=[True, True, True, True],
    ).reset_index(drop=True)


def compute_or_load_permutation_importance(
    deployed_models: Mapping[str, Mapping[str, Any]],
    evaluation_sets: Mapping[str, tuple[pd.DataFrame, pd.Series]],
    *,
    run_tag: str,
    permutation_importance_path: str | Path,
    feature_analysis_meta_path: str | Path,
    scorer: Any,
    scoring_name: str = "f2",
    n_repeats: int = 20,
    random_state: int = 42,
    n_jobs: int = 1,
    force_recompute: bool = False,
) -> tuple[pd.DataFrame, dict[str, Any], bool]:
    """Compute permutation importance or load cached results."""
    importance_path = Path(permutation_importance_path)
    meta_path = Path(feature_analysis_meta_path)

    if importance_path.exists() and meta_path.exists() and not force_recompute:
        feature_analysis_meta = json.loads(meta_path.read_text())
        matches = _cache_config_matches(
            feature_analysis_meta,
            run_tag=run_tag,
            scoring_name=scoring_name,
            n_repeats=n_repeats,
            random_state=random_state,
            n_jobs=n_jobs,
            dataset_names=list(evaluation_sets.keys()),
        )
        if matches:
            return pd.read_csv(importance_path), dict(feature_analysis_meta), True

    permutation_importance_df = _compute_permutation_importance(
        deployed_models,
        evaluation_sets,
        scorer=scorer,
        scoring_name=scoring_name,
        n_repeats=n_repeats,
        random_state=random_state,
        n_jobs=n_jobs,
    )

    importance_path.parent.mkdir(parents=True, exist_ok=True)
    permutation_importance_df.to_csv(importance_path, index=False)

    feature_analysis_meta = {
        "run_tag": run_tag,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "scoring": scoring_name,
        "n_repeats": n_repeats,
        "random_state": random_state,
        "n_jobs": n_jobs,
        "n_deployed_models": len(deployed_models),
        "datasets": list(evaluation_sets.keys()),
        "rows": int(permutation_importance_df.shape[0]),
    }
    meta_path.write_text(json.dumps(feature_analysis_meta, indent=2))
    return permutation_importance_df, feature_analysis_meta, False


def build_feature_importance_matrix(
    permutation_importance_df: pd.DataFrame,
    *,
    deploy_ids: Sequence[str],
    dataset_names: Sequence[str],
    feature_order: Sequence[str] = FINAL_FEATURE_COLUMNS,
) -> pd.DataFrame:
    """Pivot permutation-importance rows into feature x model-dataset matrix."""
    importance_matrix_long = permutation_importance_df.loc[
        :, ["feature", "deploy_id", "dataset", "importance_mean"]
    ].copy()
    importance_matrix_long["model_dataset"] = (
        importance_matrix_long["deploy_id"] + "__" + importance_matrix_long["dataset"]
    )

    ordered_model_dataset_columns = [
        f"{deploy_id}__{dataset_name}"
        for deploy_id in deploy_ids
        for dataset_name in dataset_names
    ]

    return (
        importance_matrix_long.pivot_table(
            index="feature",
            columns="model_dataset",
            values="importance_mean",
            aggfunc="first",
        )
        .reindex(index=list(feature_order))
        .reindex(columns=ordered_model_dataset_columns)
        .reset_index()
    )


def _to_single_row_dataframe(
    row: pd.Series | pd.DataFrame | Mapping[str, Any],
    *,
    feature_columns: Sequence[str],
) -> pd.DataFrame:
    if isinstance(row, pd.DataFrame):
        if row.shape[0] != 1:
            raise ValueError(f"`row` dataframe must contain exactly one row, got {row.shape[0]}.")
        row_df = row.copy()
    elif isinstance(row, pd.Series):
        row_df = pd.DataFrame([row.to_dict()])
    else:
        row_df = pd.DataFrame([dict(row)])

    selected_columns = list(feature_columns)
    missing_columns = [column for column in selected_columns if column not in row_df.columns]
    if missing_columns:
        missing_csv = ", ".join(missing_columns)
        raise KeyError(f"Missing required feature columns in `row`: {missing_csv}")

    return row_df.loc[:, selected_columns].copy()


def _resolve_positive_label(classes: Sequence[Any]) -> Any:
    class_array = np.asarray(classes)
    if class_array.shape[0] != 2:
        raise ValueError("Binary classification is required for threshold-based prediction.")
    return 1 if 1 in class_array else class_array[-1]


def _resolve_negative_label(classes: Sequence[Any], positive_label: Any) -> Any:
    class_array = np.asarray(classes)
    negatives = [label for label in class_array if label != positive_label]
    if not negatives:
        raise ValueError("Could not determine negative class label.")
    return negatives[0]


def _positive_class_index(classes: Sequence[Any], positive_label: Any) -> int:
    class_array = np.asarray(classes)
    matches = np.where(class_array == positive_label)[0]
    if matches.size == 0:
        raise ValueError(f"Positive label {positive_label!r} not found in classes: {list(class_array)!r}")
    return int(matches[0])


def _sample_background_data(
    background_data: pd.DataFrame,
    *,
    feature_columns: Sequence[str],
    max_background_rows: int,
    random_state: int,
) -> pd.DataFrame:
    selected_columns = list(feature_columns)
    missing_columns = [column for column in selected_columns if column not in background_data.columns]
    if missing_columns:
        missing_csv = ", ".join(missing_columns)
        raise KeyError(f"Missing required feature columns in `background_data`: {missing_csv}")

    sampled = background_data.loc[:, selected_columns].copy()
    if sampled.shape[0] > max_background_rows:
        sampled = sampled.sample(n=max_background_rows, random_state=random_state)
    return sampled.reset_index(drop=True)


def _extract_shap_values_for_positive_class(raw_shap_values: Any, positive_class_index: int) -> np.ndarray:
    if isinstance(raw_shap_values, list):
        values = np.asarray(raw_shap_values[positive_class_index], dtype=float)
    else:
        values = np.asarray(raw_shap_values, dtype=float)
        if values.ndim == 3:
            if values.shape[-1] > positive_class_index:
                values = values[:, :, positive_class_index]
            elif values.shape[0] > positive_class_index:
                values = values[positive_class_index, :, :]
            else:
                raise ValueError(
                    "Unexpected SHAP value shape for class extraction: "
                    f"{values.shape!r} (positive_class_index={positive_class_index})."
                )
    if values.ndim == 1:
        values = values.reshape(1, -1)
    if values.ndim != 2:
        raise ValueError(f"Expected SHAP values as 2D array, got shape {values.shape!r}.")
    return values


def _extract_expected_value_for_positive_class(expected_value: Any, positive_class_index: int) -> float:
    if isinstance(expected_value, (list, tuple, np.ndarray)):
        expected_array = np.asarray(expected_value, dtype=float).reshape(-1)
        if expected_array.size == 1:
            return float(expected_array[0])
        if expected_array.size > positive_class_index:
            return float(expected_array[positive_class_index])
    return float(expected_value)


def _resolve_estimator_components(estimator: Any) -> tuple[Any | None, Any]:
    if hasattr(estimator, "named_steps") and "clf" in estimator.named_steps:
        preprocess = estimator.named_steps.get("preprocess")
        classifier = estimator.named_steps["clf"]
        return preprocess, classifier
    return None, estimator


def _resolve_explainer_kind(classifier: Any) -> str:
    classifier_name = classifier.__class__.__name__.lower()
    tree_tokens = ("tree", "forest", "boosting", "xgb", "lgbm", "catboost")
    linear_tokens = ("logisticregression", "linear", "ridgeclassifier", "sgdclassifier")
    if any(token in classifier_name for token in tree_tokens):
        return "tree"
    if any(token in classifier_name for token in linear_tokens):
        return "linear"
    return "kernel"


def predict_and_explain_single_row(
    deployed_models: Mapping[str, Mapping[str, Any]],
    *,
    deploy_id: str,
    row: pd.Series | pd.DataFrame | Mapping[str, Any],
    background_data: pd.DataFrame,
    feature_columns: Sequence[str] = FINAL_FEATURE_COLUMNS,
    max_background_rows: int = 200,
    random_state: int = 42,
    kernel_nsamples: int = 200,
    make_waterfall_plot: bool = True,
    waterfall_max_display: int = 12,
    show_waterfall: bool = False,
    explainer_cache: dict[tuple[str, str], Any] | None = None,
) -> dict[str, Any]:
    """
    Predict and compute SHAP explanation for a single row from a deployed model.

    Returns prediction metadata, SHAP values, and optional waterfall figure objects.
    """
    try:
        import shap
    except ImportError as exc:
        raise ImportError(
            "SHAP is required for prediction explanations. Install it with: `pip install shap`."
        ) from exc

    if deploy_id not in deployed_models:
        available = ", ".join(sorted(deployed_models.keys()))
        raise KeyError(f"`deploy_id` {deploy_id!r} not found. Available: {available}")

    spec = deployed_models[deploy_id]
    tuned_model = spec["model"]
    base_estimator = tuned_model.estimator_

    selected_columns = list(feature_columns)
    row_df = _to_single_row_dataframe(row, feature_columns=selected_columns)
    background_df = _sample_background_data(
        background_data,
        feature_columns=selected_columns,
        max_background_rows=max_background_rows,
        random_state=random_state,
    )

    threshold = float(spec.get("threshold", getattr(tuned_model, "best_threshold_", 0.5)))
    tuned_classes = np.asarray(tuned_model.classes_)
    positive_label = _resolve_positive_label(tuned_classes)
    negative_label = _resolve_negative_label(tuned_classes, positive_label)

    if not hasattr(base_estimator, "predict_proba"):
        raise ValueError("SHAP prediction explanation currently expects estimators with `predict_proba`.")

    if not hasattr(base_estimator, "classes_"):
        raise ValueError("Estimator is missing `classes_`; cannot resolve positive-class score index.")
    estimator_positive_class_index = _positive_class_index(base_estimator.classes_, positive_label)
    estimator_negative_class_index = _positive_class_index(base_estimator.classes_, negative_label)
    score_matrix = base_estimator.predict_proba(row_df)
    if score_matrix.ndim != 2 or score_matrix.shape[0] != 1:
        raise ValueError(f"Unexpected predict_proba output shape: {score_matrix.shape!r}")
    score = float(score_matrix[0, estimator_positive_class_index])
    probability_positive = score
    probability_negative = float(score_matrix[0, estimator_negative_class_index])
    prediction = positive_label if score >= threshold else negative_label

    preprocess, classifier = _resolve_estimator_components(base_estimator)
    if preprocess is not None:
        x_row_model = preprocess.transform(row_df)
        x_background_model = preprocess.transform(background_df)
    else:
        x_row_model = row_df.to_numpy()
        x_background_model = background_df.to_numpy()

    if not hasattr(classifier, "predict_proba"):
        raise ValueError("SHAP prediction explanation currently expects classifiers with `predict_proba`.")
    classifier_classes = np.asarray(classifier.classes_)
    classifier_positive_class_index = _positive_class_index(classifier_classes, positive_label)

    explainer_kind = _resolve_explainer_kind(classifier)
    cache_key = (deploy_id, explainer_kind)
    explainer = explainer_cache.get(cache_key) if explainer_cache is not None else None

    if explainer is None:
        try:
            if explainer_kind == "tree":
                explainer = shap.TreeExplainer(
                    classifier,
                    data=x_background_model,
                    model_output="probability",
                )
            elif explainer_kind == "linear":
                explainer = shap.LinearExplainer(classifier, x_background_model)
            else:
                predict_positive = (
                    lambda matrix: classifier.predict_proba(matrix)[:, classifier_positive_class_index]
                )
                explainer = shap.KernelExplainer(predict_positive, x_background_model)
        except Exception:
            explainer_kind = "kernel"
            predict_positive = (
                lambda matrix: classifier.predict_proba(matrix)[:, classifier_positive_class_index]
            )
            explainer = shap.KernelExplainer(predict_positive, x_background_model)
        if explainer_cache is not None:
            explainer_cache[cache_key] = explainer

    if explainer_kind == "kernel":
        raw_shap_values = explainer.shap_values(x_row_model, nsamples=kernel_nsamples)
    else:
        raw_shap_values = explainer.shap_values(x_row_model)

    shap_values_2d = _extract_shap_values_for_positive_class(
        raw_shap_values,
        classifier_positive_class_index,
    )
    base_value = _extract_expected_value_for_positive_class(
        explainer.expected_value,
        classifier_positive_class_index,
    )
    shap_values_1d = shap_values_2d[0]

    row_display_values = row_df.iloc[0].to_numpy(dtype=float, copy=False)
    explanation = shap.Explanation(
        values=shap_values_1d,
        base_values=base_value,
        data=row_display_values,
        feature_names=selected_columns,
    )
    shap_series = (
        pd.Series(shap_values_1d, index=selected_columns, name="shap_value")
        .sort_values(key=lambda series: series.abs(), ascending=False)
    )

    waterfall_figure = None
    waterfall_axis = None
    if make_waterfall_plot:
        import matplotlib.pyplot as plt

        plt.figure(figsize=(9, 5))
        waterfall_axis = shap.plots.waterfall(
            explanation,
            max_display=waterfall_max_display,
            show=show_waterfall,
        )
        waterfall_figure = plt.gcf()

    return {
        "deploy_id": deploy_id,
        "model_name": spec["model_name"],
        "profile": spec["profile"],
        "threshold": threshold,
        "score": score,
        "probability_positive": probability_positive,
        "probability_negative": probability_negative,
        "probabilities_by_class": {
            str(negative_label): probability_negative,
            str(positive_label): probability_positive,
        },
        "prediction": prediction,
        "positive_label": positive_label,
        "negative_label": negative_label,
        "explainer_kind": explainer_kind,
        "feature_values": row_df.iloc[0].copy(),
        "shap_values": shap_series,
        "base_value": base_value,
        "explanation": explanation,
        "waterfall_figure": waterfall_figure,
        "waterfall_axis": waterfall_axis,
    }


__all__ = [
    "build_evaluation_sets",
    "build_feature_importance_matrix",
    "compute_or_load_permutation_importance",
    "get_feature_analysis_paths",
    "load_deployed_models",
    "predict_and_explain_single_row",
]
