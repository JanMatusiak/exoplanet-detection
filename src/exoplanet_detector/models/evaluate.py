"""Model-evaluation helpers for threshold tuning and deployment artifacts."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from sklearn import set_config
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    fbeta_score,
    make_scorer,
    precision_score,
    recall_score,
)
from sklearn.model_selection import RandomizedSearchCV, StratifiedGroupKFold, TunedThresholdClassifierCV

from exoplanet_detector.config import MIN_PRECISION_FLOOR, MIN_RECALL_FLOOR

ThresholdModels = dict[str, dict[str, TunedThresholdClassifierCV]]

RAW_TO_DISPLAY_PROFILE = {
    "best_f2": "f2",
    "best_recall_pmin_0_5": "recall_constrained",
    "best_precision_rmin_0_5": "precision_constrained",
}
DISPLAY_TO_RAW_PROFILE = {value: key for key, value in RAW_TO_DISPLAY_PROFILE.items()}


def build_threshold_scoring_profiles(
    *,
    min_precision: float = MIN_PRECISION_FLOOR,
    min_recall: float = MIN_RECALL_FLOOR,
) -> dict[str, Any]:
    """Build threshold-objective scorers used by TunedThresholdClassifierCV."""

    def recall_with_precision_floor(y_true, y_pred):
        precision = precision_score(y_true, y_pred, zero_division=0)
        if precision < min_precision:
            return 0.0
        return recall_score(y_true, y_pred, zero_division=0)

    def precision_with_recall_floor(y_true, y_pred):
        recall = recall_score(y_true, y_pred, zero_division=0)
        if recall < min_recall:
            return 0.0
        return precision_score(y_true, y_pred, zero_division=0)

    return {
        "best_f2": make_scorer(fbeta_score, beta=2, zero_division=0),
        "best_recall_pmin_0_5": make_scorer(recall_with_precision_floor),
        "best_precision_rmin_0_5": make_scorer(precision_with_recall_floor),
    }


def fit_or_load_threshold_models(
    search_results: Mapping[str, RandomizedSearchCV],
    x_train: pd.DataFrame,
    y_train: pd.Series,
    groups: pd.Series,
    *,
    artifact_dir: str | Path,
    scoring_profiles: Mapping[str, Any],
    candidate_model_names: Sequence[str] | None = None,
    force_retune: bool = False,
    n_splits: int = 5,
    cv_random_state: int = 42,
    thresholds: int | Sequence[float] = 100,
    n_jobs: int = -1,
    response_method: str = "auto",
    refit: bool = True,
    tuner_random_state: int = 42,
    store_cv_results: bool = True,
) -> tuple[ThresholdModels, pd.DataFrame]:
    """Fit threshold wrappers or load them from cache."""
    artifact_root = Path(artifact_dir)
    artifact_root.mkdir(parents=True, exist_ok=True)

    threshold_summary_path = artifact_root / "threshold_tuning_summary.csv"
    tuned_models_path = artifact_root / "tuned_threshold_models.joblib"

    if tuned_models_path.exists() and threshold_summary_path.exists() and not force_retune:
        try:
            loaded_models = joblib.load(tuned_models_path)
            threshold_summary = pd.read_csv(threshold_summary_path)
            return loaded_models, threshold_summary
        except Exception:
            # Cached wrappers may become unpicklable after scorer/function refactors.
            # In that case, fall back to refitting with current code and overwrite cache.
            pass

    set_config(enable_metadata_routing=True)
    cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=cv_random_state)
    selected_names = list(candidate_model_names) if candidate_model_names is not None else list(search_results.keys())

    tuned_threshold_models: ThresholdModels = {}
    summary_rows: list[dict[str, Any]] = []

    for model_name in selected_names:
        base_estimator = search_results[model_name].best_estimator_
        tuned_threshold_models[model_name] = {}

        for profile_name, profile_scorer in scoring_profiles.items():
            tuned = TunedThresholdClassifierCV(
                estimator=base_estimator,
                scoring=profile_scorer,
                response_method=response_method,
                thresholds=thresholds,
                cv=cv,
                n_jobs=n_jobs,
                refit=refit,
                random_state=tuner_random_state,
                store_cv_results=store_cv_results,
            )
            tuned.fit(x_train, y_train, groups=groups)

            tuned_threshold_models[model_name][profile_name] = tuned
            summary_rows.append(
                {
                    "model": model_name,
                    "profile": profile_name,
                    "best_threshold": float(tuned.best_threshold_),
                    "best_score": float(tuned.best_score_),
                }
            )

    threshold_summary = (
        pd.DataFrame(summary_rows)
        .sort_values(["model", "profile"], ascending=[True, True])
        .reset_index(drop=True)
    )

    joblib.dump(tuned_threshold_models, tuned_models_path, compress=3)
    threshold_summary.to_csv(threshold_summary_path, index=False)
    return tuned_threshold_models, threshold_summary


def _get_positive_scores(estimator: Any, x: pd.DataFrame) -> Any:
    if hasattr(estimator, "predict_proba"):
        return estimator.predict_proba(x)[:, 1]
    return estimator.decision_function(x)


def evaluate_or_load_threshold_models(
    tuned_threshold_models: Mapping[str, Mapping[str, TunedThresholdClassifierCV]],
    x_test: pd.DataFrame,
    y_test: pd.Series,
    x_k2p: pd.DataFrame,
    y_k2p: pd.Series,
    *,
    artifact_dir: str | Path,
    force_reevaluate: bool = False,
    include_combined: bool = True,
    profile_label_map: Mapping[str, str] = RAW_TO_DISPLAY_PROFILE,
) -> pd.DataFrame:
    """Evaluate threshold wrappers on KOI/K2P and optionally combined data."""
    artifact_root = Path(artifact_dir)
    artifact_root.mkdir(parents=True, exist_ok=True)
    comparison_path = artifact_root / "comparison_df.csv"

    if comparison_path.exists() and not force_reevaluate:
        return pd.read_csv(comparison_path)

    evaluation_rows: list[dict[str, Any]] = []
    datasets: list[tuple[str, pd.DataFrame, pd.Series]] = [
        ("KOI_test", x_test, y_test),
        ("K2P", x_k2p, y_k2p),
    ]
    if include_combined:
        x_combined = pd.concat([x_test, x_k2p], axis=0, ignore_index=True)
        y_combined = pd.concat([y_test, y_k2p], axis=0, ignore_index=True)
        datasets.append(("KOI_test_plus_K2P", x_combined, y_combined))

    for model_name, profiles in tuned_threshold_models.items():
        for raw_profile_name, tuned_model in profiles.items():
            profile_name = profile_label_map[raw_profile_name]
            threshold = float(tuned_model.best_threshold_)

            for dataset_name, x_eval, y_eval in datasets:
                y_pred = tuned_model.predict(x_eval)
                y_score = _get_positive_scores(tuned_model, x_eval)
                tn, fp, fn, tp = confusion_matrix(y_eval, y_pred, labels=[0, 1]).ravel()
                try:
                    pr_auc = float(average_precision_score(y_eval, y_score))
                except ValueError:
                    pr_auc = float("nan")

                evaluation_rows.append(
                    {
                        "model": model_name,
                        "profile": profile_name,
                        "threshold": threshold,
                        "recall": float(recall_score(y_eval, y_pred, zero_division=0)),
                        "precision": float(precision_score(y_eval, y_pred, zero_division=0)),
                        "f2": float(fbeta_score(y_eval, y_pred, beta=2, zero_division=0)),
                        "pr_auc": pr_auc,
                        "tn": int(tn),
                        "fp": int(fp),
                        "fn": int(fn),
                        "tp": int(tp),
                        "dataset": dataset_name,
                    }
                )

    comparison_df = pd.DataFrame(evaluation_rows)
    comparison_df.to_csv(comparison_path, index=False)
    return comparison_df


def format_comparison_table(comparison_df: pd.DataFrame) -> pd.DataFrame:
    """Apply the notebook-standard columns and ordering for comparison display."""
    ordered = comparison_df[
        [
            "model",
            "profile",
            "threshold",
            "recall",
            "precision",
            "f2",
            "pr_auc",
            "tn",
            "fp",
            "fn",
            "tp",
            "dataset",
        ]
    ].copy()
    return (
        ordered.sort_values(["dataset", "profile", "f2"], ascending=[True, True, False])
        .reset_index(drop=True)
    )


def save_deployment_bundle(
    tuned_threshold_models: Mapping[str, Mapping[str, TunedThresholdClassifierCV]],
    comparison_df: pd.DataFrame,
    deploy_selection: Sequence[Mapping[str, str]],
    *,
    output_dir: str | Path,
    profile_to_raw: Mapping[str, str] = DISPLAY_TO_RAW_PROFILE,
) -> tuple[Path, Path, pd.DataFrame]:
    """Save selected deployment models and a manifest table."""
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)

    deploy_bundle: dict[str, Any] = {}
    manifest_rows: list[dict[str, Any]] = []

    for item in deploy_selection:
        deploy_id = item["deploy_id"]
        model_name = item["model"]
        profile_name = item["profile"]
        raw_profile = profile_to_raw[profile_name]
        tuned_model = tuned_threshold_models[model_name][raw_profile]
        threshold = float(tuned_model.best_threshold_)

        deploy_bundle[deploy_id] = {
            "model_name": model_name,
            "profile": profile_name,
            "threshold": threshold,
            "model": tuned_model,
        }

        metric_rows = comparison_df[
            (comparison_df["model"] == model_name)
            & (comparison_df["profile"] == profile_name)
            & (comparison_df["dataset"].isin(["KOI_test", "K2P", "KOI_test_plus_K2P"]))
        ]

        metrics_by_dataset: dict[str, dict[str, Any]] = {}
        for _, row in metric_rows.iterrows():
            metrics_by_dataset[row["dataset"]] = {
                "recall": float(row["recall"]),
                "precision": float(row["precision"]),
                "f2": float(row["f2"]),
                "pr_auc": float(row["pr_auc"]),
                "tn": int(row["tn"]),
                "fp": int(row["fp"]),
                "fn": int(row["fn"]),
                "tp": int(row["tp"]),
            }

        deploy_bundle[deploy_id]["metrics"] = metrics_by_dataset
        manifest_rows.append(
            {
                "deploy_id": deploy_id,
                "model": model_name,
                "profile": profile_name,
                "threshold": threshold,
                "koi_test_f2": metrics_by_dataset.get("KOI_test", {}).get("f2", float("nan")),
                "koi_test_recall": metrics_by_dataset.get("KOI_test", {}).get("recall", float("nan")),
                "koi_test_precision": metrics_by_dataset.get("KOI_test", {}).get("precision", float("nan")),
                "k2p_f2": metrics_by_dataset.get("K2P", {}).get("f2", float("nan")),
                "k2p_recall": metrics_by_dataset.get("K2P", {}).get("recall", float("nan")),
                "k2p_precision": metrics_by_dataset.get("K2P", {}).get("precision", float("nan")),
            }
        )

    models_path = destination / "deploy_models.joblib"
    manifest_path = destination / "deploy_manifest.csv"
    joblib.dump(deploy_bundle, models_path, compress=3)

    manifest_df = pd.DataFrame(manifest_rows)
    manifest_df.to_csv(manifest_path, index=False)
    return models_path, manifest_path, manifest_df


__all__ = [
    "DISPLAY_TO_RAW_PROFILE",
    "RAW_TO_DISPLAY_PROFILE",
    "build_threshold_scoring_profiles",
    "evaluate_or_load_threshold_models",
    "fit_or_load_threshold_models",
    "format_comparison_table",
    "save_deployment_bundle",
]
