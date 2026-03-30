"""Feature-importance analysis helpers for deployed models."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
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


__all__ = [
    "build_evaluation_sets",
    "build_feature_importance_matrix",
    "compute_or_load_permutation_importance",
    "get_feature_analysis_paths",
    "load_deployed_models",
]
