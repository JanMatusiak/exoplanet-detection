"""Model training and tuning utilities."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from scipy.stats import loguniform, randint
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import fbeta_score, make_scorer
from sklearn.model_selection import RandomizedSearchCV, StratifiedGroupKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC

from exoplanet_detector.features.pipelines import build_preprocessing_pipeline

SearchResults = dict[str, RandomizedSearchCV]


def get_default_model_specs(*, random_state: int = 42) -> dict[str, dict[str, Any]]:
    """Return the default 5-model tuning registry used in notebook 03."""
    return {
        "logreg": {
            "estimator": LogisticRegression(max_iter=5000, class_weight="balanced"),
            "with_scaling": True,
            "params": {"clf__C": loguniform(1e-3, 1e2)},
        },
        "svc_rbf": {
            "estimator": SVC(probability=True, class_weight="balanced"),
            "with_scaling": True,
            "params": {
                "clf__C": loguniform(1e-2, 1e2),
                "clf__gamma": loguniform(1e-4, 1e-1),
            },
        },
        "knn": {
            "estimator": KNeighborsClassifier(),
            "with_scaling": True,
            "params": {
                "clf__n_neighbors": randint(3, 41),
                "clf__weights": ["uniform", "distance"],
            },
        },
        "rf": {
            "estimator": RandomForestClassifier(
                random_state=random_state,
                n_jobs=-1,
                class_weight="balanced",
            ),
            "with_scaling": False,
            "params": {
                "clf__n_estimators": randint(200, 800),
                "clf__max_depth": [None, 5, 10, 20],
            },
        },
        "hgb": {
            "estimator": HistGradientBoostingClassifier(random_state=random_state),
            "with_scaling": False,
            "params": {
                "clf__learning_rate": loguniform(1e-3, 1e-1),
                "clf__max_leaf_nodes": randint(15, 63),
            },
        },
    }


def get_default_scoring() -> dict[str, str | Any]:
    """Return multi-metric scoring with recall-priority via F2."""
    return {
        "recall": "recall",
        "precision": "precision",
        "f2": make_scorer(fbeta_score, beta=2),
        "pr_auc": "average_precision",
    }


def run_model_searches(
    x_train: pd.DataFrame,
    y_train: pd.Series,
    groups: pd.Series,
    *,
    model_specs: Mapping[str, Mapping[str, Any]] | None = None,
    scoring: Mapping[str, str | Any] | None = None,
    n_splits: int = 5,
    cv_random_state: int = 42,
    n_iter: int = 25,
    n_jobs: int = -1,
    search_random_state: int = 42,
    refit_metric: str = "f2",
    verbose: int = 1,
    artifact_path: str | Path | None = None,
    force_retrain: bool = False,
) -> SearchResults:
    """
    Train or load cached RandomizedSearchCV runs for each model spec.

    If `artifact_path` exists and `force_retrain` is False, cached searches are loaded.
    Otherwise, all model searches are fit and optionally persisted to disk.
    """
    specs = dict(model_specs) if model_specs is not None else get_default_model_specs()
    score_map = dict(scoring) if scoring is not None else get_default_scoring()

    resolved_artifact_path: Path | None = None
    if artifact_path is not None:
        resolved_artifact_path = Path(artifact_path)
        if resolved_artifact_path.exists() and not force_retrain:
            loaded = joblib.load(resolved_artifact_path)
            if not isinstance(loaded, dict):
                raise TypeError(
                    f"Expected cached search results dict at {resolved_artifact_path}, got {type(loaded)}."
                )
            return loaded

    cv = StratifiedGroupKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=cv_random_state,
    )
    search_results: SearchResults = {}

    for model_name, spec in specs.items():
        pipeline = Pipeline(
            [
                (
                    "preprocess",
                    build_preprocessing_pipeline(with_scaling=bool(spec["with_scaling"])),
                ),
                ("clf", spec["estimator"]),
            ]
        )

        search = RandomizedSearchCV(
            estimator=pipeline,
            param_distributions=spec["params"],
            n_iter=n_iter,
            scoring=score_map,
            cv=cv,
            n_jobs=n_jobs,
            random_state=search_random_state,
            refit=refit_metric,
            verbose=verbose,
        )
        search.fit(x_train, y_train, groups=groups)
        search_results[model_name] = search

    if resolved_artifact_path is not None:
        resolved_artifact_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(search_results, resolved_artifact_path, compress=3)

    return search_results


def build_search_summary(
    search_results: Mapping[str, RandomizedSearchCV],
    *,
    refit_metric: str = "f2",
) -> pd.DataFrame:
    """Build a per-model summary from fitted RandomizedSearchCV objects."""
    summary_rows: list[dict[str, Any]] = []
    refit_score_column = f"best_score_refit_{refit_metric}"

    for model_name, search in search_results.items():
        best_idx = search.best_index_
        results = search.cv_results_

        recall_vals = results.get("mean_test_recall")
        precision_vals = results.get("mean_test_precision")
        f2_vals = results.get("mean_test_f2")
        pr_auc_vals = results.get("mean_test_pr_auc")

        summary_rows.append(
            {
                "model": model_name,
                refit_score_column: search.best_score_,
                "best_mean_test_recall": (
                    recall_vals[best_idx] if recall_vals is not None else float("nan")
                ),
                "best_mean_test_precision": (
                    precision_vals[best_idx] if precision_vals is not None else float("nan")
                ),
                "best_mean_test_f2": f2_vals[best_idx] if f2_vals is not None else float("nan"),
                "best_mean_test_pr_auc": (
                    pr_auc_vals[best_idx] if pr_auc_vals is not None else float("nan")
                ),
                "best_params": search.best_params_,
            }
        )

    summary = pd.DataFrame(summary_rows)
    if summary.empty:
        return summary
    return summary.sort_values(refit_score_column, ascending=False).reset_index(drop=True)
