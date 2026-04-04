from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from exoplanet_detector.app import services
from exoplanet_detector.features.feature_selection import FINAL_FEATURE_COLUMNS

pytestmark = pytest.mark.integration


def _feature_frame(n_rows: int = 1) -> pd.DataFrame:
    """Build a deterministic feature-only dataframe with FINAL_FEATURE_COLUMNS."""
    return pd.DataFrame(
        {
            feature: [float(index + row + 1) for row in range(n_rows)]
            for index, feature in enumerate(FINAL_FEATURE_COLUMNS)
        }
    )


def test_get_run_context_loads_artifacts_and_feature_background(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Load run context from temporary artifacts and verify background/features wiring."""
    run_tag = "test_run"
    artifact_root = tmp_path / "artifacts"
    deployment_dir = artifact_root / "deployment" / run_tag
    evaluation_dir = artifact_root / "evaluation" / run_tag
    visualization_dir = artifact_root / "visualization" / run_tag
    feature_analysis_dir = artifact_root / "feature_analysis" / run_tag
    model_search_dir = artifact_root / "model_search" / run_tag

    # Create expected artifact directory layout for one run tag.
    for directory in (
        deployment_dir,
        evaluation_dir,
        visualization_dir,
        feature_analysis_dir,
        model_search_dir,
    ):
        directory.mkdir(parents=True, exist_ok=True)

    # Minimal artifact files read by get_run_context.
    (deployment_dir / "deploy_models.joblib").write_bytes(b"stub")
    pd.DataFrame(
        [
            {
                "deploy_id": "demo",
                "model": "dummy",
                "profile": "f2",
                "threshold": 0.5,
            }
        ]
    ).to_csv(deployment_dir / "deploy_manifest.csv", index=False)
    pd.DataFrame([{"model": "dummy", "metric": 0.9}]).to_csv(
        evaluation_dir / "comparison_df.csv",
        index=False,
    )
    pd.DataFrame(
        [
            {
                "deploy_id": "demo",
                "dataset": "KOI_test",
                "confusion_matrix_path": "cm.png",
                "roc_curve_path": "roc.png",
                "pr_curve_path": "pr.png",
            }
        ]
    ).to_csv(visualization_dir / "plot_manifest.csv", index=False)
    pd.DataFrame(
        [
            {
                "deploy_id": "demo",
                "dataset": "KOI_test",
                "feature": FINAL_FEATURE_COLUMNS[0],
                "importance_mean": 0.1,
                "importance_std": 0.01,
                "importance_rank": 1,
            }
        ]
    ).to_csv(feature_analysis_dir / "permutation_importance.csv", index=False)
    pd.DataFrame(
        [{"deploy_id": "demo", "dataset": "KOI_test", FINAL_FEATURE_COLUMNS[0]: 0.1}]
    ).to_csv(feature_analysis_dir / "feature_importance_matrix.csv", index=False)

    data_dir = tmp_path / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    background_path = data_dir / "KOI_test_set.csv"
    example_path = data_dir / "K2P_labeled_set.csv"
    _feature_frame(n_rows=2).to_csv(background_path, index=False)
    example_df = _feature_frame(n_rows=1)
    example_df.insert(0, "label", 1)
    example_df.insert(0, "group_id", 1)
    example_df.to_csv(example_path, index=False)

    fake_models = {
        "demo": {
            "model": object(),
            "model_name": "Dummy",
            "profile": "f2",
            "threshold": 0.5,
        }
    }
    fake_registry = pd.DataFrame(
        [{"deploy_id": "demo", "model_name": "Dummy", "profile": "f2", "threshold": 0.5}]
    )

    # Redirect service dependencies to temporary paths and lightweight stubs.
    monkeypatch.setattr(services, "ARTIFACTS_DIR", artifact_root)
    monkeypatch.setattr(services, "KOI_TEST_FILE", background_path)
    monkeypatch.setattr(services, "K2P_FILE", example_path)
    monkeypatch.setattr(services, "load_deployed_models", lambda _: (fake_models, fake_registry))
    monkeypatch.setattr(
        services,
        "get_run_artifact_dirs",
        lambda run_tag, create=False: {
            "model_search": model_search_dir,
            "evaluation": evaluation_dir,
            "deployment": deployment_dir,
        },
    )

    context = services.get_run_context(
        run_tag=run_tag,
        example_dataset="K2P_labeled",
        background_dataset="KOI_test",
    )

    # Core context invariants.
    assert context["run_tag"] == run_tag
    assert set(context["deployed_models"]) == {"demo"}
    assert not context["deploy_manifest_df"].empty
    assert not context["comparison_df"].empty
    assert context["background_dataset_name"] == "KOI_test"
    assert list(context["background_df"].columns) == list(FINAL_FEATURE_COLUMNS)
    assert context["background_df"].shape == (2, len(FINAL_FEATURE_COLUMNS))
