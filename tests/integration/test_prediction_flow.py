from __future__ import annotations

import numpy as np
import pytest

from exoplanet_detector.app import services
from exoplanet_detector.features.feature_selection import FINAL_FEATURE_COLUMNS

pytestmark = pytest.mark.integration


class _DummyEstimator:
    """Simple deterministic estimator returning one fixed probability row."""
    classes_ = np.array([0, 1])

    def predict_proba(self, x_row):
        assert list(x_row.columns) == list(FINAL_FEATURE_COLUMNS)
        return np.array([[0.30, 0.70]])


class _DummyTunedModel:
    """Wrapper matching the tuned-model shape expected by services.predict."""
    classes_ = np.array([0, 1])
    estimator_ = _DummyEstimator()


def _feature_values() -> dict[str, float]:
    """Create a valid feature payload for the predictor service."""
    return {feature: 1.0 for feature in FINAL_FEATURE_COLUMNS}


def test_predict_returns_thresholded_binary_output() -> None:
    """Return prediction payload with thresholded class decisions from predict_proba."""
    model_spec = {
        "model": _DummyTunedModel(),
        "model_name": "DummyModel",
        "profile": "f2",
        "threshold": 0.5,
    }
    context = {"deployed_models": {"demo": model_spec}}

    prediction = services.predict(context, deploy_id="demo", feature_values=_feature_values())

    assert prediction["deploy_id"] == "demo"
    assert prediction["prediction"] == 1
    assert prediction["probability_positive"] == pytest.approx(0.70)
    assert prediction["probability_negative"] == pytest.approx(0.30)
    assert prediction["threshold"] == pytest.approx(0.5)

    model_spec_high_threshold = dict(model_spec)
    model_spec_high_threshold["threshold"] = 0.8
    context_high_threshold = {"deployed_models": {"demo": model_spec_high_threshold}}
    prediction_high_threshold = services.predict(
        context_high_threshold,
        deploy_id="demo",
        feature_values=_feature_values(),
    )
    assert prediction_high_threshold["prediction"] == 0
