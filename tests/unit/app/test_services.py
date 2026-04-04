from __future__ import annotations

import pytest

from exoplanet_detector.app import services
from exoplanet_detector.config import K2P_FILE
from exoplanet_detector.features.feature_selection import FINAL_FEATURE_COLUMNS

pytestmark = pytest.mark.unit


def _valid_feature_values() -> dict[str, float]:
    """Return a fully valid payload keyed by FINAL_FEATURE_COLUMNS."""
    return {feature: 1.0 for feature in FINAL_FEATURE_COLUMNS}


def test_dataset_path_by_name_supports_alias_and_rejects_unknown() -> None:
    """Resolve supported dataset aliases and fail fast on unsupported names."""
    assert services._dataset_path_by_name("K2P") == K2P_FILE
    assert services._dataset_path_by_name(" K2P_labeled ") == K2P_FILE

    with pytest.raises(ValueError, match="Unsupported dataset name"):
        services._dataset_path_by_name("not_a_dataset")


def test_validate_inputs_collects_errors_non_strict_and_raises_in_strict() -> None:
    """Collect validation issues in non-strict mode and raise aggregated errors in strict mode."""
    values = _valid_feature_values()
    values.pop("orbital_period_days")
    values["transit_depth"] = "bad-number"
    values["impact_parameter"] = -1.0
    values["stellar_logg_cgs"] = 7.0

    cleaned, errors = services.validate_inputs(values, strict=False)

    assert any("Missing required feature: orbital_period_days" in error for error in errors)
    assert any("transit_depth: expected numeric value" in error for error in errors)
    assert any("impact_parameter: -1 is below minimum 0" in error for error in errors)
    assert any("stellar_logg_cgs: 7 is above maximum 6" in error for error in errors)
    assert cleaned["impact_parameter"] == -1.0
    assert cleaned["stellar_logg_cgs"] == 7.0

    with pytest.raises(ValueError, match="Missing required feature: orbital_period_days"):
        services.validate_inputs(values, strict=True)
