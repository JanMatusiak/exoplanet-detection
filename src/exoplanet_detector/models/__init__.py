"""Model training, tuning, and evaluation helpers."""

from exoplanet_detector.models.train import (
    build_search_summary,
    get_default_model_specs,
    get_default_scoring,
    run_model_searches,
)
from exoplanet_detector.models.evaluate import (
    DISPLAY_TO_RAW_PROFILE,
    RAW_TO_DISPLAY_PROFILE,
    build_threshold_scoring_profiles,
    evaluate_or_load_threshold_models,
    fit_or_load_threshold_models,
    format_comparison_table,
    save_deployment_bundle,
)

__all__ = [
    "build_search_summary",
    "build_threshold_scoring_profiles",
    "evaluate_or_load_threshold_models",
    "fit_or_load_threshold_models",
    "format_comparison_table",
    "get_default_model_specs",
    "get_default_scoring",
    "run_model_searches",
    "save_deployment_bundle",
    "RAW_TO_DISPLAY_PROFILE",
    "DISPLAY_TO_RAW_PROFILE",
]
