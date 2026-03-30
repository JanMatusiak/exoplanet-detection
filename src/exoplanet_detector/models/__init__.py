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
from exoplanet_detector.models.feature_analysis import (
    build_evaluation_sets,
    build_feature_importance_matrix,
    compute_or_load_permutation_importance,
    get_feature_analysis_paths,
    load_deployed_models,
)

__all__ = [
    "build_search_summary",
    "build_threshold_scoring_profiles",
    "evaluate_or_load_threshold_models",
    "fit_or_load_threshold_models",
    "format_comparison_table",
    "get_default_model_specs",
    "get_default_scoring",
    "get_feature_analysis_paths",
    "load_deployed_models",
    "build_evaluation_sets",
    "compute_or_load_permutation_importance",
    "build_feature_importance_matrix",
    "run_model_searches",
    "save_deployment_bundle",
    "RAW_TO_DISPLAY_PROFILE",
    "DISPLAY_TO_RAW_PROFILE",
]
