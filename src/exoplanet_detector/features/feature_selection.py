"""Feature and schema definitions extracted from notebook decisions."""

from __future__ import annotations

from typing import Sequence

KOI_COLUMNS_SET = (
    "kepid",
    "koi_disposition",
    "koi_period",
    "koi_eccen",
    "koi_longp",
    "koi_impact",
    "koi_duration",
    "koi_depth",
    "koi_ror",
    "koi_incl",
    "koi_dor",
    "koi_prad",
    "koi_sma",
    "koi_teq",
    "koi_insol",
    "koi_steff",
    "koi_slogg",
    "koi_smet",
    "koi_srad",
    "koi_smass",
    "koi_sage",
)

K2P_COLUMNS_SET = (
    "pl_name",
    "disposition",
    "pl_orbper",
    "pl_orbeccen",
    "pl_orblper",
    "pl_imppar",
    "pl_trandur",
    "pl_trandep",
    "pl_ratror",
    "pl_orbincl",
    "pl_ratdor",
    "pl_rade",
    "pl_orbsmax",
    "pl_eqt",
    "pl_insol",
    "st_teff",
    "st_logg",
    "st_met",
    "st_rad",
    "st_mass",
    "st_age",
)

KOI_RENAME_MAP = {
    "kepid": "group_id",
    "koi_disposition": "label",
    "koi_period": "orbital_period_days",
    "koi_eccen": "eccentricity",
    "koi_longp": "arg_periastron_deg",
    "koi_impact": "impact_parameter",
    "koi_duration": "transit_duration_hours",
    "koi_depth": "transit_depth",
    "koi_ror": "radius_ratio_rp_rs",
    "koi_incl": "inclination_deg",
    "koi_dor": "a_over_rs",
    "koi_prad": "planet_radius_rearth",
    "koi_sma": "semi_major_axis_au",
    "koi_teq": "equilibrium_temp_k",
    "koi_insol": "insolation_earth",
    "koi_steff": "stellar_teff_k",
    "koi_slogg": "stellar_logg_cgs",
    "koi_smet": "stellar_metallicity_dex",
    "koi_srad": "stellar_radius_rsun",
    "koi_smass": "stellar_mass_msun",
    "koi_sage": "stellar_age_gyr",
}

K2P_RENAME_MAP = {
    "pl_name": "group_id",
    "disposition": "label",
    "pl_orbper": "orbital_period_days",
    "pl_orbeccen": "eccentricity",
    "pl_orblper": "arg_periastron_deg",
    "pl_imppar": "impact_parameter",
    "pl_trandur": "transit_duration_hours",
    "pl_trandep": "transit_depth",
    "pl_ratror": "radius_ratio_rp_rs",
    "pl_orbincl": "inclination_deg",
    "pl_ratdor": "a_over_rs",
    "pl_rade": "planet_radius_rearth",
    "pl_orbsmax": "semi_major_axis_au",
    "pl_eqt": "equilibrium_temp_k",
    "pl_insol": "insolation_earth",
    "st_teff": "stellar_teff_k",
    "st_logg": "stellar_logg_cgs",
    "st_met": "stellar_metallicity_dex",
    "st_rad": "stellar_radius_rsun",
    "st_mass": "stellar_mass_msun",
    "st_age": "stellar_age_gyr",
}

# Dropped in notebook 01 due high missingness.
BASE_DROP_COLUMNS = ("arg_periastron_deg", "stellar_age_gyr")

# Dropped in notebook 02 based on low variance / correlation analysis.
ANALYSIS_DROP_COLUMNS = (
    "eccentricity",
    "semi_major_axis_au",
    "equilibrium_temp_k",
    "radius_ratio_rp_rs",
    "stellar_radius_rsun",
    "stellar_mass_msun",
)

RIGHT_SKEWED_COLUMNS = (
    "insolation_earth",
    "transit_depth",
    "a_over_rs",
    "planet_radius_rearth",
    "orbital_period_days",
    "transit_duration_hours",
)

LEFT_SKEWED_COLUMNS = ("inclination_deg",)


def model_feature_columns(columns: Sequence[str]) -> list[str]:
    """Return final model features after notebook-driven drops."""
    excluded = set(ANALYSIS_DROP_COLUMNS) | {"group_id", "label"}
    return [col for col in columns if col not in excluded]


# Backward-compatible aliases kept for existing notebook imports.
koi_columns_set = list(KOI_COLUMNS_SET)
k2p_columns_set = list(K2P_COLUMNS_SET)
