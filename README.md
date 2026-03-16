# exoplanet-detection

## Feature selection
### Canonical column definitions

This project harmonizes KOI (Kepler Objects of Interest) and K2P (K2 Planets & Candidates) into a unified schema. Below are the final (canonical) columns used after renaming, along with their meaning and key dependencies/redundancies.
#### group_id

**Meaning:** Identifier used for grouping/splitting to prevent leakage.

* KOI: `kepid` (host star ID; multiple candidates can share a group)
* K2P: `pl_name` (planet name; typically unique per object)

#### label

**Meaning:** Target derived from archive disposition (planet-like vs not).

---

#### orbital_period_days

**Meaning:** Orbital period in days (time between consecutive transits).
**Dependencies/redundancy:** Related to `semi_major_axis_au` and `stellar_mass_msun` via Kepler’s 3rd law (approx.); often correlated with `a_over_rs`.

#### eccentricity

**Meaning:** Orbit shape parameter (0=circular, >0=elliptical).
**Dependencies/redundancy:** Interacts with `arg_periastron_deg`; if `eccentricity ≈ 0`, periastron angle is physically irrelevant.

#### arg_periastron_deg

**Meaning:** Argument/longitude of periastron in degrees (orientation of the ellipse).
**Dependencies/redundancy:** Only meaningful when `eccentricity > 0`; otherwise effectively undefined.

#### impact_parameter

**Meaning:** Transit impact parameter (b) (0=center crossing, ~1=grazing).
**Dependencies/redundancy:** Geometrically linked to `inclination_deg` and `a_over_rs`.

#### inclination_deg

**Meaning:** Orbital inclination in degrees (transits require inclination near 90°).
**Dependencies/redundancy:** Strongly linked to `impact_parameter` and `a_over_rs` by geometry.

---

#### transit_duration_hours

**Meaning:** Total transit duration in hours.
**Dependencies/redundancy:** Depends on orbit geometry and scale (often linked to `orbital_period_days`, `a_over_rs`, `impact_parameter`, and to a lesser extent `radius_ratio_rp_rs`).

#### transit_depth_ppm

**Meaning:** Transit depth in parts-per-million (fractional brightness drop).
**Dependencies/redundancy:** Approximately (\text{depth} \approx (R_p/R_s)^2), so it is tightly related to `radius_ratio_rp_rs`.

#### radius_ratio_rp_rs

**Meaning:** Radius ratio (R_p/R_s) (planet radius over stellar radius).
**Dependencies/redundancy:** Squared value is closely related to `transit_depth_ppm`; also linked to `planet_radius_rearth` and `stellar_radius_rsun` via (R_p = (R_p/R_s)\cdot R_s).

---

#### a_over_rs

**Meaning:** Scaled semi-major axis (a/R_s).
**Dependencies/redundancy:** Related to `semi_major_axis_au` and `stellar_radius_rsun` (if derived); also correlated with `orbital_period_days` and `stellar_mass_msun` through Kepler’s law.

#### semi_major_axis_au

**Meaning:** Semi-major axis (a) in astronomical units (AU).
**Dependencies/redundancy:** Related to `orbital_period_days` and `stellar_mass_msun` via Kepler’s 3rd law (approx.).

---

#### planet_radius_rearth

**Meaning:** Planet radius in Earth radii.
**Dependencies/redundancy:** Derived from `radius_ratio_rp_rs` and `stellar_radius_rsun`; also correlated with `transit_depth_ppm`.

#### equilibrium_temp_k

**Meaning:** Planet equilibrium temperature in Kelvin (based on simplifying assumptions).
**Dependencies/redundancy:** Derived from stellar properties and orbital distance; correlated with `insolation_earth`, `stellar_teff_k`, and `semi_major_axis_au`.

#### insolation_earth

**Meaning:** Incident stellar flux on the planet in Earth units (Earth=1).
**Dependencies/redundancy:** Derived from stellar luminosity and orbital distance; correlated with `equilibrium_temp_k` and `semi_major_axis_au`.

---

#### stellar_teff_k

**Meaning:** Host star effective temperature in Kelvin.
**Dependencies/redundancy:** Correlated with `stellar_radius_rsun`, `stellar_mass_msun`, and derived irradiation (`equilibrium_temp_k`, `insolation_earth`).

#### stellar_logg_cgs

**Meaning:** Host star surface gravity (\log_{10}(g)) in cgs units (cm/s²).
**Dependencies/redundancy:** Linked to `stellar_mass_msun` and `stellar_radius_rsun` via (g \propto M/R^2).

#### stellar_metallicity_dex

**Meaning:** Host star metallicity in dex (typically ([Fe/H])).
**Dependencies/redundancy:** Not directly derivable from the other columns here; can correlate weakly with stellar population and planet occurrence.

#### stellar_radius_rsun

**Meaning:** Host star radius in solar radii.
**Dependencies/redundancy:** Used to derive `planet_radius_rearth` from `radius_ratio_rp_rs`; correlated with `stellar_mass_msun` and `stellar_logg_cgs`.

#### stellar_mass_msun

**Meaning:** Host star mass in solar masses.
**Dependencies/redundancy:** Together with `orbital_period_days` determines `semi_major_axis_au` (Kepler’s law); also linked to `stellar_logg_cgs` and `stellar_radius_rsun`.

#### stellar_age_gyr

**Meaning:** Host star age in gigayears.
**Dependencies/redundancy:** Typically model-derived and loosely correlated with mass/metallicity/temperature, but not directly derivable from the other columns here.
