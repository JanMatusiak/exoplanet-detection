# Feature Dictionary

This document explains the canonical feature names after KOI/K2P harmonization (`KOI_RENAME_MAP`, `K2P_RENAME_MAP`).

If you are new to astronomy: the model mostly uses measurements from the **transit method** (when a planet passes in front of a star and slightly dims its light).

## IDs and target

### group_id
ID used to group related rows (for example, multiple candidates around the same star).  
It is mainly used for safe train/test splitting to avoid leakage.

### label
Training target:
- `1`: planet-like
- `0`: non-planet-like

## Orbital and transit geometry features

### orbital_period_days
How long the planet takes to complete one orbit.  
Unit: days.  
Example intuition: `365` means "roughly one Earth year per orbit."

### eccentricity
How non-circular the orbit is.  
Unit: unitless, usually `0` to `<1`.  
Interpretation:
- near `0`: almost circular
- larger value: more stretched ellipse

### arg_periastron_deg
Direction/orientation of the elliptical orbit in its plane.  
Unit: degrees (`0` to `360`).  
This matters mostly when eccentricity is not near zero.

### impact_parameter
How centrally the planet crosses the star disk during transit.  
Unit: unitless.  
Interpretation:
- near `0`: crosses near the center of the star
- near `1` or more: more edge/grazing transit

### inclination_deg
Tilt of the orbit relative to our line of sight.  
Unit: degrees.  
Interpretation:
- near `90`: edge-on from our viewpoint, transit is more likely visible

### a_over_rs
Distance from planet to star, scaled by star radius (`a/Rs`).  
Unit: unitless ratio.  
Larger values mean the planet orbits farther away relative to star size.

## Transit signal shape/intensity features

### transit_duration_hours
How long each transit event lasts.  
Unit: hours.  
Influenced by orbit speed, orbit geometry, and star size.

### transit_depth
How much the star brightness drops during transit.  
Unit: parts per million - harmonized during preprocessing (KOI and K2P are aligned to one scale).  
Interpretation:
- larger depth: stronger dimming signal, often larger planet relative to star

### radius_ratio_rp_rs
Planet radius divided by star radius (`Rp/Rs`).  
Unit: unitless ratio.  
Bigger ratio usually means deeper transit.

## Planet physical property features

### planet_radius_rearth
Estimated planet size in Earth radii.  
Unit: Earth radii (`R_earth`).  
Interpretation:
- `1` is Earth-sized
- larger values indicate larger planets

### semi_major_axis_au
Average orbital distance from star (semi-major axis).  
Unit: astronomical units (`AU`), where `1 AU` is Earth-Sun distance.

### equilibrium_temp_k
Estimated planet equilibrium temperature.  
Unit: Kelvin (`K`).  
This is an approximation based on received starlight, not a direct surface measurement.

### insolation_earth
How much stellar energy reaches the planet compared to Earth.  
Unit: Earth-relative ratio (`Earth = 1`).  
Interpretation:
- `1`: Earth-like received flux
- `10`: ten times more than Earth

## Host star features

### stellar_teff_k
Star effective temperature (photosphere temperature).  
Unit: Kelvin (`K`).

### stellar_logg_cgs
Star surface gravity on a log scale (`log10(g)` in cgs units).  
Unit: dex-like logarithmic scale.  
Useful for distinguishing compact vs extended stars.

### stellar_metallicity_dex
Star metallicity (abundance of elements heavier than hydrogen/helium) on a log scale.  
Unit: dex (logarithmic).

### stellar_radius_rsun
Star radius relative to the Sun.  
Unit: solar radii (`R_sun`).

### stellar_mass_msun
Star mass relative to the Sun.  
Unit: solar masses (`M_sun`).

### stellar_age_gyr
Estimated age of the star.  
Unit: gigayears (`Gyr`, billions of years).

## Final training feature set

Final model training does **not** use all mapped columns above.  
The final subset is `FINAL_FEATURE_COLUMNS` in:
`src/exoplanet_detector/features/feature_selection.py`.
