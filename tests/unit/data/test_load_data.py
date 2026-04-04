from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from exoplanet_detector.data.load_data import convert_transit_depth_percent_to_ppm

pytestmark = pytest.mark.unit


def test_convert_transit_depth_percent_to_ppm_converts_numeric_and_coerces_invalid() -> None:
    """Convert mixed transit-depth inputs from percent to ppm while preserving invalids as NaN."""
    source = pd.DataFrame(
        {
            "transit_depth": [0.1, "0.25", "bad", None],
            "other_col": [1, 2, 3, 4],
        }
    )

    converted = convert_transit_depth_percent_to_ppm(source)

    # Conversion correctness for valid values.
    assert converted is not source
    assert converted.loc[0, "transit_depth"] == pytest.approx(1000.0)
    assert converted.loc[1, "transit_depth"] == pytest.approx(2500.0)
    # Invalid/non-numeric values are coerced to NaN.
    assert np.isnan(converted.loc[2, "transit_depth"])
    assert np.isnan(converted.loc[3, "transit_depth"])
    # Source dataframe remains unchanged.
    assert source.loc[0, "transit_depth"] == 0.1
