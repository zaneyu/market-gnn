"""Hygiene test 5 -- normalization scope: cross-sectional normalization uses only
within-date statistics (no cross-date pooling) and is shift/scale-invariant."""

import numpy as np
import pandas as pd

from marketgnn.features import cross_sectional_normalize


def _cross_section(seed):
    rng = np.random.default_rng(seed)
    idx = pd.Index([f"A{i}" for i in range(20)], name="asset")
    return pd.DataFrame(rng.normal(size=(20, 4)), index=idx, columns=["a", "b", "c", "d"])


def test_no_cross_date_pooling_via_shift_invariance():
    feat = _cross_section(0)
    # A different "date" with a wildly shifted+scaled distribution must normalize
    # to the SAME values if (and only if) no cross-date statistics are used.
    shifted = feat * 7.0 + 100.0
    pd.testing.assert_frame_equal(
        cross_sectional_normalize(feat), cross_sectional_normalize(shifted)
    )
    pd.testing.assert_frame_equal(
        cross_sectional_normalize(feat, method="rank"),
        cross_sectional_normalize(shifted, method="rank"),
    )


def test_output_is_standardized_per_column():
    res = cross_sectional_normalize(_cross_section(1))
    assert np.allclose(res.mean(axis=0), 0, atol=1e-12)
    assert np.allclose(res.std(axis=0, ddof=0), 1, atol=1e-9)


def test_all_nan_column_becomes_zero_not_nan():
    feat = _cross_section(2)
    feat["a"] = np.nan
    res = cross_sectional_normalize(feat)
    assert res["a"].notna().all()
    assert np.allclose(res["a"], 0.0)
