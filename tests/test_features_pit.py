"""Hygiene test 5a -- feature point-in-time purity: features at ``asof`` are
identical whether or not future rows exist (or are corrupted)."""

import numpy as np
import pandas as pd

from marketgnn.features import BASE_FEATURES, compute_features, forward_return


def test_features_are_pit_pure(panel):
    prices, volume, assets = panel
    asof = prices.index[300]
    base = compute_features(prices, volume, asof, assets)

    corrupted_px = prices.copy()
    corrupted_vol = volume.copy()
    corrupted_px.loc[corrupted_px.index > asof] = 1e6
    corrupted_vol.loc[corrupted_vol.index > asof] = 1e6
    after = compute_features(corrupted_px, corrupted_vol, asof, assets)

    pd.testing.assert_frame_equal(base[BASE_FEATURES], after[BASE_FEATURES])


def test_features_are_finite_with_enough_history(panel):
    prices, volume, assets = panel
    feat = compute_features(prices, volume, prices.index[300], assets)
    # with >252 rows of history every feature is defined for every asset
    assert feat[BASE_FEATURES].notna().all().all()


def test_forward_return_reads_the_future(panel):
    prices, volume, assets = panel
    asof = prices.index[300]
    y = forward_return(prices, asof, assets, horizon=21)
    manual = prices.iloc[321] / prices.iloc[300] - 1
    pd.testing.assert_series_equal(y, manual.reindex(assets), check_names=False)


def test_forward_return_is_nan_at_the_edge(panel):
    prices, volume, assets = panel
    y = forward_return(prices, prices.index[-5], assets, horizon=21)
    assert y.isna().all()
