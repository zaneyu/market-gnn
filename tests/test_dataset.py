"""Dataset assembly: shapes align, labels are PIT-consistent, and the 'none' graph
kind truly blindfolds the neighbour feature."""

import numpy as np
import pandas as pd

from marketgnn.dataset import build_dataset, make_synthetic, rebalance_dates
from marketgnn.features import FEATURES


def _small():
    prices, volume, sectors, market = make_synthetic(n_days=700, n_assets=30, n_sectors=4, seed=0)
    rebal = rebalance_dates(prices.index, "W")
    return prices, volume, sectors, market, rebal


def test_build_dataset_shapes_and_alignment():
    prices, volume, sectors, market, rebal = _small()
    ds = build_dataset(prices, volume, sectors, market, rebal, graph_kind="correlation",
                       label_horizon=5, corr_window=60, k=6, warmup=260)
    assert list(ds.X.columns) == FEATURES
    assert ds.X.index.equals(ds.y_ret.index) and ds.X.index.equals(ds.y_vol.index)
    assert len(ds.dates) > 10
    assert ds.X.notna().all().all()  # normalization leaves no NaN features


def test_none_graph_blindfolds_neighbour_feature():
    prices, volume, sectors, market, rebal = _small()
    ds = build_dataset(prices, volume, sectors, market, rebal, graph_kind="none",
                       label_horizon=5, corr_window=60, k=6, warmup=260)
    # with no graph, nbr_ret is constant 0 -> normalized to 0 everywhere
    assert np.allclose(ds.X["nbr_ret"].to_numpy(), 0.0)
    assert all(g is None for g in ds.graphs.values())


def test_labels_match_direct_computation():
    prices, volume, sectors, market, rebal = _small()
    ds = build_dataset(prices, volume, sectors, market, rebal, graph_kind="correlation",
                       label_horizon=5, corr_window=60, k=6, warmup=260)
    d = ds.dates[5]
    assets = ds.X.loc[d].index.tolist()
    i = prices.index.get_indexer([d])[0]
    manual = (prices.iloc[i + 5][assets] / prices.iloc[i][assets] - 1).to_numpy()
    assert np.allclose(ds.y_ret.loc[d].to_numpy(), manual, equal_nan=True)
