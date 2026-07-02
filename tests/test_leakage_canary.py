"""Hygiene tests 1 & 2 -- label-shuffle and future-injection canary.

These are the two tests a leaker fails. Shuffle: destroying the pred<->target
alignment must collapse IC to ~0. Canary: a feature equal to the future label
must produce near-perfect IC, proving the evaluation *would* light up if the
future leaked into the feature set.
"""

import numpy as np
import pandas as pd

from marketgnn.evaluate import per_date_ic, rank_ic
from marketgnn.features import leakage_canary


def _panel(seed=0, n_dates=40, n_assets=30):
    rng = np.random.default_rng(seed)
    rows = []
    for d in range(n_dates):
        y = rng.normal(size=n_assets)
        signal = 0.3 * y + rng.normal(size=n_assets)  # weakly informative predictor
        for a in range(n_assets):
            rows.append({"date": d, "asset": a, "pred": signal[a], "target": y[a]})
    return pd.DataFrame(rows)


def test_future_injection_canary_spikes_ic():
    df = _panel()
    canary = leakage_canary(df["target"])
    ic = per_date_ic(canary, df["target"], df["date"])
    assert ic.mean() > 0.999, "a feature == future label must ace IC (detector has teeth)"


def test_label_shuffle_collapses_ic():
    df = _panel()
    rng = np.random.default_rng(123)
    shuffled = df.groupby("date")["target"].transform(lambda s: rng.permutation(s.to_numpy()))
    ic = per_date_ic(df["pred"], shuffled, df["date"])
    assert abs(ic.mean()) < 0.05, "shuffling targets within date must destroy IC"


def test_real_signal_sits_between():
    df = _panel()
    ic = per_date_ic(df["pred"], df["target"], df["date"])
    assert 0.05 < ic.mean() < 0.95, "weak real signal should be clearly positive but far from 1"


def test_rank_ic_nan_on_constant():
    assert np.isnan(rank_ic(np.ones(10), np.arange(10)))
