"""Run 12 — backtest-overfitting hardening: PBO via CSCV + deflated Sharpe.

The two calibration tests are seed-AVERAGED: single-realization PBO is hugely dispersed
(the C(S, S/2) splits reuse the same S block statistics), so per-draw assertions would be
flaky or vacuous. Convention pinned in the spec: PBO = P(IS-best's OOS rank strictly below
the median); for N=9 noise configs the theoretical value is 4/9. Torch-free."""

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from marketgnn import overfit


# --------------------------------------------------------------- CSCV / PBO calibration
def _noise_matrix(rng, T=480, n_cfg=9):
    return pd.DataFrame(rng.normal(0, 0.01, size=(T, n_cfg)),
                        index=pd.bdate_range("2015-01-01", periods=T))


def test_cscv_pbo_detects_pure_noise():
    # negative control: no config has skill -> mean PBO over seeds ~ 4/9 (pinned convention)
    pbos = []
    for s in range(24):
        rng = np.random.default_rng(s)
        out = overfit.cscv_pbo(_noise_matrix(rng), n_blocks=8, embargo=0)
        pbos.append(out["pbo"])
    assert np.mean(pbos) == pytest.approx(4 / 9, abs=0.10)


def test_cscv_pbo_low_for_true_skill():
    # positive control: one config with a genuine mean shift keeps winning OOS
    pbos = []
    for s in range(24):
        rng = np.random.default_rng(100 + s)
        M = _noise_matrix(rng)
        M[0] = M[0] + 0.004  # strong true skill, well above the noise floor
        pbos.append(overfit.cscv_pbo(M, n_blocks=8, embargo=0)["pbo"])
    assert np.mean(pbos) < 0.25


# --------------------------------------------------------------- PSR / DSR formulas
def test_sharpe_psr_dsr_formulas():
    rng = np.random.default_rng(0)
    r = pd.Series(rng.normal(0.0005, 0.01, size=400))
    sr = overfit.sharpe(r)
    # normal case: PSR = Phi((SR - SR*) * sqrt(T-1) / sqrt(1 + SR^2/2)) — the SR^2/2 term
    # does NOT vanish at skew 0 / raw kurt 3 (Lo/Mertens); hand-compute with SAMPLE moments
    x = r.to_numpy()
    m2 = np.mean((x - x.mean()) ** 2)
    g3 = np.mean((x - x.mean()) ** 3) / m2**1.5
    g4 = np.mean((x - x.mean()) ** 4) / m2**2      # RAW kurtosis (normal ~ 3)
    T = len(x)
    expect = stats.norm.cdf(sr * np.sqrt(T - 1) / np.sqrt(1 - g3 * sr + (g4 - 1) / 4 * sr**2))
    assert overfit.psr(r) == pytest.approx(expect, abs=1e-12)
    # feeding EXCESS kurtosis (the classic bug) would change the answer
    wrong = stats.norm.cdf(sr * np.sqrt(T - 1) / np.sqrt(1 - g3 * sr + (g4 - 4) / 4 * sr**2))
    assert abs(overfit.psr(r) - wrong) > 0  # raw-kurtosis semantics pinned
    # E[max] domain and monotonicity (n >= 2)
    with pytest.raises(AssertionError):
        overfit.expected_max_sharpe(1, 0.01)
    e2, e5, e25 = (overfit.expected_max_sharpe(n, 0.01) for n in (2, 5, 25))
    assert e2 < e5 < e25
    # DSR < PSR(0) whenever something was searched (n >= 2, V > 0)
    assert overfit.dsr(r, n_trials=9, var_sr=0.01) < overfit.psr(r)
    # small-sample guard
    assert np.isnan(overfit.psr(pd.Series([0.01] * 5)))


# --------------------------------------------------------------- strategy construction
def test_daily_strategy_returns_is_pit_and_tranched():
    rng = np.random.default_rng(3)
    n_days, n = 320, 20
    prices = pd.DataFrame(100 * np.exp(np.cumsum(rng.normal(0, 0.01, size=(n_days, n)), axis=0)),
                          index=pd.bdate_range("2020-01-01", periods=n_days),
                          columns=[f"S{i:02d}" for i in range(n)])
    r1 = overfit.daily_strategy_returns(prices, lookback=1, horizon=3, warmup=260)
    # PIT: perturbing strictly-future prices leaves earlier daily returns unchanged
    p2 = prices.copy()
    p2.iloc[300:] *= 3.0
    r2 = overfit.daily_strategy_returns(p2, lookback=1, horizon=3, warmup=260)
    common = r1.index[r1.index < prices.index[297]]  # safely before the perturbation window
    assert np.allclose(r1.loc[common], r2.loc[common])
    # daily axis: consecutive business days, starting right after warmup
    assert r1.index[0] == prices.index[261]
    assert (np.diff(r1.index.to_numpy()).astype("timedelta64[D]") <= np.timedelta64(4, "D")).all()
    # tranche structure on a hand-computable panel: h=1 book equals the single-tranche
    # long-short of yesterday's signal quintiles
    rh1 = overfit.daily_strategy_returns(prices, lookback=1, horizon=1, warmup=260)
    t = 280  # formation at close of day t, marked on day t+1
    sig = -(prices.iloc[t] / prices.iloc[t - 1] - 1)
    k = max(1, int(round(0.2 * n)))
    order = sig.sort_values()
    lo, hi = order.index[:k], order.index[-k:]
    rets_next = prices.iloc[t + 1] / prices.iloc[t] - 1
    expect = rets_next[hi].mean() - rets_next[lo].mean()
    assert rh1.loc[prices.index[t + 1]] == pytest.approx(expect)


def test_run_overfit_grid_shape():
    rng = np.random.default_rng(7)
    n_days, n = 700, 30
    prices = pd.DataFrame(100 * np.exp(np.cumsum(rng.normal(0, 0.01, size=(n_days, n)), axis=0)),
                          index=pd.bdate_range("2018-01-01", periods=n_days),
                          columns=[f"S{i:02d}" for i in range(n)])
    table, summary = overfit.run_overfit(prices, n_blocks=8)
    assert len(table) == 9                                   # the pre-registered grid
    assert {"lookback", "horizon", "sharpe"} <= set(table.columns)
    for key in ("pbo", "psr", "psr_teff", "t_eff", "dsr_n9", "dsr_n25", "dsr_n100"):
        assert key in summary, f"missing {key}"
    # the embargo actually fires: a no-op embargo would pass the calibration tests
    M = pd.DataFrame(rng.normal(0, 0.01, size=(400, 9)),
                     index=pd.bdate_range("2019-01-01", periods=400))
    out = overfit.cscv_pbo(M, n_blocks=8, embargo=4)
    assert out["n_days_used"] == 400 - (8 - 1) * 4
