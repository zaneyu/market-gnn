"""Evaluation: rank IC, HAC significance, block-bootstrap CIs, FDR control.

Predictions are scored per date (cross-sectionally) then aggregated across dates.
The per-date IC series is autocorrelated (overlapping label horizons, regime
persistence), so i.i.d. inference overstates significance -- the single most
common way a "rigorous" market study gets shot down. Hence:

* significance via Newey-West/HAC standard errors (``newey_west_tstat``),
* CIs via a moving-**block** bootstrap (``block_bootstrap_ci``),
* the naive i.i.d. versions kept alongside so the paper can show the inflation,
* multiple comparisons across the ablation grid controlled with Benjamini-Hochberg.

Depends only on numpy / pandas / scipy, so it runs in CI without the model stack.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats


def rank_ic(pred: np.ndarray, target: np.ndarray) -> float:
    """Spearman rank correlation for one cross-section (NaN if degenerate)."""
    pred = np.asarray(pred, float)
    target = np.asarray(target, float)
    mask = np.isfinite(pred) & np.isfinite(target)
    if mask.sum() < 3:
        return np.nan
    p, t = pred[mask], target[mask]
    if np.ptp(p) == 0 or np.ptp(t) == 0:  # constant -> Spearman undefined
        return np.nan
    ic, _ = stats.spearmanr(p, t)
    return float(ic)


def per_date_ic(pred, target, dates) -> pd.Series:
    """rank_ic for every date, indexed by date. Inputs are aligned, one row per
    (date, asset) observation."""
    df = pd.DataFrame({"pred": np.asarray(pred), "target": np.asarray(target), "date": np.asarray(dates)})
    return df.groupby("date", sort=True).apply(
        lambda g: rank_ic(g["pred"].to_numpy(), g["target"].to_numpy()), include_groups=False
    )


def long_short_spread(pred: np.ndarray, target: np.ndarray, q: float = 0.2) -> float:
    """Top-quantile minus bottom-quantile mean target (one date). NaN if the split
    is degenerate (e.g. constant predictions), never a misleading 0."""
    pred = np.asarray(pred, float)
    target = np.asarray(target, float)
    mask = np.isfinite(pred) & np.isfinite(target)
    pred, target = pred[mask], target[mask]
    if len(pred) < 5:
        return np.nan
    lo, hi = np.quantile(pred, q), np.quantile(pred, 1 - q)
    if hi <= lo:
        return np.nan
    top, bottom = target[pred >= hi], target[pred <= lo]
    if len(top) == 0 or len(bottom) == 0:
        return np.nan
    return float(top.mean() - bottom.mean())


def newey_west_tstat(x, lag: int | None = None) -> tuple[float, float]:
    """HAC (Newey-West) t-stat and standard error for the mean of ``x``.

    ``lag`` should be >= the label horizon in rebalance steps; defaults to the
    Newey-West rule of thumb ``floor(4*(n/100)**(2/9))``.
    """
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    n = len(x)
    if n < 3:
        return (np.nan, np.nan)
    if lag is None:
        lag = int(np.floor(4 * (n / 100) ** (2 / 9)))
    lag = max(0, min(lag, n - 1))
    e = x - x.mean()
    var = (e @ e) / n  # gamma_0
    for l in range(1, lag + 1):
        w = 1.0 - l / (lag + 1)  # Bartlett kernel
        var += 2 * w * (e[l:] @ e[:-l]) / n
    se = np.sqrt(var / n)
    if se == 0:
        return (np.nan, 0.0)
    return (float(x.mean() / se), float(se))


def ic_summary(ic: pd.Series, *, hac_lag: int | None = None) -> dict:
    """Headline numbers for a per-date IC series, with naive AND HAC inference so
    the autocorrelation inflation is visible."""
    ic = ic.dropna()
    n = len(ic)
    if n == 0:
        return {"mean_ic": np.nan, "ic_std": np.nan, "ic_ir": np.nan, "naive_t": np.nan, "hac_t": np.nan, "n": 0}
    mean, std = ic.mean(), ic.std(ddof=1)
    naive_t = mean / (std / np.sqrt(n)) if std else np.nan
    hac_t, _ = newey_west_tstat(ic.to_numpy(), lag=hac_lag)
    return {
        "mean_ic": float(mean),
        "ic_std": float(std),
        "ic_ir": float(mean / std) if std else np.nan,
        "naive_t": float(naive_t) if std else np.nan,
        "hac_t": hac_t,
        "n": n,
    }


def block_bootstrap_ci(x, *, block: int, alpha: float = 0.05, n_boot: int = 10_000, seed: int = 0):
    """Moving-block bootstrap CI for the mean -- preserves short-range dependence
    that the i.i.d. bootstrap destroys. ``block`` should be >= the label horizon."""
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    n = len(x)
    if n < 2:
        return (np.nan, np.nan)
    block = max(1, min(block, n))
    n_blocks = int(np.ceil(n / block))
    rng = np.random.default_rng(seed)
    starts = rng.integers(0, n - block + 1, size=(n_boot, n_blocks))
    offsets = np.arange(block)
    idx = (starts[:, :, None] + offsets[None, None, :]).reshape(n_boot, -1)[:, :n]
    means = x[idx].mean(axis=1)
    return (float(np.quantile(means, alpha / 2)), float(np.quantile(means, 1 - alpha / 2)))


def iid_bootstrap_ci(x, *, alpha: float = 0.05, n_boot: int = 10_000, seed: int = 0):
    """Naive i.i.d. bootstrap -- kept ONLY to contrast against the block bootstrap
    and expose how much the autocorrelation inflates confidence."""
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    if len(x) < 2:
        return (np.nan, np.nan)
    rng = np.random.default_rng(seed)
    means = rng.choice(x, size=(n_boot, len(x)), replace=True).mean(axis=1)
    return (float(np.quantile(means, alpha / 2)), float(np.quantile(means, 1 - alpha / 2)))


def benjamini_hochberg(pvalues, alpha: float = 0.05):
    """Benjamini-Hochberg FDR control across the ablation grid. Returns (reject
    mask, q-values) aligned to the input order."""
    p = np.asarray(pvalues, float)
    m = len(p)
    order = np.argsort(p)
    ranked = p[order]
    q = ranked * m / (np.arange(1, m + 1))
    q = np.minimum.accumulate(q[::-1])[::-1]  # enforce monotonicity
    reject_sorted = ranked <= (np.arange(1, m + 1) / m) * alpha
    thresh = np.where(reject_sorted)[0]
    cutoff = thresh.max() if thresh.size else -1
    reject = np.zeros(m, bool)
    reject[order[: cutoff + 1]] = True
    qvals = np.empty(m)
    qvals[order] = np.clip(q, 0, 1)
    return reject, qvals


def qlike(realized_var, forecast_var) -> float:
    """QLIKE loss (lower is better; 0 iff forecast == realized), the standard
    proper loss for variance forecasts and robust to noise in the realized proxy.
    QLIKE = mean( r/f - log(r/f) - 1 ) over positive, finite (realized, forecast)."""
    r = np.asarray(realized_var, float)
    f = np.asarray(forecast_var, float)
    mask = np.isfinite(r) & np.isfinite(f) & (r > 0) & (f > 0)
    if mask.sum() == 0:
        return np.nan
    ratio = r[mask] / f[mask]
    return float(np.mean(ratio - np.log(ratio) - 1))


def two_sided_p(t_stat: float) -> float:
    """Normal-approx two-sided p-value from a (HAC) t-stat."""
    if not np.isfinite(t_stat):
        return np.nan
    return float(2 * stats.norm.sf(abs(t_stat)))
