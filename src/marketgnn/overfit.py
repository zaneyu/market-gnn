"""Run 12 — backtest-overfitting hardening: PBO via CSCV and the deflated Sharpe ratio.

Applies the standard Bailey–Borwein–López de Prado–Zhu CSCV/PBO and Bailey–López de Prado
PSR/DSR machinery to the repo's two surviving positives (the short-term-reversal control and,
at the IC level, the Run 11 vol row). Design decisions that matter (see the spec):

- Every config in the pre-registered 3×3 reversal grid is converted to a DAILY return series
  via Jegadeesh–Titman overlapping tranches (1/h of the quintile long-short book re-formed
  daily, fixed formation weights, ramp-up renormalized over live tranches) — mixed-horizon
  step-sampling would put configs on incompatible date grids with incommensurable returns.
- PBO convention pinned for odd N: P(IS-best's OOS rank STRICTLY below the median); 9 noise
  configs ⇒ 4/9, not 0.5. Single-realization PBO is hugely dispersed (the splits reuse the
  same block statistics), so calibration is by seed-average and the real-data PBO is one
  realization reported with that caveat.
- PSR/DSR use the repo's own HAC standard via T_eff = T·(se_naive/se_HAC)²; `ic_psr` is plain
  PSR (nothing was searched for Run 11 — N=1 has no maximum to deflate against).

See docs/superpowers/specs/2026-07-16-overfit-hardening-design.md.
"""

from __future__ import annotations

from itertools import combinations

import numpy as np
import pandas as pd
from scipy import stats

from .evaluate import newey_west_tstat

EULER_GAMMA = 0.5772156649015329
GRID = [(lb, h) for lb in (1, 3, 5) for h in (1, 3, 5)]  # pre-registered reversal grid


# --------------------------------------------------------------------------- strategy
def daily_strategy_returns(prices: pd.DataFrame, *, lookback: int, horizon: int,
                           q: float = 0.2, warmup: int = 260) -> pd.Series:
    """DAILY gross return series of the reversal quintile long-short via Jegadeesh–Titman
    overlapping tranches. Each day t >= warmup a tranche is formed from the reversal signal
    (long the top-q by signal = past losers, short the bottom-q) and held ``horizon`` days
    with fixed equal formation weights (tranche daily return = mean long-leg return − mean
    short-leg return over formation membership). The book on day u averages all live
    tranches (renormalized during the first horizon−1 ramp-up days)."""
    rets = prices.pct_change()
    sig_panel = -(prices / prices.shift(lookback) - 1.0)   # reversal: high = past loser
    idx = prices.index
    T, n = prices.shape
    k = max(1, int(round(q * n)))

    acc = np.zeros(T)     # sum of live-tranche returns per day
    cnt = np.zeros(T)     # live-tranche count per day
    R = rets.to_numpy()
    for t in range(warmup, T - 1):
        sig = sig_panel.iloc[t]
        sig = sig[np.isfinite(sig)]
        if len(sig) < 2 * k:
            continue
        order = sig.sort_values()
        lo = prices.columns.get_indexer(order.index[:k])       # short: past winners
        hi = prices.columns.get_indexer(order.index[-k:])      # long: past losers
        for u in range(t + 1, min(t + 1 + horizon, T)):
            r_hi = R[u, hi]
            r_lo = R[u, lo]
            if np.all(np.isfinite(r_hi)) and np.all(np.isfinite(r_lo)):
                acc[u] += r_hi.mean() - r_lo.mean()
                cnt[u] += 1
    live = cnt > 0
    out = pd.Series(acc[live] / cnt[live], index=idx[live])
    return out


# --------------------------------------------------------------------------- CSCV / PBO
def cscv_pbo(returns_matrix: pd.DataFrame, *, n_blocks: int = 16, embargo: int = 4) -> dict:
    """CSCV probability of backtest overfitting over a [n_days × n_configs] matrix on ONE
    common time axis. Contiguous blocks; ``embargo`` days dropped at the end of each interior
    block boundary; per-block precomputed sums / sums-of-squares (never recomputed per split).
    Convention (pinned): PBO = fraction of splits where the IS-best config's OOS Sharpe rank
    is STRICTLY below the median (rank <= (N-1)/2 for odd N; pure noise => 4/9)."""
    assert n_blocks % 2 == 0, "n_blocks must be even"
    M = returns_matrix.to_numpy(float)
    n_days, n_cfg = M.shape
    assert n_days >= 10 * n_blocks, "blocks would be too short for a meaningful IS Sharpe"

    # contiguous blocks with an embargo at each interior boundary (end of blocks 0..S-2)
    edges = np.linspace(0, n_days, n_blocks + 1, dtype=int)
    block_rows = []
    used = 0
    for b in range(n_blocks):
        lo, hi = edges[b], edges[b + 1]
        if b < n_blocks - 1:
            hi = hi - embargo
        rows = np.arange(lo, hi)
        block_rows.append(rows)
        used += len(rows)

    sums = np.stack([M[r].sum(axis=0) for r in block_rows])          # [S, N]
    sqs = np.stack([(M[r] ** 2).sum(axis=0) for r in block_rows])    # [S, N]
    lens = np.array([len(r) for r in block_rows], dtype=float)       # [S]

    def _sharpe_from(blocks: tuple[int, ...]) -> np.ndarray:
        b = list(blocks)
        n_obs = lens[b].sum()
        mean = sums[b].sum(axis=0) / n_obs
        var = sqs[b].sum(axis=0) / n_obs - mean**2
        sd = np.sqrt(np.maximum(var, 1e-18))
        return mean / sd

    all_blocks = frozenset(range(n_blocks))
    half = n_blocks // 2
    below_median = 0
    rank_freq = np.zeros(n_cfg, dtype=int)
    n_splits = 0
    median_rank = (n_cfg + 1) / 2.0
    for is_blocks in combinations(range(n_blocks), half):
        oos_blocks = tuple(sorted(all_blocks - set(is_blocks)))
        sr_is = _sharpe_from(is_blocks)
        sr_oos = _sharpe_from(oos_blocks)
        best = int(np.argmax(sr_is))
        # ascending rank of the IS-best among OOS Sharpes: 1 = worst, N = best
        rank = int((sr_oos <= sr_oos[best]).sum())
        rank_freq[rank - 1] += 1
        if rank < median_rank:
            below_median += 1
        n_splits += 1
    return {"pbo": below_median / n_splits,
            "oos_rank_freq": rank_freq / n_splits,
            "n_splits": n_splits,
            "n_days_used": used}


# --------------------------------------------------------------------------- PSR / DSR
def sharpe(returns: pd.Series) -> float:
    """Per-period (non-annualized) Sharpe on the series' native cadence."""
    x = np.asarray(returns, float)
    x = x[np.isfinite(x)]
    sd = x.std(ddof=1)
    return float(x.mean() / sd) if sd > 0 else np.nan


def psr(returns: pd.Series, sr_star: float = 0.0, *, t_eff: float | None = None) -> float:
    """Probabilistic Sharpe ratio (Bailey–López de Prado 2012) with RAW (non-excess)
    kurtosis. Normal case reduces to Φ((SR−SR*)·√(T−1)/√(1+SR²/2)) — the SR²/2 term does not
    vanish. ``t_eff`` (HAC-aware effective sample size) replaces T when given."""
    x = np.asarray(returns, float)
    x = x[np.isfinite(x)]
    T = len(x)
    if T < 10:
        return np.nan
    sr = sharpe(pd.Series(x))
    xc = x - x.mean()
    m2 = np.mean(xc**2)
    g3 = np.mean(xc**3) / m2**1.5
    g4 = np.mean(xc**4) / m2**2          # RAW kurtosis: normal ~ 3 (never feed excess)
    T_use = t_eff if t_eff is not None else T
    denom = np.sqrt(max(1e-12, 1.0 - g3 * sr + (g4 - 1.0) / 4.0 * sr**2))
    return float(stats.norm.cdf((sr - sr_star) * np.sqrt(T_use - 1) / denom))


def expected_max_sharpe(n_trials: int, var_sr: float) -> float:
    """E[max Sharpe] under the null across ``n_trials`` (Bailey–López de Prado 2014).
    Domain n_trials >= 2: at n=1 the formula is −∞ (nothing was searched — use plain PSR)."""
    assert n_trials >= 2, "expected_max_sharpe requires n_trials >= 2 (N=1: use psr directly)"
    return float(np.sqrt(var_sr) * ((1 - EULER_GAMMA) * stats.norm.ppf(1 - 1 / n_trials)
                                    + EULER_GAMMA * stats.norm.ppf(1 - 1 / (n_trials * np.e))))


def dsr(returns: pd.Series, n_trials: int, var_sr: float, *, t_eff: float | None = None) -> float:
    """Deflated Sharpe ratio: PSR evaluated at the expected max Sharpe under the null."""
    return psr(returns, sr_star=expected_max_sharpe(n_trials, var_sr), t_eff=t_eff)


def hac_t_eff(returns: pd.Series) -> float:
    """HAC-aware effective sample size T_eff = T·(se_naive/se_HAC)². se_naive is taken from
    ``newey_west_tstat(x, lag=0)`` — at lag 0 the Bartlett terms vanish, giving √(γ₀/n) in
    the SAME 1/n variance convention as the HAC se (std(ddof=1)/√T would differ by
    √(n/(n−1)))."""
    x = np.asarray(returns, float)
    x = x[np.isfinite(x)]
    _, se_hac = newey_west_tstat(x)
    _, se_naive = newey_west_tstat(x, lag=0)
    if not (np.isfinite(se_hac) and np.isfinite(se_naive)) or se_hac <= 0:
        return float(len(x))
    return float(len(x) * (se_naive / se_hac) ** 2)


def ic_psr(ic_series: pd.Series) -> float:
    """Run 11's IC-level probabilistic significance: plain PSR with the HAC-aware T_eff.
    Never touches expected_max_sharpe — the config was pre-registered, N=1, there is no
    maximum to deflate against."""
    return psr(ic_series, sr_star=0.0, t_eff=hac_t_eff(ic_series))


# --------------------------------------------------------------------------- Run 12
def run_overfit(prices: pd.DataFrame, *, n_blocks: int = 16, warmup: int = 260
                ) -> tuple[pd.DataFrame, dict]:
    """Build the 9-config daily return matrix, run CSCV/PBO, and compute SR/PSR/DSR
    (N ∈ {9, 25, 100}; iid-T and HAC T_eff variants) for the pre-registered headline config
    (lookback=1, horizon=1 — the Run 6 reversal)."""
    series = {}
    rows = []
    for lb, h in GRID:
        s = daily_strategy_returns(prices, lookback=lb, horizon=h, warmup=warmup)
        series[(lb, h)] = s
        rows.append({"lookback": lb, "horizon": h, "sharpe": sharpe(s),
                     "ann_sharpe": sharpe(s) * np.sqrt(252), "n_days": len(s)})
    table = pd.DataFrame(rows)

    M = pd.concat({k: v for k, v in series.items()}, axis=1).dropna()
    pbo_out = cscv_pbo(M, n_blocks=n_blocks)

    head = series[(1, 1)]
    var_sr = float(np.var([r["sharpe"] for r in rows], ddof=1))
    teff = hac_t_eff(head)
    hx = head.to_numpy()
    hx = hx[np.isfinite(hx)] - np.nanmean(hx)
    m2 = np.mean(hx**2)
    summary = {
        "pbo": pbo_out["pbo"], "oos_rank_freq": pbo_out["oos_rank_freq"],
        "n_splits": pbo_out["n_splits"], "n_days_used": pbo_out["n_days_used"],
        "headline": (1, 1), "sharpe": sharpe(head), "ann_sharpe": sharpe(head) * np.sqrt(252),
        "psr": psr(head), "psr_teff": psr(head, t_eff=teff), "t_eff": teff,
        "T": int(np.isfinite(head.to_numpy()).sum()), "var_sr": var_sr,
        "skew": float(np.mean(hx**3) / m2**1.5), "kurtosis": float(np.mean(hx**4) / m2**2),
    }
    for n in (9, 25, 100):
        summary[f"dsr_n{n}"] = dsr(head, n_trials=n, var_sr=var_sr)
        summary[f"dsr_n{n}_teff"] = dsr(head, n_trials=n, var_sr=var_sr, t_eff=teff)
    return table, summary


def main():
    import argparse

    ap = argparse.ArgumentParser(description="Run 12: PBO/CSCV + deflated Sharpe hardening")
    ap.add_argument("--synthetic-planted", action="store_true",
                    help="calibration demo on noise + one skilled config")
    ap.add_argument("--run11-ic", action="store_true",
                    help="reproduce Run 11's ic_psr on the co-holding spill_resid IC series")
    args = ap.parse_args()

    if args.synthetic_planted:
        rng = np.random.default_rng(0)
        M = pd.DataFrame(rng.normal(0, 0.01, size=(2000, 9)),
                         index=pd.bdate_range("2015-01-01", periods=2000))
        print("=== CSCV calibration: pure noise ===")
        print("NOTE: this is ONE realization — single-draw PBO is hugely dispersed "
              "(measured 0.26-0.94); the seed-AVERAGED expectation is 4/9 = 0.444, which is "
              "what the calibration tests assert.")
        print({k: (round(v, 3) if isinstance(v, float) else v)
               for k, v in cscv_pbo(M).items() if k != "oos_rank_freq"})
        M[0] = M[0] + 0.004
        print("=== one genuinely skilled config (expect low PBO) ===")
        print({k: (round(v, 3) if isinstance(v, float) else v)
               for k, v in cscv_pbo(M).items() if k != "oos_rank_freq"})
        return

    if args.run11_ic:
        _run11_ic()
        return

    from .data.download import load_market
    from .data.universe import default_universe

    prices, _vol, _sectors, _mkt = load_market(
        synthetic=False, tickers=default_universe(), start="2014-01-01", end="2024-12-31")
    print("=== Run 12: PBO/CSCV + deflated Sharpe for the reversal grid (2014-2024) ===")
    table, s = run_overfit(prices)
    with pd.option_context("display.float_format", lambda v: f"{v:+.4f}"):
        print(table.to_string(index=False))
    print(f"\nCSCV: S=16, {s['n_splits']} splits, {s['n_days_used']} days after embargo")
    print(f"PBO (strictly-below-median convention; noise => 0.444): {s['pbo']:.3f}")
    print("OOS-rank histogram (1=worst..9=best): "
          + " ".join(f"{f:.3f}" for f in s["oos_rank_freq"]))
    print(f"\nHeadline config (lookback=1, horizon=1): SR/day {s['sharpe']:+.4f} "
          f"(annualized {s['ann_sharpe']:+.2f}), T={s['T']}, T_eff={s['t_eff']:.0f}")
    print(f"PSR(0):      iid {s['psr']:.4f}   HAC-T_eff {s['psr_teff']:.4f}")
    for n in (9, 25, 100):
        print(f"DSR (N={n:>3}): iid {s[f'dsr_n{n}']:.4f}   HAC-T_eff {s[f'dsr_n{n}_teff']:.4f}")


def _run11_ic():
    """Committed reproduction path for RESULTS' 'ic_psr 0.9946, T_eff 85 of 125': rebuilds
    Run 11's co-holding spill_resid per-date IC series (mirrors run_volspill's coholding
    path exactly) and applies ic_psr. ~2 min on the cached panel."""
    from . import volspill
    from .coholding import cusip_map, make_provider
    from .data.download import load_market
    from .evaluate import per_date_ic

    nodes = list(cusip_map())
    prices, _v, _s, _m = load_market(
        synthetic=False, tickers=nodes, start="2014-01-01", end="2024-12-31")
    prices = prices.reindex(columns=nodes).dropna(how="all")
    provider, _ = make_provider(list(prices.columns), k=10)
    rets = prices.pct_change()
    vol_s = volspill.trailing_vol(rets, lookback=20)
    vol_l = volspill.trailing_vol(rets, lookback=250)
    innov = vol_s - vol_l
    idx = prices.index
    recs = []
    for i in range(260, len(idx) - 20, 20):
        t = idx[i]
        g = provider["coholding"](t, 0)
        sig = volspill.neighbour_innovation(innov.iloc[i], g)
        fwd_win = rets.iloc[i + 1:i + 21]
        fwd = fwd_win.std(ddof=1).where(fwd_win.notna().sum() >= 20).reindex(sig.index)
        recs.append(pd.DataFrame({"date": t, "sig": sig.to_numpy(),
                                  "own_s": vol_s.iloc[i].reindex(sig.index).to_numpy(),
                                  "own_l": vol_l.iloc[i].reindex(sig.index).to_numpy(),
                                  "fwd": fwd.to_numpy()}))
    df = pd.concat(recs, ignore_index=True)
    resid = df.groupby("date", sort=True, group_keys=False).apply(
        lambda gdf: pd.Series(volspill._residualize_multi(
            gdf["fwd"].to_numpy(), np.column_stack([gdf["own_s"], gdf["own_l"]])),
            index=gdf.index),
        include_groups=False)
    ic = per_date_ic(df["sig"], resid.reindex(df.index), df["date"]).dropna()
    print(f"Run 11 coholding spill_resid IC: mean {ic.mean():+.4f}, n={len(ic)}")
    print(f"T_eff (HAC-aware): {hac_t_eff(ic):.0f} of {len(ic)}")
    print(f"ic_psr: {ic_psr(ic):.4f}  (N=1 — pre-registered config, plain PSR, no deflation)")


if __name__ == "__main__":
    main()
