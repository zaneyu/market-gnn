"""Known-anomaly positive controls on REAL data.

The graph channels are nulls (Runs 1-5). Are they real absences, or can this
pipeline just not find anything? This module answers by pointing the SAME leak-free
HAC harness at pre-specified, well-documented cross-sectional anomalies:

- short-term reversal (Lehmann 1990 / Jegadeesh 1990): yesterday's losers outperform
  today. A genuine microstructure/liquidity effect.
- 12-1 momentum (Jegadeesh-Titman): trailing-year-ex-recent-month winners.

These are NOT discovered here and NOT tuned -- they are textbook effects used as a
real-data positive control. Labels are non-overlapping (step = horizon), so the HAC
inference is clean. Reported GROSS: reversal especially is high-turnover and largely
arbitraged after transaction costs -- the honest caveat is printed alongside.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .evaluate import benjamini_hochberg, ic_summary, per_date_ic, two_sided_p

# (name, lookback days, forward horizon days, sign): sign −1 = reversal, +1 = momentum
DEFAULT_CONTROLS = [
    {"name": "reversal_1d", "lookback": 1, "horizon": 1, "sign": -1},
    {"name": "reversal_1w", "lookback": 5, "horizon": 5, "sign": -1},
    {"name": "momentum_12_1", "lookback": 231, "horizon": 21, "sign": +1, "skip": 21},
]


def _signal(prices: pd.DataFrame, i: int, lookback: int, sign: int, skip: int) -> pd.Series:
    return sign * (prices.iloc[i - skip] / prices.iloc[i - skip - lookback] - 1)


def run_positive_controls(prices: pd.DataFrame, *, warmup: int = 260, controls=None) -> pd.DataFrame:
    """Per-date rank-IC of each known anomaly on real prices, with HAC t-stats,
    BH-FDR, and a crude turnover proxy (higher = more cost-fragile)."""
    controls = controls or DEFAULT_CONTROLS
    idx = prices.index
    rets = prices.pct_change()
    rows = []
    for c in controls:
        lb, h, sign, skip = c["lookback"], c["horizon"], c["sign"], c.get("skip", 0)
        # non-overlapping labels: step = horizon -> clean HAC inference
        starts = range(max(warmup, lb + skip), len(idx) - h, h)
        recs, prev_rank, turn = [], None, []
        for i in starts:
            sig = _signal(prices, i, lb, sign, skip)
            fwd = prices.iloc[i + h] / prices.iloc[i] - 1
            recs.append(pd.DataFrame({"date": idx[i], "sig": sig.to_numpy(), "fwd": fwd.to_numpy()}))
            r = sig.rank()
            if prev_rank is not None:
                turn.append((r - prev_rank).abs().mean() / len(r))
            prev_rank = r
        df = pd.concat(recs, ignore_index=True)
        ic = per_date_ic(df["sig"], df["fwd"], df["date"]).dropna()
        s = ic_summary(ic)
        rows.append({
            "signal": c["name"], "mean_ic": s["mean_ic"], "hac_t": s["hac_t"],
            "naive_t": s["naive_t"], "p": two_sided_p(s["hac_t"]), "turnover": float(np.mean(turn)),
            "n_dates": s["n"],
        })
    table = pd.DataFrame(rows)
    reject, q = benjamini_hochberg(table["p"].fillna(1.0).to_numpy())
    table["fdr_sig"], table["q"] = reject, q
    return table


def main():
    from .data.download import load_market
    from .data.universe import default_universe

    prices, *_ = load_market(synthetic=False, tickers=default_universe(),
                             start="2014-01-01", end="2024-12-31")
    table = run_positive_controls(prices)
    print("=== known-anomaly positive controls on real data (GROSS) ===")
    cols = ["signal", "mean_ic", "hac_t", "p", "turnover", "n_dates", "fdr_sig"]
    print(table[cols].to_string(index=False, float_format=lambda v: f"{v:+.4f}"))
    print("\nNOTE: gross rank-IC. reversal_1d is high-turnover (~daily) and largely "
          "arbitraged after realistic transaction costs; it is a statistical positive "
          "control that the harness finds REAL return signal, not a tradable strategy.")


if __name__ == "__main__":
    main()
