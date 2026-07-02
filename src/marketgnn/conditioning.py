"""Liquidity-conditioned anomaly analysis.

Short-term reversal is documented to be an illiquidity / arbitrage-cost effect: it
should concentrate in less-liquid names and fade where capital can trade it away.
This splits each cross-section into liquidity terciles (trailing dollar volume) and
measures the signal within each, plus a HAC test on the low-minus-high IC spread --
a direct test of the gradient. It's the economically-motivated companion to the
raw positive control in `signals.py`.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .evaluate import ic_summary, newey_west_tstat, rank_ic, two_sided_p


def run_liquidity_conditioning(
    prices: pd.DataFrame, volume: pd.DataFrame, *, spec: dict, n_groups: int = 3,
    warmup: int = 260, liq_window: int = 20,
) -> tuple[pd.DataFrame, dict]:
    """Per-liquidity-group IC for one signal, plus the low-minus-high spread test.

    Group 0 = least liquid, group n-1 = most liquid. Labels are non-overlapping
    (step = horizon), so the per-date ICs and the spread series are clean for HAC.
    """
    idx = prices.index
    dollar = (prices * volume).rolling(liq_window).mean()
    lb, h, sign, skip = spec["lookback"], spec["horizon"], spec["sign"], spec.get("skip", 0)

    per_group: dict[int, dict] = {g: {} for g in range(n_groups)}
    for i in range(max(warmup, lb + skip), len(idx) - h, h):
        liq = dollar.iloc[i]
        sig = sign * (prices.iloc[i - skip] / prices.iloc[i - skip - lb] - 1)
        fwd = prices.iloc[i + h] / prices.iloc[i] - 1
        q = liq.rank(pct=True)
        for g in range(n_groups):
            cols = liq.index[((q > g / n_groups) & (q <= (g + 1) / n_groups)).to_numpy()]
            if len(cols) >= 3:
                ic = rank_ic(sig[cols].to_numpy(), fwd[cols].to_numpy())
                if np.isfinite(ic):
                    per_group[g][idx[i]] = ic

    labels = {0: "least liquid", n_groups - 1: "most liquid"}
    ic_series, rows = {}, []
    for g in range(n_groups):
        s = pd.Series(per_group[g]).sort_index()
        ic_series[g] = s
        summ = ic_summary(s)
        rows.append({
            "group": g, "label": labels.get(g, f"mid {g}"), "mean_ic": summ["mean_ic"],
            "hac_t": summ["hac_t"], "p": two_sided_p(summ["hac_t"]), "n_dates": summ["n"],
        })

    low, high = ic_series[0], ic_series[n_groups - 1]
    common = low.index.intersection(high.index)
    spread = (low.loc[common] - high.loc[common]).dropna()
    st, _ = newey_west_tstat(spread.to_numpy())
    spread_stats = {
        "mean_spread": float(spread.mean()), "hac_t": st, "p": two_sided_p(st), "n_dates": len(spread),
    }
    return pd.DataFrame(rows), spread_stats


def main():
    from .data.download import download_prices
    from .data.universe import extended_universe

    prices, volume, _ = download_prices(extended_universe(), "2014-01-01", "2024-12-31", cache_key="extended")
    spec = {"name": "reversal_1d", "lookback": 1, "horizon": 1, "sign": -1}
    table, spread = run_liquidity_conditioning(prices, volume, spec=spec)
    print(f"=== short-term reversal by liquidity ({prices.shape[1]} names) ===")
    print(table.to_string(index=False, float_format=lambda v: f"{v:+.4f}"))
    print(f"\nlow-minus-high IC spread: {spread['mean_spread']:+.4f}  "
          f"HAC t {spread['hac_t']:+.2f}  p {spread['p']:.4f}  "
          f"-> reversal {'IS' if spread['p'] < 0.05 else 'is NOT'} significantly stronger in illiquid names")


if __name__ == "__main__":
    main()
