"""Point-in-time universe + delisting handling -- the survivorship defense.

Using *today's* index members for a historical study is the confound that can
manufacture the very predictability we test for (the review's #1 data issue). This
module builds a boolean [date x asset] membership mask so each cross-section uses
only the names that were actually in the index as-of that date, and patches
delisting returns so blow-ups aren't silently dropped from the label window.

The membership/delisting SOURCES are pluggable: point a CSV of index change events
(date, ticker, added/removed) and a CSV of delistings (ticker, date, final_return)
at ``build_membership`` / ``apply_delistings``. Absent those, callers pass
``membership=None`` and MUST restrict claims accordingly (documented in the README).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def build_membership(changes_csv: str | Path, dates: pd.DatetimeIndex, tickers) -> pd.DataFrame:
    """Reconstruct a [date x ticker] boolean membership mask from an add/remove log.

    ``changes_csv`` columns: ``date, ticker, action`` where action in {add, remove}.
    A ticker is a member from its add date (inclusive) until its remove date.
    """
    changes = pd.read_csv(changes_csv, parse_dates=["date"]).sort_values("date")
    mask = pd.DataFrame(False, index=dates, columns=list(tickers))
    state = {t: False for t in tickers}
    ci = 0
    ev = changes.to_dict("records")
    for d in dates:
        while ci < len(ev) and ev[ci]["date"] <= d:
            t, action = ev[ci]["ticker"], ev[ci]["action"]
            if t in state:
                state[t] = action == "add"
            ci += 1
        row = mask.loc[d]
        for t, member in state.items():
            row[t] = member
    return mask


def apply_delistings(prices: pd.DataFrame, delistings_csv: str | Path) -> pd.DataFrame:
    """Extend each delisted name's price path one step with its final return so the
    label window is complete (−100% for bankruptcy, deal price for acquisition).

    ``delistings_csv`` columns: ``ticker, date, final_return`` (e.g. -1.0 for a wipeout).
    """
    dl = pd.read_csv(delistings_csv, parse_dates=["date"])
    out = prices.copy()
    for r in dl.itertuples(index=False):
        if r.ticker not in out.columns:
            continue
        col = out[r.ticker]
        last_valid = col.last_valid_index()
        if last_valid is None:
            continue
        final_price = col.loc[last_valid] * (1.0 + float(r.final_return))
        # place the terminal price at the delist date (or the next available index)
        pos = out.index.get_indexer([r.date], method="bfill")[0]
        if pos != -1:
            out.iloc[pos, out.columns.get_loc(r.ticker)] = final_price
    return out


def russell1000_placeholder(tickers) -> None:
    """Real runs need PIT Russell 1000 membership (see README for sourcing). Returning
    None signals 'no membership mask' so downstream code restricts claims honestly."""
    return None
