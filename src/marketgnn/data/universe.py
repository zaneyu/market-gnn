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


# A liquid large-cap starter universe for the first real run. NOTE: this is the
# *current* membership -> survivorship-biased. Return claims on this set are
# restricted; the volatility target is far more robust. Replace with PIT Russell
# 1000 membership (build_membership) for defensible return claims. See PLAN.md.
DEFAULT_UNIVERSE = [
    "AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA", "TSLA", "AVGO", "ADBE", "CRM",
    "ORCL", "CSCO", "ACN", "INTC", "AMD", "QCOM", "TXN", "IBM", "INTU", "NOW",
    "JPM", "BAC", "WFC", "GS", "MS", "C", "AXP", "BLK", "SCHW", "SPGI",
    "V", "MA", "PYPL",
    "UNH", "JNJ", "LLY", "PFE", "MRK", "ABBV", "TMO", "ABT", "DHR", "BMY", "AMGN", "CVS",
    "HD", "MCD", "NKE", "SBUX", "LOW", "TGT", "BKNG", "TJX",
    "PG", "KO", "PEP", "COST", "WMT", "MDLZ", "CL", "MO", "PM",
    "XOM", "CVX", "COP", "SLB", "EOG",
    "BA", "CAT", "GE", "HON", "UPS", "RTX", "LMT", "DE", "MMM", "UNP",
    "DIS", "NFLX", "CMCSA", "T", "VZ", "TMUS",
    "LIN", "NEE", "DUK", "SO", "AMT", "PLD", "SPG",
]


def default_universe() -> list[str]:
    """Current large-cap set for a first real run. Survivorship-biased by
    construction -- callers must restrict return claims accordingly."""
    return list(DEFAULT_UNIVERSE)




# Honest (approximate GICS) sector labels for the default universe, so the sector
# graph is real structure rather than a placeholder. Static, as sectors should be.
_SECTORS = {
    "tech": ["AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA", "TSLA", "AVGO", "ADBE", "CRM",
             "ORCL", "CSCO", "ACN", "INTC", "AMD", "QCOM", "TXN", "IBM", "INTU", "NOW"],
    "financials": ["JPM", "BAC", "WFC", "GS", "MS", "C", "AXP", "BLK", "SCHW", "SPGI",
                   "V", "MA", "PYPL"],
    "healthcare": ["UNH", "JNJ", "LLY", "PFE", "MRK", "ABBV", "TMO", "ABT", "DHR", "BMY", "AMGN", "CVS"],
    "consumer_disc": ["HD", "MCD", "NKE", "SBUX", "LOW", "TGT", "BKNG", "TJX"],
    "consumer_staples": ["PG", "KO", "PEP", "COST", "WMT", "MDLZ", "CL", "MO", "PM"],
    "energy": ["XOM", "CVX", "COP", "SLB", "EOG"],
    "industrials": ["BA", "CAT", "GE", "HON", "UPS", "RTX", "LMT", "DE", "MMM", "UNP"],
    "communications": ["DIS", "NFLX", "CMCSA", "T", "VZ", "TMUS"],
    "utilities_re_materials": ["LIN", "NEE", "DUK", "SO", "AMT", "PLD", "SPG"],
}
_TICKER_SECTOR = {t: s for s, ts in _SECTORS.items() for t in ts}


def default_sectors(tickers) -> "pd.Series":
    """Sector label per ticker (approx GICS), 'other' if unknown."""
    return pd.Series({t: _TICKER_SECTOR.get(t, "other") for t in tickers}, name="sector")


# Extended universe: ~90 additional mid/large names, ALL public before 2014 (no IPO
# gaps over the window), spanning well below the mega-caps in liquidity. Widens the
# liquidity range (for liquidity-conditioned tests) and gives every experiment more
# cross-sectional power. Still current-membership -> survivorship caveat unchanged.
_EXTENDED = {
    "tech": ["AKAM", "FFIV", "JNPR", "NTAP", "STX", "WDC", "SWKS", "MCHP", "TER", "ZBRA",
             "PTC", "ANSS", "CDW", "JKHY", "TYL", "BR", "PAYX", "FAST"],
    "healthcare": ["WAT", "MTD", "IDXX", "RMD", "ZBH", "BAX", "HOLX", "DGX", "LH", "STE", "XRAY", "UHS"],
    "industrials": ["AOS", "NDSN", "DOV", "ROK", "EXPD", "JBHT", "CHRW", "PNR", "IEX", "SNA", "SWK", "PH", "ITW"],
    "consumer_disc": ["WHR", "LEG", "MHK", "HAS", "RL", "TPR", "PVH", "DPZ", "GRMN", "POOL", "WYNN", "MGM"],
    "consumer_staples": ["CLX", "CHD", "HRL", "MKC", "SJM", "CAG", "K", "GIS", "HSY", "KMB", "TSN", "TAP"],
    "financials": ["CINF", "L", "MKL", "WRB", "AFL", "ALL", "TRV", "PGR", "HIG", "FITB", "KEY", "RF", "HBAN", "CFG"],
    "materials": ["ALB", "CF", "MOS", "FMC", "NUE", "STLD", "IP", "PKG", "SEE", "ECL", "PPG", "SHW", "NEM", "FCX"],
    "energy": ["HAL", "OXY", "DVN", "HES", "APA", "BKR", "WMB", "OKE", "KMI"],
}
for _s, _ts in _EXTENDED.items():
    for _t in _ts:
        _TICKER_SECTOR.setdefault(_t, _s)


def extended_universe() -> list[str]:
    """~180 names (large + mid caps, all public pre-2014) for higher power and a
    genuine liquidity range. Deduplicated, order-stable."""
    extra = [t for ts in _EXTENDED.values() for t in ts]
    return list(dict.fromkeys(list(DEFAULT_UNIVERSE) + extra))


# --- Point-in-time S&P 500 membership from the public change log ---------------

_WIKI_SP500 = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"


def fetch_sp500_changes() -> "pd.DataFrame":
    """Long DataFrame of index change events [date, ticker, action] from Wikipedia.

    Needs a browser User-Agent (Wikipedia 403s the default urllib agent). This is
    inclusion/removal *timing*; it does NOT provide delisted-name prices -- see
    ``sp500_membership`` for the survivorship caveat.
    """
    import io
    import urllib.request

    req = urllib.request.Request(_WIKI_SP500, headers={"User-Agent": "Mozilla/5.0 (marketgnn research)"})
    html = urllib.request.urlopen(req, timeout=30).read().decode("utf-8")
    ch = pd.read_html(io.StringIO(html))[1]
    ch.columns = ["date", "add_t", "add_s", "rem_t", "rem_s", "reason"]
    ch["date"] = pd.to_datetime(ch["date"], errors="coerce")
    events = []
    for _, r in ch.iterrows():
        if pd.notna(r["add_t"]):
            events.append((r["date"], r["add_t"], "add"))
        if pd.notna(r["rem_t"]):
            events.append((r["date"], r["rem_t"], "remove"))
    return pd.DataFrame(events, columns=["date", "ticker", "action"]).dropna(subset=["date"])


def sp500_membership(tickers, dates: "pd.DatetimeIndex", changes: "pd.DataFrame | None" = None) -> "pd.DataFrame":
    """Boolean [date x ticker] membership mask: a name is a member only on/after its
    most recent S&P 500 *add* date. Corrects forward-looking inclusion bias (e.g.
    TSLA absent from pre-2020 cross-sections).

    LIMITATION (be honest): this fixes inclusion *timing* for current members. It
    does NOT restore delisted names -- yfinance lacks their prices -- so a run on a
    current-member universe is inclusion-corrected but NOT fully survivorship-free.
    Full survivorship-freedom needs a vendor with delisted prices (e.g. CRSP).
    """
    if changes is None:
        changes = fetch_sp500_changes()
    adds = changes[changes["action"] == "add"]
    mask = pd.DataFrame(True, index=dates, columns=list(tickers))
    for t in tickers:
        t_adds = adds.loc[adds["ticker"] == t, "date"]
        if len(t_adds):
            mask.loc[mask.index < t_adds.max(), t] = False
    return mask
