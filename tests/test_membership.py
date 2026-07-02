"""Point-in-time membership mask: a name is a member only on/after its add date.
Tests the reconstruction logic with a synthetic change log (no network)."""

import pandas as pd

from marketgnn.data.universe import sp500_membership


def test_membership_excludes_names_before_their_add_date():
    dates = pd.bdate_range("2018-01-01", "2022-12-31", freq="W-FRI")
    tickers = ["OLD", "NEW"]
    # OLD has no add event in-window (member throughout); NEW added mid-2020
    changes = pd.DataFrame(
        [(pd.Timestamp("2020-07-01"), "NEW", "add")], columns=["date", "ticker", "action"]
    )
    mask = sp500_membership(tickers, dates, changes=changes)

    assert mask["OLD"].all()  # no add event -> member for the whole window
    assert not mask.loc[mask.index < "2020-07-01", "NEW"].any()  # absent before add
    assert mask.loc[mask.index >= "2020-07-01", "NEW"].all()  # present on/after add


def test_membership_uses_most_recent_add():
    dates = pd.bdate_range("2015-01-01", "2021-12-31", freq="W-FRI")
    changes = pd.DataFrame(
        [
            (pd.Timestamp("2016-01-01"), "X", "add"),
            (pd.Timestamp("2018-01-01"), "X", "remove"),
            (pd.Timestamp("2020-01-01"), "X", "add"),  # re-added; most recent add governs
        ],
        columns=["date", "ticker", "action"],
    )
    mask = sp500_membership(["X"], dates, changes=changes)
    assert not mask.loc[mask.index < "2020-01-01", "X"].any()
    assert mask.loc[mask.index >= "2020-01-01", "X"].all()
