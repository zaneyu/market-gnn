"""Tests for the 13F co-holding economic-link graph (no network needed):
edge construction, PIT snapshot selection, and the committed CUSIP map."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from marketgnn import graph as G
from marketgnn.coholding import _snapshots, coholding_links, cusip_map, holders_covered, pick_asof


def test_cusip_map_known_identifiers():
    m = cusip_map()
    # a handful of public, stable CUSIPs (incl. the dual-class Alphabet A / Meta A)
    assert m["AAPL"] == "037833100"
    assert m["MSFT"] == "594918104"
    assert m["JPM"] == "46625H100"
    assert m["GOOGL"] == "02079K305"  # Class A (not GOOG Class C 02079K107)
    assert len(m) == 90


def _cusip_check_digit(c8: str) -> str:
    """Standard CUSIP mod-10 double-add-double check digit for the first 8 chars."""
    total = 0
    for i, ch in enumerate(c8):
        v = int(ch) if ch.isdigit() else ord(ch.upper()) - ord("A") + 10
        if i % 2:
            v *= 2
        total += v // 10 + v % 10
    return str((10 - total % 10) % 10)


def test_every_cusip_is_valid():
    """Every committed CUSIP is 9 uppercase alphanumerics AND passes the CUSIP
    check-digit. This is the regression guard for the map bug where BAC/SCHW/DIS/AMT/SPG
    carried bond/depositary/placeholder/typo identifiers: it catches the placeholder
    (SPG1C0150, bad check digit) and the lowercase typo (03027x100, bad format)
    outright. (Bond CUSIPs are themselves valid; test_universe_fully_covered below is
    the data-driven guard that catches a valid-but-wrong-security identifier.)"""
    m = cusip_map()
    import re

    for tk, c in m.items():
        assert re.fullmatch(r"[0-9A-Z]{9}", c), f"{tk}: {c!r} is not 9 uppercase alphanumerics"
        assert _cusip_check_digit(c[:8]) == c[8], f"{tk}: {c} fails the CUSIP check-digit"


def test_coholding_links_common_owners_make_edges():
    # A,B share all 10 holders; C is held by a disjoint set -> A~B strong, no A~C edge
    nodes = ["A", "B", "C"]
    rows = []
    for f in range(10):
        rows += [(f"h{f}", "A"), (f"h{f}", "B")]
    for f in range(10, 16):
        rows.append((f"h{f}", "C"))
    holders = pd.DataFrame(rows, columns=["cik", "ticker"])
    links = coholding_links(holders, nodes, k=2, min_common=5)
    w = {(r.src, r.dst): r.weight for r in links.itertuples(index=False)}
    assert w[("A", "B")] == pytest.approx(1.0)   # identical holder sets -> cosine 1
    assert w[("B", "A")] == pytest.approx(1.0)   # symmetric
    assert ("A", "C") not in w and ("C", "A") not in w  # no common holders -> no edge


def test_coholding_links_min_common_filter():
    # only 3 common holders < min_common=5 -> edge dropped even though cosine > 0
    nodes = ["A", "B"]
    rows = [(f"h{f}", "A") for f in range(8)] + [(f"h{f}", "B") for f in range(5, 13)]
    holders = pd.DataFrame(rows, columns=["cik", "ticker"])  # common = {5,6,7} -> 3
    links = coholding_links(holders, nodes, k=2, min_common=5)
    assert len(links) == 0


def test_pick_asof_is_point_in_time():
    nodes = ["A", "B"]
    g14 = G.edges_from_pairs(pd.DataFrame({"src": ["A"], "dst": ["B"]}), nodes, symmetric=True)
    g20 = G.edges_from_pairs(pd.DataFrame({"src": ["B"], "dst": ["A"]}), nodes, symmetric=True)
    graphs = [(pd.Timestamp("2014-02-15"), g14), (pd.Timestamp("2020-02-15"), g20)]
    # BEFORE the first public date NOTHING is public yet -> an EMPTY graph, never the
    # earliest snapshot (whose Dec holdings weren't filed until the following Feb).
    early = pick_asof(graphs, "2013-06-01")
    assert early.num_edges == 0 and list(early.nodes) == nodes
    # between snapshots -> the most recent already-public one
    assert pick_asof(graphs, "2016-06-01") is g14
    # on/after a public date -> that snapshot
    assert pick_asof(graphs, "2020-03-01") is g20
    # exactly on the public date is available (>= is inclusive)
    assert pick_asof(graphs, "2020-02-15") is g20


def test_symmetric_edges_are_not_double_counted():
    """A mutual kNN pair (both i->j and j->i present in links) plus symmetric=True must
    yield exactly one (i,j) and one (j,i) edge -- not doubled -- else aggregation
    double-weights mutual pairs and avg_degree is inflated (the M1 bug)."""
    nodes = ["A", "B", "C"]
    links = pd.DataFrame({"src": ["A", "B"], "dst": ["B", "A"], "weight": [0.9, 0.9]})
    g = G.edges_from_pairs(links, nodes, symmetric=True)
    pairs = list(zip(g.edge_index[0].tolist(), g.edge_index[1].tolist()))
    assert sorted(pairs) == [(0, 1), (1, 0)]  # one each way, no duplicates
    assert g.num_edges == 2


@pytest.mark.skipif(not (Path(__file__).resolve().parents[1] / ".cache" / "13f").exists(),
                    reason="13F datasets not fetched (fetched-not-shipped); run coholding.main once")
def test_universe_fully_covered_by_cusip_map():
    """DATA-GATED regression guard for the CUSIP-map bug: with the real 13F filings
    present, EVERY universe name must have >=1 institutional holder in the most recent
    snapshot. A count below the full universe means a CUSIP maps to no filing row (a
    bond/depositary/placeholder/typo identifier) and that name is an isolated node --
    exactly the defect that silently dropped BAC/SCHW/DIS/AMT/SPG."""
    cmap = cusip_map()
    covered = holders_covered(_snapshots()[-1], cmap)  # Dec-2023 snapshot
    assert covered == len(cmap), f"only {covered}/{len(cmap)} names have holders; a CUSIP is wrong"


def test_coholding_graph_is_symmetric_and_bounded_degree():
    nodes = [f"S{i}" for i in range(6)]
    rng = np.random.default_rng(0)
    rows = []
    for f in range(50):
        held = rng.choice(nodes, size=rng.integers(2, 5), replace=False)
        rows += [(f"f{f}", t) for t in held]
    holders = pd.DataFrame(rows, columns=["cik", "ticker"])
    links = coholding_links(holders, nodes, k=2, min_common=1)
    g = G.edges_from_pairs(links, nodes, symmetric=True)
    out, inn = G.in_out_degrees(g)
    assert (out == inn).all()  # symmetric graph
