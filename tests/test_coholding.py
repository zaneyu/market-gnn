"""Tests for the 13F co-holding economic-link graph (no network needed):
edge construction, PIT snapshot selection, and the committed CUSIP map."""

import numpy as np
import pandas as pd
import pytest

from marketgnn import graph as G
from marketgnn.coholding import coholding_links, cusip_map, pick_asof


def test_cusip_map_known_identifiers():
    m = cusip_map()
    # a handful of public, stable CUSIPs (incl. the dual-class Alphabet A / Meta A)
    assert m["AAPL"] == "037833100"
    assert m["MSFT"] == "594918104"
    assert m["JPM"] == "46625H100"
    assert m["GOOGL"] == "02079K305"  # Class A (not GOOG Class C 02079K107)
    assert len(m) == 90


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
    # before the first public date -> earliest (never future data)
    assert pick_asof(graphs, "2013-06-01") is g14
    # between snapshots -> the most recent already-public one
    assert pick_asof(graphs, "2016-06-01") is g14
    # on/after a public date -> that snapshot
    assert pick_asof(graphs, "2020-03-01") is g20
    # exactly on the public date is available (>= is inclusive)
    assert pick_asof(graphs, "2020-02-15") is g20


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
