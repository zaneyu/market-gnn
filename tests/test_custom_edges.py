"""The economic-link drop-in: a graph built from an explicit link table, and the
reproducibility manifest hash."""

import pandas as pd

from marketgnn.data.snapshot import content_hash
from marketgnn.graph import edges_from_pairs


def test_edges_from_pairs_maps_tickers_and_drops_unknowns():
    nodes = ["AAPL", "MSFT", "NVDA"]
    links = pd.DataFrame({"src": ["AAPL", "MSFT", "AAPL"], "dst": ["NVDA", "NVDA", "ZZZ"], "weight": [0.5, 0.8, 1.0]})
    g = edges_from_pairs(links, nodes)
    assert g.num_edges == 2  # AAPL->NVDA, MSFT->NVDA; the ZZZ link is dropped
    assert set(map(tuple, g.edge_index.T.tolist())) == {(0, 2), (1, 2)}


def test_edges_from_pairs_symmetric():
    nodes = ["A", "B"]
    g = edges_from_pairs(pd.DataFrame({"src": ["A"], "dst": ["B"]}), nodes, symmetric=True)
    assert g.num_edges == 2
    assert set(map(tuple, g.edge_index.T.tolist())) == {(0, 1), (1, 0)}


def test_content_hash_is_order_independent():
    idx = pd.bdate_range("2020-01-01", periods=10)
    a = pd.DataFrame({"AAPL": range(10), "MSFT": range(10, 20)}, index=idx).astype(float)
    b = a[["MSFT", "AAPL"]]  # reordered columns -> same hash
    assert content_hash(a) == content_hash(b)
    c = a.copy(); c.iloc[0, 0] += 1.0
    assert content_hash(c) != content_hash(a)  # changed data -> different hash
