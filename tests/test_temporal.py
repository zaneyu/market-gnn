"""The temporal (GConvGRU) A/B has power: on a synthetic market with a PLANTED
lead-lag effect, the graph variant recovers it AND adds incremental OOS IC over the
graph-blindfolded variant (self-loops only), whose sole difference is the missing
topology. Torch-gated (skips in the core torch-free CI run).

Honest caveat baked into the assertions: the self-loop baseline is NOT a clean null
here. In a block-factor market a name's own trailing return is correlated with its
neighbours' (they comove), so own-momentum partially proxies the planted spillover
and the blindfolded model still scores positive. So the test asserts what is actually
true -- the graph adds signal ON TOP of that leaky baseline -- not the stronger (and
false) claim that only the graph recovers anything."""

import pytest

pytest.importorskip("torch")

from marketgnn.leadlag import make_synthetic_leadlag
from marketgnn.models.temporal import run_temporal


def test_temporal_graph_adds_incremental_ic_over_blindfold():
    prices, _vol, _sectors, _mkt, true_graph = make_synthetic_leadlag(n_assets=40, n_days=900, seed=1)
    table = run_temporal(prices, lambda asof: true_graph, warmup=200, epochs=60, seed=0)
    row = {r["graph"]: r for r in table.to_dict("records")}
    ic_graph = row["yes"]["mean_ic"]
    ic_none = row["no (self-loops)"]["mean_ic"]
    # the graph carries the planted lead-lag: clearly positive OOS IC ...
    assert ic_graph > 0.02
    # ... and adds signal beyond the (own-momentum-correlated, non-null) baseline.
    assert ic_graph > ic_none + 0.01
