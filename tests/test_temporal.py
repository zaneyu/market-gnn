"""The temporal (GConvGRU) A/B has power: on a synthetic market with a PLANTED
lead-lag effect, the graph variant recovers it and the graph-blindfolded variant
(self-loops) does not. Torch-gated (skips in the core torch-free CI run)."""

import pytest

pytest.importorskip("torch")

from marketgnn.leadlag import make_synthetic_leadlag
from marketgnn.models.temporal import run_temporal


def test_temporal_recovers_planted_leadlag_only_with_graph():
    prices, _vol, _sectors, _mkt, true_graph = make_synthetic_leadlag(n_assets=40, n_days=900, seed=1)
    table = run_temporal(prices, lambda asof: true_graph, warmup=200, epochs=60, seed=0)
    row = {r["graph"]: r for r in table.to_dict("records")}
    ic_graph = row["yes"]["mean_ic"]
    ic_none = row["no (self-loops)"]["mean_ic"]
    # the graph must carry the planted lead-lag: clearly positive OOS IC, and it must
    # beat the graph-blindfolded model (whose only difference is the missing topology)
    assert ic_graph > 0.02
    assert ic_graph > ic_none + 0.01
