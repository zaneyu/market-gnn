"""Temporal (spatiotemporal) GNN -- DEFERRED, not built. See PLAN.md.

The static model here captures *contemporaneous* relative-strength. The
economically interesting return channel is lead-lag / momentum spillover
(neighbour's return at t -> own return at t+h), which is inherently temporal and
cannot be expressed by same-date message passing. The honest v2 is a GConvGRU-style
model: a GNN spatial layer per snapshot, its embeddings fed through a GRU over a
window of dates. Left as a two-line footnote deliberately -- shipping a finished,
tight static study beats a half-built temporal one.
"""


class TemporalGNN:
    def __init__(self, *_, **__):
        raise NotImplementedError(
            "Temporal GNN is a documented v2 extension; the MVP tests the "
            "contemporaneous graph hypothesis only. See PLAN.md > Lead-lag channel."
        )
