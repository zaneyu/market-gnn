"""A *real* economic-link graph from SEC 13F institutional co-holding, and the
lead-lag test on it -- the strongest form of the question this repo asks.

The correlation/sector graph is a *proxy* for economic linkage. Anton & Polk
("Connected Stocks", JF 2014) show a genuine one: stocks held by the *same*
institutions comove and lead-lag through correlated fund flows. This module builds
that graph from the SEC's structured 13F datasets and runs the identical leak-free
lead-lag harness on it. If even a real economic link adds no detectable signal on
these names, the null is stronger than the proxy-graph null; if it does, the proxy
was hiding a real effect. Either way it is a sharper result.

Point-in-time by construction
------------------------------
13F holdings are filed within 45 days of quarter-end. The SEC "{Y}q1" dataset holds
the Dec-(Y-1) period reports, filed Jan-Feb Y and public from ~mid-Feb Y. A
co-holding graph built from that snapshot is only ever APPLIED to dates on/after its
public date (`_public`), so the graph never sees information unavailable at the time
-- the same PIT discipline as the rest of the study. Institutional portfolios move
slowly, so an annually-rebuilt graph is a faithful, low-variance linkage estimate.

Data is fetched from the SEC, not shipped (it is public but large). The committed
`data/cusip_map.csv` (ticker -> CUSIP, auditable) is the only issuer identification
needed; everything else is derived from the raw filings.
"""

from __future__ import annotations

import io
import os
import zipfile
from pathlib import Path
from urllib.request import Request, urlopen

import numpy as np
import pandas as pd

from . import graph as G

SEC_UA = os.environ.get("SEC_UA", "market-gnn research (zaneyu2005@gmail.com)")
_BASE = "https://www.sec.gov/files/structureddata/data/form-13f-data-sets"


def _snapshots() -> list[dict]:
    """One year-end snapshot per year: the {Y}q1 dataset carries Dec-(Y-1) holdings,
    public ~mid-Feb Y. 2024's dataset uses the newer date-range file name."""
    snaps = []
    for y in range(2014, 2024):  # Dec-2013 ... Dec-2022
        snaps.append({"period": f"31-DEC-{y - 1}", "public": pd.Timestamp(f"{y}-02-15"),
                      "zip": f"{y}q1.zip", "url": f"{_BASE}/{y}q1_form13f.zip"})
    snaps.append({"period": "31-DEC-2023", "public": pd.Timestamp("2024-02-15"),
                  "zip": "2024q1.zip", "url": f"{_BASE}/01jan2024-29feb2024_form13f.zip"})
    return snaps


def _cache_dir() -> Path:
    d = Path(__file__).resolve().parents[2] / ".cache" / "13f"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _fetch(url: str, dest: Path) -> None:
    if dest.exists() and dest.stat().st_size > 100_000:
        return
    req = Request(url, headers={"User-Agent": SEC_UA})
    with urlopen(req, timeout=180) as r:
        dest.write_bytes(r.read())


def cusip_map() -> dict[str, str]:
    """ticker -> CUSIP, from the committed, human-auditable map (resolved once from
    the 13F issuer names; see data/cusip_map.csv)."""
    p = Path(__file__).resolve().parent / "data" / "cusip_map.csv"
    m = pd.read_csv(p, dtype=str)
    return dict(zip(m["ticker"], m["cusip"]))


def _read_tsv(zf: zipfile.ZipFile, name: str, usecols) -> pd.DataFrame:
    with zf.open(name) as fh:
        return pd.read_csv(io.TextIOWrapper(fh, "utf-8"), sep="\t", usecols=usecols,
                           dtype=str, na_filter=False)


def load_holders(snap: dict, cmap: dict[str, str]) -> pd.DataFrame:
    """Distinct (cik, ticker) institutional-holder pairs for the universe in one
    snapshot: original 13F-HR share positions (options and amendments excluded),
    restricted to the year-end period AND to filings actually public by the snapshot's
    public date, mapped from CUSIP to ticker. Cached (small) so reruns don't re-parse
    the ~250 MB info table.

    Two PIT guards live here: (1) ``SUBMISSIONTYPE == "13F-HR"`` is an *exact* match, so
    restated amendments (13F-HR/A) that were not public at the original deadline never
    enter the graph; (2) ``FILING_DATE <= public`` drops late/tardy filers that a
    reader on the public date could not yet have seen. Both bound the look-ahead the
    naive "everything in the quarterly file" approach would admit.

    The cache key carries a schema version (``_v2``) so changing the filter logic
    invalidates stale caches automatically rather than silently returning pre-guard data."""
    cache = _cache_dir() / f"holders_v2_{snap['period']}.csv"
    if cache.exists():
        return pd.read_csv(cache, dtype={"cik": str, "ticker": str})

    zpath = _cache_dir() / snap["zip"]
    _fetch(snap["url"], zpath)
    cusip_to_ticker = {v: k for k, v in cmap.items()}
    universe_cusips = set(cusip_to_ticker)
    with zipfile.ZipFile(zpath) as zf:
        info = _read_tsv(zf, "INFOTABLE.tsv",
                         ["ACCESSION_NUMBER", "CUSIP", "SSHPRNAMTTYPE", "PUTCALL"])
        sub = _read_tsv(zf, "SUBMISSION.tsv",
                        ["ACCESSION_NUMBER", "FILING_DATE", "SUBMISSIONTYPE", "CIK", "PERIODOFREPORT"])
    info = info[(info["SSHPRNAMTTYPE"] == "SH") & (info["PUTCALL"] == "")]
    info = info[info["CUSIP"].isin(universe_cusips)]
    filing_date = _parse_sec_date(sub["FILING_DATE"])
    if filing_date.isna().all():
        raise ValueError(f"all FILING_DATE values failed to parse for {snap['period']}; "
                         f"refusing to build an empty (all-isolated) co-holding graph")
    sub = sub[(sub["SUBMISSIONTYPE"] == "13F-HR")
              & (sub["PERIODOFREPORT"] == snap["period"])
              & (filing_date <= snap["public"])]
    merged = info.merge(sub[["ACCESSION_NUMBER", "CIK"]], on="ACCESSION_NUMBER", how="inner")
    merged["ticker"] = merged["CUSIP"].map(cusip_to_ticker)
    out = merged[["CIK", "ticker"]].drop_duplicates().rename(columns={"CIK": "cik"})
    if out.empty:
        raise ValueError(f"no 13F-HR holders parsed for {snap['period']} — likely a filter "
                         f"or identifier bug; refusing to cache an empty result")
    out.to_csv(cache, index=False)
    return out


# SEC FILING_DATE is always "DD-MON-YYYY" with an ENGLISH uppercase month abbreviation,
# regardless of machine locale. Parse it with an explicit month map so a non-English
# LC_TIME can't coerce every date to NaT (which would silently empty the graph).
_MONTHS = {"JAN": "01", "FEB": "02", "MAR": "03", "APR": "04", "MAY": "05", "JUN": "06",
           "JUL": "07", "AUG": "08", "SEP": "09", "OCT": "10", "NOV": "11", "DEC": "12"}


def _parse_sec_date(s: pd.Series) -> pd.Series:
    """Locale-independent parse of SEC 'DD-MON-YYYY' dates -> Timestamps (NaT on junk)."""
    parts = s.str.upper().str.split("-", expand=True)
    iso = parts[2] + "-" + parts[1].map(_MONTHS) + "-" + parts[0]
    return pd.to_datetime(iso, format="%Y-%m-%d", errors="coerce")


def holders_covered(snap: dict, cmap: dict[str, str]) -> int:
    """How many universe tickers have >=1 institutional holder in this snapshot. A
    value below len(universe) means a CUSIP-map entry failed to match any 13F row
    (wrong/stale identifier) and that name is an isolated node -- the exact failure the
    committed map must avoid."""
    h = load_holders(snap, cmap)
    return int(h["ticker"].nunique())


def coholding_links(holders: pd.DataFrame, nodes, *, k: int = 10, min_common: int = 5) -> pd.DataFrame:
    """kNN links from institutional co-holding. Two stocks are similar when the same
    institutions hold both: cosine of their binary holder vectors,
    S_ij = |holders_i & holders_j| / sqrt(|holders_i| * |holders_j|). Each node keeps
    its top-k neighbours (symmetric union), matching the correlation graph's shape so
    the only thing that changes is *which* linkage defines the edges."""
    nodes = list(nodes)
    pos = {t: i for i, t in enumerate(nodes)}
    holders = holders[holders["ticker"].isin(pos)]
    # incidence: rows = filers, cols = stocks (binary)
    ciks = holders["cik"].unique()
    cpos = {c: i for i, c in enumerate(ciks)}
    M = np.zeros((len(ciks), len(nodes)), dtype=np.float64)
    for cik, tk in holders.itertuples(index=False):
        M[cpos[cik], pos[tk]] = 1.0
    common = M.T @ M                      # [n, n] shared-holder counts
    h = np.diag(common).copy()            # holders per stock
    denom = np.sqrt(np.outer(h, h))
    with np.errstate(divide="ignore", invalid="ignore"):
        S = np.where(denom > 0, common / denom, 0.0)
    S[common < min_common] = 0.0          # ignore edges with too few common holders
    np.fill_diagonal(S, 0.0)
    src, dst, wt = [], [], []
    for i in range(len(nodes)):
        order = np.argsort(S[i])[::-1]
        for j in order[:k]:
            if S[i, j] <= 0:
                break
            src.append(nodes[i]); dst.append(nodes[j]); wt.append(float(S[i, j]))
    return pd.DataFrame({"src": src, "dst": dst, "weight": wt})


def build_snapshot_graphs(nodes, *, k: int = 10, cmap: dict[str, str] | None = None) -> list[tuple[pd.Timestamp, G.Graph]]:
    """(public_date, co-holding Graph) for every year-end snapshot, PIT-ordered."""
    cmap = cmap or cusip_map()
    graphs = []
    for snap in _snapshots():
        holders = load_holders(snap, cmap)
        links = coholding_links(holders, nodes, k=k)
        graphs.append((snap["public"], G.edges_from_pairs(links, nodes, symmetric=True)))
    return graphs


def pick_asof(graphs: list[tuple[pd.Timestamp, G.Graph]], asof) -> G.Graph:
    """The most recent snapshot graph whose filings were already public by ``asof``
    (PIT). Before the first public date NOTHING is public yet, so return an *empty*
    graph -- never the earliest snapshot, whose Dec holdings were not filed until the
    following Feb and would be a look-ahead. (In practice ``warmup`` pushes the first
    evaluated date well past the first public date, but the boundary must be correct.)"""
    avail = [g for (pub, g) in graphs if pub <= pd.Timestamp(asof)]
    if avail:
        return avail[-1]
    nodes = list(graphs[0][1].nodes)
    return G.edges_from_pairs(pd.DataFrame({"src": [], "dst": []}), nodes)


def make_provider(nodes, *, k: int = 10, cmap: dict[str, str] | None = None):
    """PIT graph providers for run_leadlag: as-of a date, use the most recent snapshot
    whose filings were already public. Returns {'coholding': f, 'rewire': f_rewire}."""
    graphs = build_snapshot_graphs(nodes, k=k, cmap=cmap)

    def as_of(asof) -> G.Graph:
        return pick_asof(graphs, asof)

    def coholding(asof, seed=0):
        return as_of(asof)

    def rewire(asof, seed=0):
        return G.degree_preserving_rewire(as_of(asof), seed=seed)

    return {"coholding": coholding, "rewire": rewire}, graphs


def make_synthetic_coholding_planted(graph: G.Graph, *, n_days=1600, h=5, beta_ll=0.35,
                                     noise=0.012, seed=0):
    """Plant a lead-lag effect ALONG the real co-holding graph, so we can prove the
    pipeline recovers signal over *this* graph (power) before trusting a null on it.
    Each name's return depends on its co-holding neighbours' return h days earlier."""
    rng = np.random.default_rng(seed)
    nodes = list(graph.nodes)
    n = len(nodes)
    dates = pd.bdate_range("2015-01-01", periods=n_days)
    # row-normalized neighbour-averaging matrix from the co-holding edges
    A = np.zeros((n, n))
    src, dst = graph.edge_index
    np.add.at(A, (src, dst), np.abs(graph.edge_weight))
    rs = A.sum(1, keepdims=True)
    A = np.divide(A, rs, out=np.zeros_like(A), where=rs > 0)

    mkt = rng.normal(0.0003, 0.008, size=n_days)
    idio = rng.normal(0, noise, size=(n_days, n))
    r = np.zeros((n_days, n))
    for t in range(n_days):
        r[t] = mkt[t] + idio[t]
        if t >= h:
            r[t] += beta_ll * (A @ r[t - h])
    prices = pd.DataFrame(100 * np.exp(np.cumsum(r, axis=0)), index=dates, columns=nodes)
    sectors = pd.Series("blk", index=nodes, name="sector")  # unused by the provider path
    market = pd.Series(mkt, index=dates, name="MKT")
    return prices, sectors, market


def main():
    from .leadlag import run_leadlag

    cmap = cusip_map()
    nodes = list(cmap)  # the 90-name default universe, in map order
    print(f"Building PIT co-holding graphs for {len(nodes)} names from SEC 13F "
          f"({len(_snapshots())} year-end snapshots)...")
    provider, graphs = make_provider(nodes, k=10, cmap=cmap)
    degs = [g.avg_degree for _, g in graphs]
    # institution counts per snapshot, so "~N institutions" is auditable, not folklore
    insts = [load_holders(snap, cmap)["cik"].nunique() for snap in _snapshots()]
    covered = [holders_covered(snap, cmap) for snap in _snapshots()]
    print(f"co-holding graph avg degree {np.mean(degs):.1f} (per snapshot {degs[0]:.1f}..{degs[-1]:.1f})")
    print(f"distinct institutions (13F filers touching the universe): "
          f"{min(insts)}..{max(insts)} across snapshots, ~{int(np.median(insts))} median")
    print(f"universe names with >=1 holder: {min(covered)}..{max(covered)} of {len(nodes)} "
          f"(0 means a CUSIP-map miss -> isolated node)")

    # --- power: plant a lead-lag on the real co-holding graph and recover it ---
    print("\n=== A. planted-signal recovery on the REAL co-holding graph (power) ===")
    latest = graphs[-1][1]
    sprices, ssec, smkt = make_synthetic_coholding_planted(latest)
    sprov, _ = make_provider(nodes, k=10, cmap=cmap)
    tbl_pow = run_leadlag(sprices, ssec, smkt, edge_kinds=("coholding", "rewire"),
                          graph_provider=sprov, warmup=260)
    _show(tbl_pow)

    # --- real data: does the real economic link carry lead-lag signal? ---
    print("\n=== B. lead-lag over the REAL co-holding graph, real prices (2014-2024) ===")
    from .data.download import load_market

    prices, _vol, sectors, market = load_market(
        synthetic=False, tickers=nodes, start="2014-01-01", end="2024-12-31")
    tbl = run_leadlag(prices, sectors, market, edge_kinds=("coholding", "rewire"),
                      graph_provider=provider)
    _show(tbl)


def _show(table: pd.DataFrame) -> None:
    cols = ["edges", "signal", "mean_ic", "hac_t", "p", "mde_80", "n_dates", "fdr_sig"]
    print(table[cols].to_string(index=False, float_format=lambda v: f"{v:+.3f}"))


if __name__ == "__main__":
    main()
