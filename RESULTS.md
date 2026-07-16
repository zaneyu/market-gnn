# Results log

Honest running log of what the experiments actually show. Numbers are gross rank-IC,
out-of-sample under purged walk-forward, HAC t-stats, BH-FDR across the grid.

## Run 1 — ridge baseline, real data (2014–2024)

- **Universe:** 90 large-cap US names, *current* membership (survivorship-biased →
  **return claims restricted; volatility target is robust**). PIT membership is the fix (TODO).
- **Cadence:** weekly, 5-day-ahead labels, corr window 60d, k=10, 338 test weeks.
- **Model:** ridge (graph enters only via the neighbour-return feature — the full
  topology test needs the GNN).

| graph        | target | mean IC | HAC t | FDR sig |
|--------------|--------|---------|-------|---------|
| correlation  | ret    | +0.004  | 0.27  | no      |
| shrinkage    | ret    | +0.004  | 0.24  | no      |
| frozen       | ret    | +0.002  | 0.12  | no      |
| sector       | ret    | +0.003  | 0.21  | no      |
| none         | ret    | +0.004  | 0.25  | no      |
| correlation  | vol    | +0.470  | 49.4  | **yes** |
| shrinkage    | vol    | +0.470  | 49.4  | **yes** |
| frozen       | vol    | +0.471  | 49.5  | **yes** |
| sector       | vol    | +0.470  | 49.8  | **yes** |
| none         | vol    | +0.471  | 49.5  | **yes** |

**Read:**
1. **Returns are unpredictable** here — every graph kind ≈ 0, none significant. Expected.
2. **Volatility is highly predictable** (clustering: trailing vol → forward vol). Anchor OK.
3. **The graph adds nothing (this model).** Dynamic correlation ≈ frozen static ≈ *no graph*.
   The weekly correlation-graph churn does not earn its keep; a frozen graph does as well,
   and so does none. Shrinkage doesn't move it because there was no lift to recover.

**Caveats / next:**
- Ridge sees the graph only through one engineered feature. The topology test is the
  GNN≡MLP A/B — run next on this data (synthetic GNN already showed mild over-smoothing,
  i.e. graph *hurting*).
- vol IC ≈ 0.47 is large but consistent with volatility persistence; leak/PIT tests pass.
  A robustness check (longer horizon, QLIKE vs a naive trailing-vol forecast) is on the list.
- Survivorship: current-membership universe. Volatility conclusions are robust to it;
  return conclusions are held pending PIT membership.

*Interpretation stance:* "the graph doesn't help, and here's the frozen-graph control that
proves the churn isn't the point" is the finding — not a failure.

## Run 2 — GNN vs matched MLP, real data (the topology test)

Same universe/cadence. GNN ≡ MLP + graph-conv (shared head, ranking loss, purged-val early
stopping, equal budget). `none` is the integrity check; `correlation` is the real test.

| graph        | model | target | mean IC | HAC t | FDR sig |
|--------------|-------|--------|---------|-------|---------|
| none         | mlp   | vol    | +0.483  | 46.8  | yes     |
| none         | gnn   | vol    | +0.483  | 46.8  | yes     |
| correlation  | mlp   | vol    | +0.489  | 44.5  | yes     |
| correlation  | gnn   | vol    | +0.486  | 42.9  | yes     |
| correlation  | mlp   | ret    | +0.001  | 0.07  | no      |
| correlation  | gnn   | ret    | +0.009  | 0.49  | no      |

**Read:**
1. **Integrity check passes on real data:** `none` GNN == MLP to the digit (0.483/0.483) —
   zeroing the graph recovers the MLP exactly. The A/B is clean.
2. **Topology adds nothing.** On volatility, message passing over the correlation graph makes
   it *slightly worse* (0.489 → 0.486) — over-smoothing, as predicted. On returns, both are
   indistinguishable from zero (GNN +0.009, HAC t 0.49, n.s.).

**Bottom line across both runs:** the graph does not earn its keep on this data — not as a
frozen or dynamic feature (Run 1), not as GNN topology (Run 2). That is the honest, controlled
answer. The lead-lag / temporal channel (v2) remains the one untested route by which a graph
could plausibly help returns.

## Run 3 — volatility robustness: model vs naive random-walk forecast

Is the ~0.48 vol IC real skill or just volatility persistence? Compared ridge against the
naive forecast "forward vol = trailing 20d vol" on the same OOS observations.

| metric              | model  | naive  | delta   |
|---------------------|--------|--------|---------|
| rank-IC             | +0.479 | +0.435 | +0.044  |
| QLIKE (lower=better)| 1.79   | 0.658  | +1.13   |

**Read:** the naive forecast alone gets **0.435 of the 0.479** — ~90% of the vol
"predictability" is just **volatility persistence**, not model skill. The model adds a small,
genuine ranking increment (**+0.044**) but is **worse on level accuracy (QLIKE)** — ridge on
cross-sectionally-normalized features ranks marginally better while producing a badly-scaled
level forecast. **Do not oversell the 0.48**: the honest statement is "cross-sectional vol is
highly persistent; the model adds a modest ranking edge over trailing vol and does not improve
the calibrated level forecast." (`python -m marketgnn.robustness`)

## Run 4 — point-in-time S&P 500 membership (inclusion-bias correction)

Reconstructed PIT membership from the public S&P 500 change log (Wikipedia) and re-ran with
each cross-section restricted to as-of members. 7 of the 90 names were added mid-window
(TSLA 2020-12, AMD 2017-03, PYPL 2015-07, NOW, TMUS, GOOGL, AVGO); universe grows 83 -> 90.

| graph        | target | IC (no membership) | IC (PIT membership) |
|--------------|--------|--------------------|---------------------|
| correlation  | ret    | +0.010 (t 0.65)    | +0.008 (t 0.49)     |
| none         | ret    | +0.010 (t 0.63)    | +0.004 (t 0.25)     |
| correlation  | vol    | +0.479             | +0.473              |
| none         | vol    | +0.480             | +0.474              |

**Read:** correcting inclusion timing **shrinks the (already-insignificant) return IC** — a
small survivorship inflation removed, in the expected direction. Volatility and the core
conclusion (graph doesn't help) are unchanged.

**LIMITATION (blunt):** this corrects inclusion *timing* for current members. It does NOT
restore delisted names — yfinance has no prices for them — so the run is inclusion-corrected
but **not fully survivorship-free**. That needs a vendor with delisted prices (e.g. CRSP).
The machinery (`apply_delistings`, membership mask) is built and tested; the binding
constraint is the data source, and it's stated rather than papered over.

## Run 5 — the lead-lag / spillover channel (the one that should carry return signal)

Runs 1-2 tested the *contemporaneous* graph — the configuration that structurally cannot
carry return signal. The channel that can is **lead-lag / momentum spillover**: neighbours'
*past* returns predicting a name's *future* return (Cohen-Frazzini economic links,
Menzly-Ozbas cross-industry lead-lag, Moskowitz-Grinblatt industry momentum). Signal at t =
edge-weighted mean of neighbours' return over (t-h, t]; target = own return over (t, t+h].
Zero parameters — nothing to overfit.

**First, prove the pipeline can detect it (planted-signal recovery, synthetic).** A synthetic
market with a *planted* block-lead-lag effect: `python -m marketgnn.leadlag --synthetic-planted`

| edges  | signal        | mean IC | HAC t | MDE₈₀ | FDR sig |
|--------|---------------|---------|-------|-------|---------|
| sector | leadlag       | +0.089  | 9.59  | 0.028 | **yes** |
| sector | leadlag_resid | +0.085  | 9.47  | 0.027 | **yes** |
| rewire | leadlag       | −0.001  | −0.49 | 0.021 | no      |

The pipeline **recovers** the planted effect over the true graph (well above MDE), it
**survives** the own-momentum control (`_resid`), and the degree-preserving **rewire null
finds nothing**. So a null on real data means *absent*, not *undetectable*.

**Then real data (90 large-caps, 2014-2024, weekly):**

| edges       | signal        | mean IC | HAC t | MDE₈₀ | FDR sig |
|-------------|---------------|---------|-------|-------|---------|
| sector      | leadlag       | +0.006  | 0.64  | 0.027 | no      |
| sector      | leadlag_resid | +0.008  | 1.17  | 0.022 | no      |
| correlation | leadlag       | −0.008  | −0.75 | 0.032 | no      |
| rewire      | leadlag       | +0.000  | −0.01 | 0.014 | no      |

**Read — a POWERED null, which is the actual result:** we had 80% power to detect an IC of
~0.03 (the lead-lag literature implies ~0.02-0.04) and found ~0.006. Industry/correlation
lead-lag is **not present in liquid large-caps at weekly frequency over 2014-2024** — fully
consistent with the literature, where the effect lives in **small, illiquid names** (which
this universe excludes by construction) and has decayed post-2000. The rewire null and the
planted-recovery bracket the claim: the harness would have seen it; it isn't there.

## Run 6 — real-data positive control: the harness DOES find a non-null

Are the graph nulls real absences, or can this pipeline just not find return signal?
Answer by pointing the same leak-free HAC harness at pre-specified, textbook cross-sectional
anomalies (not discovered or tuned here). `python -m marketgnn.signals`

| signal        | mean IC | HAC t | p      | turnover | FDR sig |
|---------------|---------|-------|--------|----------|---------|
| **reversal_1d** | **+0.0151** | **+3.39** | **0.0007** | 0.33 | **YES** |
| reversal_1w   | +0.0002 | +0.02 | 0.99   | 0.33     | no      |
| momentum_12_1 | +0.0089 | +0.42 | 0.67   | 0.10     | n.s.    |

**Read — a genuine non-null.** Short-term (1-day) reversal (Lehmann 1990 / Jegadeesh 1990) is
**strongly significant** on real data — IC +0.015, HAC t 3.4, p<0.001 over 2,506 non-overlapping
daily cross-sections, clearing FDR. So the pipeline finds real return signal *where it exists*;
the graph nulls are genuine absences, not a broken harness. Note it decays by the weekly horizon
(reversal_1w null) — which is exactly why the weekly graph runs saw nulls.

**Cost model makes the caveat a number (`python -m marketgnn.costs`).** The reversal_1d decile
long-short is +0.38 Sharpe **gross** but its **breakeven cost is 0.9 bps** — below the ~2–5 bp
large-cap round-trip — so net Sharpe is **−1.77 at 5 bps**, −3.9 at 10 bps. It is a genuine
statistical effect (real return signal exists) that is **arbitraged net of costs**: a positive
control, not a tradable strategy. Momentum_12-1 has lower turnover (breakeven ~12 bps) but isn't
statistically significant here. Reporting the effect *and* exactly where costs kill it is the point.

## Run 7 — extended universe (194 names) + liquidity-conditioned reversal

Widened to ~180 mid/large names (all public pre-2014, so no IPO gaps), giving a real
liquidity range and more power. Short-term reversal is documented to be an illiquidity /
arbitrage-cost effect, so it should concentrate in less-liquid names. Split each cross-section
into liquidity terciles (trailing dollar volume): `python -m marketgnn.conditioning`

| liquidity tercile | mean IC | HAC t | p       |
|-------------------|---------|-------|---------|
| least liquid      | +0.0193 | 4.18  | <0.001  |
| mid               | +0.0198 | 3.96  | <0.001  |
| most liquid       | +0.0132 | 2.83  | 0.005   |
| **low − high spread** | **+0.0061** | **1.45** | **0.15** |

**Read — honestly nuanced.** Reversal is **significant across all liquidity terciles** on 194
names (stronger t-stats than the 90-name run — power gained). It is directionally weaker in the
most-liquid names, but the **low-minus-high gradient is *not* statistically significant** (p 0.15).
So the illiquidity-concentration hypothesis is **not confirmed** here — expected, because even the
"least liquid" tercile of a mid/large-cap universe is still fairly liquid; the documented gradient
lives in genuinely small/illiquid names this set still excludes. Refusing to claim the gradient it
doesn't support is the point. Figures: `figures/reversal_by_liquidity.png`, `reversal_cost_decay.png`.

## Run 8 — a REAL economic-link graph: 13F institutional co-holding

Runs 1–5 tested the graph as a **correlation/industry proxy** for economic linkage. The
lead-lag literature (Anton–Polk, "Connected Stocks", JF 2014) points at a *genuine* link:
stocks held by the **same institutions** comove and lead-lag through correlated fund flows.
So build that graph from the actual filings and run the identical harness on it. Source:
the SEC's structured **13F** datasets — 11 year-end snapshots (Dec-2013 … Dec-2023), **~2,500–
5,800 filing institutions per snapshot** (median ~3,600, printed at runtime), covering all 90
names in the latest snapshot (fewer in the earliest years, before names like META/ABBV/PYPL were
public — a coverage figure the runner prints, not folklore). Two stocks are linked by the cosine
of their binary institutional-holder vectors (kNN, k=10; avg degree ~16.9). **Point-in-time by
construction:** the `{Y}q1` dataset holds Dec-(Y-1) positions filed within the 45-day deadline
and public from ~mid-Feb Y, and each snapshot graph is only applied to dates on/after its public
date. Two PIT guards in the loader make that exact: an **exact `13F-HR`** match (restated
amendments excluded) and a **`FILING_DATE <= public`** filter (tardy filers not yet public are
dropped). (`python -m marketgnn.coholding`; ticker→CUSIP in the committed, auditable,
check-digit-validated `data/cusip_map.csv`.)

**First, power — plant a lead-lag ALONG the real co-holding graph and recover it:**

| edges       | signal        | mean IC | HAC t | MDE₈₀ | FDR sig |
|-------------|---------------|---------|-------|-------|---------|
| coholding   | leadlag       | +0.027  | 4.05  | 0.020 | **yes** |
| coholding   | leadlag_resid | +0.026  | 4.02  | 0.020 | **yes** |
| rewire      | leadlag       | −0.007  | −2.59 | 0.018 | no      |

The pipeline **recovers** a planted effect over *this* graph and the degree-preserving rewire
finds nothing — so, as everywhere in this repo, a null on the real data is *absent*, not
*undetectable*.

**Then real data (90 large-caps, 2014–2024, weekly):**

| edges       | signal        | mean IC | HAC t | p    | MDE₈₀ | FDR sig |
|-------------|---------------|---------|-------|------|-------|---------|
| coholding   | leadlag       | +0.012  | 1.40  | 0.16 | 0.025 | no      |
| coholding   | leadlag_resid | +0.011  | 1.57  | 0.12 | 0.021 | no      |
| rewire      | leadlag       | +0.003  | +0.89 | 0.37 | 0.014 | no      |

**Read — a *powered* null on the real economic link, which is the stronger result.** Even the
genuine co-holding graph the literature points to adds **no lead-lag return signal that
survives** on liquid large-caps: IC +0.012 with 80% power to detect ~0.025, not significant,
rewire-null clean, above the own-momentum control. It is directionally the *warmest* linkage
tested (+0.012 vs the sector/correlation proxies' +0.006 / −0.008 in Run 5) — consistent with a
real-but-tiny effect — but it does not clear the bar. This closes the "maybe the proxy graph
was hiding a real link" escape hatch: the real link is measured, powered, and still null here.
(On these mega/large-caps the co-holding graph is also very *central* — nearly every institution
holds AAPL/MSFT, so the biggest names neighbour everything; the Anton–Polk effect is documented
to be strongest in less-universally-held names, which this universe excludes by construction.)

## Run 9 — a LEARNED temporal graph model (GConvGRU)

The zero-parameter lead-lag signal (Runs 5, 8) is a *fixed* linear read of neighbours'
past returns. A fair objection: maybe the linkage signal is there but non-linear or
needs a longer memory, and only a *learned* temporal model would find it. So build one —
a **graph-conv spatial layer per date feeding a GRU over the date sequence** (GConvGRU,
Seo et al. 2018), the model whose absence the plan flagged. The A/B is the repo's usual
blindfold: `use_graph=False` swaps the real edges for self-loops, so architecture,
parameter count, GRU, head and loss are identical and only topology changes.
(`python -m marketgnn.models.temporal`.)

**First, power — recover the planted lead-lag (synthetic):**

| graph            | mean IC (OOS) | HAC t | p      |
|------------------|---------------|-------|--------|
| yes              | +0.067        | 3.91  | <0.001 |
| no (self-loops)  | −0.012        | −0.94 | 0.35   |

The learned model **recovers** the planted spillover *only when it can see the graph*;
blindfolded it finds nothing. So the model has the capacity and the graph is what
carries the signal — the A/B is doing real work.

**Then real data over the real 13F co-holding graph (90 large-caps, 2014–2024):**

| graph            | mean IC (OOS) | HAC t | p    |
|------------------|---------------|-------|------|
| yes (co-holding) | +0.006        | 0.44  | 0.66 |
| no (self-loops)  | +0.007        | 0.51  | 0.61 |

**Read — the null survives a learned model that CAN see time.** Given the real economic
link *and* the capacity to exploit temporal structure, the GConvGRU's graph variant scores
+0.006 IC — if anything *below* its own graph-blindfolded twin (+0.007), and neither clears
significance (t < 1). So the earlier lead-lag nulls were **not a modelling limitation** — a
linear zero-parameter read and a learned spatiotemporal network reach the same answer on these
names. Same conclusion, one more escape hatch (\"you just needed a bigger model\") closed.

## Headline (nine runs)

The contemporaneous graph is a **red herring** (Runs 1-2: frozen ≈ dynamic ≈ none; GNN ≈ MLP).
The lead-lag channel — the only one that *should* carry return signal — is a **powered null on
liquid large-caps**, whether the edges are the correlation/industry proxy (Run 5) or a **real
economic link (13F institutional co-holding, Run 8)** — both validated by a planted-signal
recovery over that exact graph, so the null is absence, not blindness. The vol "predictability"
is ~90% persistence (Run 3), and inclusion-timing correction shrinks the already-null return IC
(Run 4).

A **learned spatiotemporal model** (GConvGRU, Run 9) — given the real link *and* the
capacity to use time — reaches the same null (graph +0.006 vs no-graph +0.007, both n.s.),
after recovering a planted effect (+0.067, t 3.9); so the null is not a modelling limitation.

The **pre-registered primary endpoint (H1)** is tested directly, not eyeballed: the *paired*
per-date IC-difference series (GNN minus MLP on the same dates/universe) with a HAC t and
block-bootstrap CI (`train.primary_endpoint`). It isolates the incremental value of *message-
passing* over an already-graph-informed MLP (both consume the neighbour-return feature).
Synthetic: ΔIC +0.005, t 0.33, p 0.75 — no GNN-over-MLP gap. Honest power note: the MDE (~0.047)
sits *above* a plausible message-passing edge (~0.005–0.02), so H1 is **underpowered for a small
edge** — it is consistent with, but does not rule out, one. The broader "graph adds return
signal" question is carried by the zero-parameter lead-lag test (Runs 5/8) and its planted-signal
power, not by H1.

**What DOES carry return signal (Run 6):** the same harness finds **short-term (1-day) reversal
strongly significant on real data** (IC +0.015, HAC t 3.4, p<0.001, FDR-sig) — a genuine
non-null (though a *survivorship-exposed* positive claim: unlike the graph nulls, bias here runs
toward finding signal, so treat it as a harness-can-find-things control, not a clean tradable
result — a fully PIT reversal run needs delisted-name prices). So the nulls above are **real
absences bracketed by positive controls on both real data (reversal) and synthetic data
(planted lead-lag)**, not a pipeline that can't find anything.

Net: **the graph adds no cross-sectional return signal that survives leak-free, powered,
control-checked evaluation; the return signal that IS present (short-horizon reversal) is
non-graph and cost-fragile — and here is the harness that tells real effects apart from the
survivorship / overlap / over-smoothing artifacts that make graph "alpha" look real.**
