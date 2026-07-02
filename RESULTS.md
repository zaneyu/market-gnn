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
