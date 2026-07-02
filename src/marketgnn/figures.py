"""Generate the figures referenced in the README / note. Visual evidence lands
harder than tables: the reversal-by-liquidity gradient and the cost-decay curve
that shows exactly where the gross signal dies. Writes PNGs to figures/.

Run: `python -m marketgnn.figures` (needs the `viz` extra: matplotlib).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

FIGDIR = Path(__file__).resolve().parents[2] / "figures"
_AMBER, _INK, _MUTED = "#c9860f", "#2b2620", "#8a8172"


def _style(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.tick_params(colors=_INK, labelsize=9)
    ax.yaxis.label.set_color(_INK)
    ax.title.set_color(_INK)


def plot_reversal_by_liquidity(table, path: Path):
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(5.2, 3.4), dpi=140)
    labels = table["label"].tolist()
    ic = table["mean_ic"].to_numpy()
    ax.bar(labels, ic, color=_AMBER, width=0.6)
    for i, (v, t) in enumerate(zip(ic, table["hac_t"])):
        ax.text(i, v + 0.0004, f"t={t:.1f}", ha="center", fontsize=8, color=_MUTED)
    ax.axhline(0, color=_MUTED, lw=0.8)
    ax.set_ylabel("mean rank-IC")
    ax.set_title("Short-term reversal by liquidity (gross)")
    _style(ax)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_cost_decay(cost, path: Path, cost_grid=(0, 1, 5, 10, 20)):
    import matplotlib.pyplot as plt

    sharpe = [cost["sharpe_gross"] if c == 0 else cost.get(f"sharpe_net_{c}bps") for c in cost_grid]
    fig, ax = plt.subplots(figsize=(5.2, 3.4), dpi=140)
    ax.plot(cost_grid, sharpe, "-o", color=_AMBER, lw=2, ms=5)
    ax.axhline(0, color=_MUTED, lw=0.8)
    be = cost["breakeven_bps"]
    ax.axvline(be, color=_INK, ls="--", lw=1)
    ax.text(be + 0.3, min(sharpe), f"breakeven {be:.1f} bps", fontsize=8, color=_INK)
    ax.set_xlabel("transaction cost (bps)")
    ax.set_ylabel("annualized Sharpe")
    ax.set_title("Reversal long-short: Sharpe vs cost")
    _style(ax)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def main():
    from .conditioning import run_liquidity_conditioning
    from .costs import cost_summary
    from .data.download import download_prices
    from .data.universe import extended_universe
    from .signals import signal_panel

    FIGDIR.mkdir(exist_ok=True)
    prices, volume, _ = download_prices(extended_universe(), "2014-01-01", "2024-12-31", cache_key="extended")
    spec = {"name": "reversal_1d", "lookback": 1, "horizon": 1, "sign": -1}

    table, _ = run_liquidity_conditioning(prices, volume, spec=spec)
    plot_reversal_by_liquidity(table, FIGDIR / "reversal_by_liquidity.png")
    plot_cost_decay(cost_summary(signal_panel(prices, spec)), FIGDIR / "reversal_cost_decay.png")
    print(f"wrote figures to {FIGDIR}/")


if __name__ == "__main__":
    main()
