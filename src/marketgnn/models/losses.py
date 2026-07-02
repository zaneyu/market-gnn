"""Differentiable ranking loss. The metric is rank-IC, so both neural models train
on a soft-rank objective rather than MSE -- otherwise the GNN-vs-MLP comparison
would be measuring fit to a mismatched surrogate. Torch is imported lazily by the
neural models, so this module is only loaded when torch is present."""

from __future__ import annotations

import torch


def soft_spearman_loss(pred: torch.Tensor, target: torch.Tensor, *, reg: float = 1e-2) -> torch.Tensor:
    """1 - (differentiable) Spearman correlation within one cross-section.

    Soft ranks via a temperature-smoothed pairwise comparison, then Pearson on the
    soft ranks. Minimizing this maximizes rank agreement -- the quantity IC measures.
    """
    n = pred.shape[0]
    if n < 3:
        return pred.sum() * 0.0

    def soft_rank(x: torch.Tensor) -> torch.Tensor:
        diff = x[:, None] - x[None, :]
        # sigmoid(diff/reg) ~ 1 when x_i > x_j; row-sum = soft rank position
        return torch.sigmoid(diff / reg).sum(dim=1)

    pr, tr = soft_rank(pred), soft_rank(target)
    pr = pr - pr.mean()
    tr = tr - tr.mean()
    denom = torch.sqrt((pr @ pr) * (tr @ tr)) + 1e-8
    corr = (pr @ tr) / denom
    return 1.0 - corr
