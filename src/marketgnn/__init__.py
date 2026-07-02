"""market-gnn: does graph structure add out-of-sample signal over a matched
non-graph model? The whole package is built to answer that honestly and
leak-free. See README.md for the research question and the ablation it drives.
"""

from .splits import Fold, PurgedWalkForward

__all__ = ["PurgedWalkForward", "Fold"]
__version__ = "0.1.0"
