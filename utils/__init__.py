"""Utility modules for the WikiArt artist classification project.

Exposes:
    plot_history     (from .plotting)     training curve plotter
    evaluate_model   (from .evaluation)   test-set metrics and confusion matrix
"""

from .evaluation import evaluate_model
from .plotting import plot_history

__all__ = ["evaluate_model", "plot_history"]
