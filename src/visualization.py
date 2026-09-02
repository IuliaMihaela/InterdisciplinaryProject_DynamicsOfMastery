"""
This module contains visualization helpers for the learning curve analysis.
It generates report-ready Matplotlib figures for raw trajectories, model fits,
residual series, and Detrended Fluctuation Analysis (DFA) scaling.
"""

import matplotlib.pyplot as plt
import numpy as np


def plot_learning_curve(x, y, player_id=None):
    """
    Plot raw learning trajectory.
    """

    fig, ax = plt.subplots(figsize=(10, 5))

    ax.plot(
        x,
        y,
        marker=".",
        linestyle="-",
        alpha=0.7
    )

    ax.set_xlabel("Solve Index")
    ax.set_ylabel("Best Solve Time (seconds)")
    ax.set_title(
        f"Learning Trajectory ({player_id})"
        if player_id
        else "Learning Trajectory"
    )

    ax.grid(True, alpha=0.3)

    return fig, ax



def plot_model_fits(
    x,
    y,
    power_pred,
    exp_pred,
    player_id=None
):
    """
    Plot observed data together with fitted models.
    """

    fig, ax = plt.subplots(figsize=(10, 5))

    ax.scatter(
        x,
        y,
        s=10,
        alpha=0.4,
        label="Observed"
    )

    ax.plot(
        x,
        power_pred,
        linewidth=2,
        label="Power Law"
    )

    ax.plot(
        x,
        exp_pred,
        linewidth=2,
        label="Exponential"
    )

    ax.set_xlabel("Solve Index")
    ax.set_ylabel("Best Solve Time (seconds)")

    title = "Learning Curve Model Comparison"
    if player_id:
        title += f" ({player_id})"

    ax.set_title(title)

    ax.legend()

    ax.grid(True, alpha=0.3)

    return fig, ax

def plot_dfa_scaling(scales, fluctuations, slope=None, intercept=None, player_id=None):
    """Plot DFA scaling on log-log axes with an optional fitted line."""

    scales = np.asarray(scales, dtype=float)
    fluctuations = np.asarray(fluctuations, dtype=float)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.scatter(scales, fluctuations, s=24, alpha=0.75, label="Observed fluctuations")

    if slope is not None and intercept is not None:
        fitted = np.exp(intercept + slope * np.log(scales))
        ax.plot(scales, fitted, linewidth=2, label="Fitted line")

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Scale n")
    ax.set_ylabel("Fluctuation F(n)")
    ax.grid(True, which="both", alpha=0.3)

    title = "DFA Scaling"
    if player_id:
        title += f" ({player_id})"
    ax.set_title(title)
    ax.legend()

    return fig, ax


def plot_residual_series(x, residuals, player_id=None):
    """Plot residuals against solve index."""

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(x, residuals, marker=".", linestyle="-", alpha=0.8)
    ax.axhline(0, linestyle="--", color="black", linewidth=1)
    ax.grid(True, alpha=0.3)

    ax.set_xlabel("Solve Index")
    ax.set_ylabel("Residual")

    title = "Residual Series"
    if player_id:
        title += f" ({player_id})"
    ax.set_title(title)

    return fig, ax


def plot_distribution_overlay(early_values, late_values, player_id=None, bins=None):
    """Plot early-vs-late performance distributions with mean markers."""

    early_values = np.asarray(early_values, dtype=float)
    late_values = np.asarray(late_values, dtype=float)

    if bins is None:
        # Keep behavior close to notebook defaults while adapting to sample size.
        bins = max(12, int(np.sqrt(min(len(early_values), len(late_values)))))

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(early_values, bins=bins, alpha=0.55, label="Early (first 25%)", density=True)
    ax.hist(late_values, bins=bins, alpha=0.55, label="Late (last 25%)", density=True)
    ax.axvline(np.mean(early_values), linestyle="--", color="C0", linewidth=1)
    ax.axvline(np.mean(late_values), linestyle="--", color="C1", linewidth=1)

    ax.set_xlabel("Solve time (seconds)")
    ax.set_ylabel("Density")

    title = "Early vs Late Distribution"
    if player_id:
        title += f" ({player_id})"
    ax.set_title(title)

    ax.legend()
    ax.grid(True, alpha=0.3)

    return fig, ax


__all__ = [
    "plot_learning_curve",
    "plot_model_fits",
    "plot_dfa_scaling",
    "plot_residual_series",
    "plot_distribution_overlay",
]