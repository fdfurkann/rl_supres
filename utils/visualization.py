"""
Visualization utilities for trading results.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from typing import List, Optional, Dict, Any

try:
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    _MATPLOTLIB_AVAILABLE = True
except ImportError:
    _MATPLOTLIB_AVAILABLE = False


def _check_matplotlib():
    if not _MATPLOTLIB_AVAILABLE:
        raise ImportError(
            "matplotlib is required for visualization. "
            "Install it with: pip install matplotlib"
        )


def plot_equity_curve(
    equity_curve: List[float],
    title: str = "Equity Curve",
    figsize: tuple = (12, 5),
    save_path: Optional[str] = None,
) -> None:
    """
    Plot the equity curve over time.

    Args:
        equity_curve: List of equity values.
        title:        Plot title.
        figsize:      Matplotlib figure size.
        save_path:    If provided, save figure to this path instead of showing.
    """
    _check_matplotlib()
    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(equity_curve, linewidth=1.5, color="steelblue")
    ax.set_title(title)
    ax.set_xlabel("Step")
    ax.set_ylabel("Equity")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
        plt.close(fig)
    else:
        plt.show()


def plot_trades_on_price(
    df: pd.DataFrame,
    trades: list,
    title: str = "Trades on Price",
    figsize: tuple = (14, 6),
    save_path: Optional[str] = None,
) -> None:
    """
    Plot OHLCV close price with entry/exit markers for each trade.

    Args:
        df:         OHLCV DataFrame with integer or datetime index.
        trades:     List of closed Trade objects from PositionManager.
        title:      Plot title.
        figsize:    Matplotlib figure size.
        save_path:  Save path.
    """
    _check_matplotlib()
    close = df["close"].values
    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(close, color="gray", linewidth=0.8, alpha=0.7, label="Close")

    for t in trades:
        color = "green" if t.direction == "long" else "red"
        ax.axvline(t.entry_bar, color=color, alpha=0.4, linewidth=0.5)
        if t.exit_bar is not None:
            ax.axvline(t.exit_bar, color=color, alpha=0.2, linewidth=0.5, linestyle="--")
        ax.scatter(t.entry_bar, t.entry_price, marker="^" if t.direction == "long" else "v",
                   color=color, s=40, zorder=5)

    long_patch = mpatches.Patch(color="green", label="Long")
    short_patch = mpatches.Patch(color="red", label="Short")
    ax.legend(handles=[long_patch, short_patch])
    ax.set_title(title)
    ax.set_xlabel("Bar")
    ax.set_ylabel("Price")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
        plt.close(fig)
    else:
        plt.show()


def plot_drawdown(
    equity_curve: List[float],
    title: str = "Drawdown",
    figsize: tuple = (12, 4),
    save_path: Optional[str] = None,
) -> None:
    """
    Plot the rolling drawdown series.

    Args:
        equity_curve: List of equity values.
        title:        Plot title.
        figsize:      Figure size.
        save_path:    Save path.
    """
    _check_matplotlib()
    eq = np.asarray(equity_curve, dtype=np.float64)
    roll_max = np.maximum.accumulate(eq)
    dd = (roll_max - eq) / np.where(roll_max != 0, roll_max, 1.0) * 100

    fig, ax = plt.subplots(figsize=figsize)
    ax.fill_between(range(len(dd)), -dd, 0, color="red", alpha=0.4)
    ax.plot(-dd, color="red", linewidth=0.8)
    ax.set_title(title)
    ax.set_xlabel("Step")
    ax.set_ylabel("Drawdown (%)")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
        plt.close(fig)
    else:
        plt.show()


def plot_metrics_summary(
    metrics: Dict[str, Any],
    title: str = "Performance Metrics",
    figsize: tuple = (8, 5),
    save_path: Optional[str] = None,
) -> None:
    """
    Display a bar chart of selected performance metrics.

    Args:
        metrics:   Dict returned by utils.metrics.compute_metrics.
        title:     Plot title.
        figsize:   Figure size.
        save_path: Save path.
    """
    _check_matplotlib()
    display_keys = [
        "total_return_pct",
        "sharpe_ratio",
        "max_drawdown_pct",
        "win_rate_pct",
        "profit_factor",
    ]
    labels = []
    values = []
    for k in display_keys:
        v = metrics.get(k, float("nan"))
        if not np.isnan(float(v)) and not np.isinf(float(v)):
            labels.append(k.replace("_", "\n"))
            values.append(float(v))

    colors = ["green" if v >= 0 else "red" for v in values]
    fig, ax = plt.subplots(figsize=figsize)
    ax.bar(labels, values, color=colors, alpha=0.7)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_title(title)
    ax.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
        plt.close(fig)
    else:
        plt.show()


def plot_training_progress(
    log_path: str,
    figsize: tuple = (12, 8),
    save_path: Optional[str] = None,
) -> None:
    """
    Plot training progress from a Stable-Baselines3 monitor CSV log file.

    Args:
        log_path:  Path to the monitor.csv file (or directory containing it).
        figsize:   Figure size.
        save_path: Save path.
    """
    _check_matplotlib()
    import os

    if os.path.isdir(log_path):
        candidates = [f for f in os.listdir(log_path) if f.endswith(".csv")]
        if not candidates:
            raise FileNotFoundError(f"No CSV files found in {log_path}")
        log_path = os.path.join(log_path, candidates[0])

    df = pd.read_csv(log_path, skiprows=1)
    if "r" not in df.columns:
        raise ValueError("Expected 'r' (reward) column in monitor CSV.")

    df["cumulative_mean_reward"] = df["r"].expanding().mean()

    fig, axes = plt.subplots(2, 1, figsize=figsize, sharex=True)

    axes[0].plot(df["r"].values, alpha=0.4, label="Episode reward", color="blue")
    axes[0].plot(
        df["cumulative_mean_reward"].values,
        linewidth=2,
        label="Cumulative mean",
        color="navy",
    )
    axes[0].set_ylabel("Reward")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    if "l" in df.columns:
        axes[1].plot(df["l"].values, alpha=0.6, color="orange", label="Episode length")
        axes[1].set_ylabel("Episode Length")
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

    axes[-1].set_xlabel("Episode")
    fig.suptitle("Training Progress", fontsize=12)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
        plt.close(fig)
    else:
        plt.show()
