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


def plot_backtest_report(
    df: pd.DataFrame,
    trades: list,
    equity_curve: List[float],
    metrics: dict,
    title: str = "Backtest Report — BTC/USD",
    figsize: tuple = (14, 10),
    save_path: Optional[str] = None,
) -> None:
    """
    Three-panel backtest report:
        1. Price chart with entry (▲/▼) and exit (×) markers + SL/TP levels.
        2. Equity / profit curve.
        3. Rolling drawdown.

    Args:
        df:           OHLCV DataFrame used in the backtest.
        trades:       List of closed Trade objects from PositionManager.
        equity_curve: Equity values recorded during the backtest.
        metrics:      Dict from utils.metrics.compute_metrics.
        title:        Overall figure title.
        figsize:      Matplotlib figure size.
        save_path:    If provided, saves to this file path instead of showing.
    """
    _check_matplotlib()

    close = df["close"].values
    n = len(close)

    # Build index array for the backtest bars (equity curve may be shorter)
    eq = np.asarray(equity_curve, dtype=np.float64)
    eq_x = np.linspace(0, n - 1, len(eq))

    roll_max = np.maximum.accumulate(eq)
    dd = (roll_max - eq) / np.where(roll_max != 0, roll_max, 1.0) * 100

    fig, axes = plt.subplots(3, 1, figsize=figsize,
                             gridspec_kw={"height_ratios": [3, 2, 1]},
                             sharex=False)
    ax_price, ax_equity, ax_dd = axes

    # --- Panel 1: Price + Trades -------------------------------------------
    ax_price.plot(close, color="#2c3e50", linewidth=0.9, label="Close")

    for t in trades:
        color = "#27ae60" if t.direction == "long" else "#e74c3c"
        marker_entry = "^" if t.direction == "long" else "v"

        # Entry marker
        ax_price.scatter(
            t.entry_bar, t.entry_price,
            marker=marker_entry, color=color, s=70, zorder=6,
            label=f"{'Long' if t.direction == 'long' else 'Short'} entry"
        )
        # SL line
        ax_price.hlines(
            t.sl_price, t.entry_bar,
            t.exit_bar if t.exit_bar is not None else t.entry_bar + 1,
            colors="orange", linewidths=0.8, linestyles="--", alpha=0.7
        )
        # TP line
        ax_price.hlines(
            t.tp_price, t.entry_bar,
            t.exit_bar if t.exit_bar is not None else t.entry_bar + 1,
            colors="#3498db", linewidths=0.8, linestyles="--", alpha=0.7
        )
        # Exit marker
        if t.exit_bar is not None and t.exit_price is not None:
            exit_color = "#27ae60" if t.pnl_pct >= 0 else "#e74c3c"
            ax_price.scatter(
                t.exit_bar, t.exit_price,
                marker="x", color=exit_color, s=60, zorder=6, linewidths=1.5
            )

    # Deduplicate legend entries
    handles, labels = ax_price.get_legend_handles_labels()
    seen = {}
    for h, l in zip(handles, labels):
        if l not in seen:
            seen[l] = h
    ax_price.legend(seen.values(), seen.keys(), fontsize=7, loc="upper left")
    ax_price.set_ylabel("Price (USD)")
    ax_price.set_title(title, fontsize=11, fontweight="bold")
    ax_price.grid(True, alpha=0.2)

    # Annotate key metrics on the price chart
    metric_txt = (
        f"Total return: {metrics.get('total_return_pct', 'n/a'):.2f}%  |  "
        f"Sharpe: {metrics.get('sharpe_ratio', 'n/a'):.2f}  |  "
        f"Max DD: {metrics.get('max_drawdown_pct', 'n/a'):.2f}%  |  "
        f"Win rate: {metrics.get('win_rate_pct', 'n/a'):.1f}%  |  "
        f"Trades: {metrics.get('n_trades', 0)}"
    )
    ax_price.text(
        0.01, 0.02, metric_txt,
        transform=ax_price.transAxes, fontsize=7.5,
        verticalalignment="bottom",
        bbox=dict(facecolor="white", alpha=0.7, edgecolor="gray", boxstyle="round,pad=0.3"),
    )

    # --- Panel 2: Equity curve ---------------------------------------------
    start_eq = eq[0]
    pnl_pct = (eq - start_eq) / start_eq * 100
    ax_equity.plot(eq_x, pnl_pct, color="#2980b9", linewidth=1.2)
    ax_equity.axhline(0, color="black", linewidth=0.6, linestyle="--")
    ax_equity.fill_between(eq_x, pnl_pct, 0,
                           where=(pnl_pct >= 0), alpha=0.15, color="#27ae60")
    ax_equity.fill_between(eq_x, pnl_pct, 0,
                           where=(pnl_pct < 0), alpha=0.15, color="#e74c3c")
    ax_equity.set_ylabel("Cumulative P&L (%)")
    ax_equity.set_title("Equity Curve (P&L %)", fontsize=9)
    ax_equity.grid(True, alpha=0.2)

    # --- Panel 3: Drawdown -------------------------------------------------
    ax_dd.fill_between(eq_x, -dd, 0, color="#e74c3c", alpha=0.45)
    ax_dd.plot(eq_x, -dd, color="#e74c3c", linewidth=0.7)
    ax_dd.set_ylabel("Drawdown (%)")
    ax_dd.set_xlabel("Bar index")
    ax_dd.set_title("Drawdown", fontsize=9)
    ax_dd.grid(True, alpha=0.2)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"Chart saved → {save_path}")
    else:
        plt.show()
