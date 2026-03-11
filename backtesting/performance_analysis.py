"""
Performance analysis for backtesting results.
Provides aggregation utilities for walk-forward runs and detailed reporting.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from typing import List, Dict, Any

from backtesting.backtest_engine import BacktestResult


def aggregate_walk_forward(results: List[BacktestResult]) -> Dict[str, Any]:
    """
    Aggregate metrics across multiple walk-forward folds.

    Args:
        results: List of BacktestResult from BacktestEngine.walk_forward_backtest.

    Returns:
        Dict with mean, std, and per-fold values for each metric key.
    """
    if not results:
        return {}

    metric_keys = list(results[0].metrics.keys())
    aggregated: Dict[str, Any] = {}

    for key in metric_keys:
        vals = []
        for r in results:
            v = r.metrics.get(key, float("nan"))
            if v is not None and not (isinstance(v, float) and np.isnan(v)):
                vals.append(float(v))
        if vals:
            aggregated[f"{key}_mean"] = round(float(np.mean(vals)), 4)
            aggregated[f"{key}_std"] = round(float(np.std(vals)), 4)
            aggregated[f"{key}_folds"] = vals

    aggregated["total_trades_per_fold"] = [len(r.trades) for r in results]
    aggregated["total_trades_mean"] = float(np.mean(aggregated["total_trades_per_fold"]))
    return aggregated


def analyze_trades(trades: list) -> pd.DataFrame:
    """
    Return a DataFrame with per-trade statistics.

    Args:
        trades: List of Trade objects from PositionManager.

    Returns:
        DataFrame with one row per trade.
    """
    records = []
    for t in trades:
        records.append(
            {
                "direction": t.direction,
                "entry_bar": t.entry_bar,
                "exit_bar": t.exit_bar,
                "duration_bars": (t.exit_bar - t.entry_bar) if t.exit_bar is not None else None,
                "entry_price": t.entry_price,
                "exit_price": t.exit_price,
                "exit_reason": t.exit_reason,
                "pnl_pct": round(t.pnl_pct * 100, 4),
                "pnl_points": round(t.pnl_points, 6),
                "is_winner": t.pnl_pct > 0,
            }
        )
    return pd.DataFrame(records) if records else pd.DataFrame()


def print_walk_forward_summary(agg: Dict[str, Any]) -> None:
    """Pretty-print aggregated walk-forward metrics."""
    print("\n" + "=" * 60)
    print("  WALK-FORWARD SUMMARY")
    print("=" * 60)
    skip_suffixes = ("_folds",)
    for k, v in agg.items():
        if any(k.endswith(s) for s in skip_suffixes):
            continue
        if isinstance(v, float):
            print(f"  {k:45s}: {v:.4f}")
        else:
            print(f"  {k:45s}: {v}")
    print("=" * 60)


def monthly_returns(equity_curve: List[float], freq_bars_per_day: int = 24) -> pd.Series:
    """
    Compute approximate monthly returns from an equity curve.

    Assumes each bar is one step and groups by month using the bar count.

    Args:
        equity_curve:       Equity curve list.
        freq_bars_per_day:  Bars per day (24 for hourly data, 1 for daily, etc.).

    Returns:
        pd.Series of monthly returns indexed by approximate month number.
    """
    bars_per_month = freq_bars_per_day * 30
    eq = np.asarray(equity_curve)
    monthly_rets = []
    for start in range(0, len(eq) - bars_per_month, bars_per_month):
        end = min(start + bars_per_month, len(eq) - 1)
        r = (eq[end] - eq[start]) / eq[start] if eq[start] != 0 else 0.0
        monthly_rets.append(r)
    return pd.Series(monthly_rets, name="monthly_return")
