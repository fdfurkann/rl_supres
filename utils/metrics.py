"""
Trading performance metrics.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from typing import List, Optional


def sharpe_ratio(
    returns: np.ndarray,
    risk_free_rate: float = 0.0,
    periods_per_year: int = 252,
) -> float:
    """
    Annualised Sharpe ratio.

    Args:
        returns:          Array of period returns (arithmetic, not log).
        risk_free_rate:   Annual risk-free rate (default 0).
        periods_per_year: Trading periods per year (252 for daily, 8760 for hourly, …).

    Returns:
        Sharpe ratio (NaN if std is zero).
    """
    if len(returns) < 2:
        return float("nan")
    excess = returns - risk_free_rate / periods_per_year
    std = np.std(excess, ddof=1)
    if std == 0:
        return float("nan")
    return float(np.mean(excess) / std * np.sqrt(periods_per_year))


def max_drawdown(equity_curve: np.ndarray) -> float:
    """
    Maximum drawdown as a positive fraction (e.g. 0.25 = −25 % peak-to-trough).

    Args:
        equity_curve: Array of equity values over time.

    Returns:
        Maximum drawdown in [0, 1].
    """
    equity_curve = np.asarray(equity_curve, dtype=np.float64)
    if len(equity_curve) < 2:
        return 0.0
    roll_max = np.maximum.accumulate(equity_curve)
    drawdowns = (roll_max - equity_curve) / np.where(roll_max != 0, roll_max, 1.0)
    return float(np.max(drawdowns))


def profit_factor(trade_pnls: List[float]) -> float:
    """
    Gross profit / gross loss.

    Returns:
        Profit factor (inf if no losses, NaN if empty).
    """
    if not trade_pnls:
        return float("nan")
    gross_profit = sum(p for p in trade_pnls if p > 0)
    gross_loss = abs(sum(p for p in trade_pnls if p < 0))
    if gross_loss == 0:
        return float("inf") if gross_profit > 0 else float("nan")
    return float(gross_profit / gross_loss)


def win_rate(trade_pnls: List[float]) -> float:
    """
    Fraction of trades that are profitable.

    Returns:
        Win rate in [0, 1] (NaN if no trades).
    """
    if not trade_pnls:
        return float("nan")
    wins = sum(1 for p in trade_pnls if p > 0)
    return float(wins / len(trade_pnls))


def average_trade(trade_pnls: List[float]) -> float:
    """Average P&L per trade (NaN if empty)."""
    if not trade_pnls:
        return float("nan")
    return float(np.mean(trade_pnls))


def total_return(equity_curve: np.ndarray) -> float:
    """
    Total return over the equity curve.

    Returns:
        Fractional total return (e.g. 0.30 = +30 %).
    """
    equity_curve = np.asarray(equity_curve)
    if len(equity_curve) < 2 or equity_curve[0] == 0:
        return float("nan")
    return float((equity_curve[-1] - equity_curve[0]) / equity_curve[0])


def annualised_return(
    equity_curve: np.ndarray,
    periods: int,
    periods_per_year: int = 252,
) -> float:
    """
    Annualised return from the equity curve.

    Args:
        equity_curve:     Equity curve array.
        periods:          Number of periods covered.
        periods_per_year: Periods per calendar year.

    Returns:
        Annualised return as a fraction.
    """
    tr = total_return(equity_curve)
    if np.isnan(tr) or periods <= 0:
        return float("nan")
    years = periods / periods_per_year
    return float((1 + tr) ** (1 / years) - 1)


def calmar_ratio(
    equity_curve: np.ndarray,
    periods: int,
    periods_per_year: int = 252,
) -> float:
    """
    Calmar ratio = annualised return / max drawdown.

    Returns:
        Calmar ratio (NaN if max drawdown is zero).
    """
    ann_ret = annualised_return(equity_curve, periods, periods_per_year)
    mdd = max_drawdown(equity_curve)
    if mdd == 0:
        return float("nan")
    return float(ann_ret / mdd)


def compute_metrics(
    equity_curve: List[float],
    trades: list,
    periods_per_year: int = 252,
) -> dict:
    """
    Compute a comprehensive set of performance metrics.

    Args:
        equity_curve:     List of equity values (one per step).
        trades:           List of closed Trade objects from PositionManager.
        periods_per_year: Periods per year for annualisation.

    Returns:
        Dict with metric names and values.
    """
    eq = np.asarray(equity_curve, dtype=np.float64)
    returns = np.diff(eq) / np.where(eq[:-1] != 0, eq[:-1], 1.0)

    trade_pnls = [t.pnl_pct for t in trades] if trades else []
    durations = [
        (t.exit_bar - t.entry_bar)
        for t in trades
        if t.exit_bar is not None
    ]

    return {
        "total_return_pct": round(total_return(eq) * 100, 4),
        "annualised_return_pct": round(
            annualised_return(eq, len(eq), periods_per_year) * 100, 4
        ),
        "sharpe_ratio": round(sharpe_ratio(returns, periods_per_year=periods_per_year), 4),
        "max_drawdown_pct": round(max_drawdown(eq) * 100, 4),
        "calmar_ratio": round(calmar_ratio(eq, len(eq), periods_per_year), 4),
        "profit_factor": round(profit_factor(trade_pnls), 4),
        "win_rate_pct": round(win_rate(trade_pnls) * 100, 4),
        "n_trades": len(trade_pnls),
        "avg_trade_pct": round(average_trade(trade_pnls) * 100 if trade_pnls else float("nan"), 4),
        "avg_trade_duration_bars": round(np.mean(durations), 2) if durations else float("nan"),
    }
