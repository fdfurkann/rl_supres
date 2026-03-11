"""
Binance BTCUSDT Live Test
=========================
Fetches real Binance BTCUSDT OHLCV data (or falls back to 2021 daily data
if the Binance API is unreachable), trains a PPO agent on the first 80% of
the data, then backtests on the remaining 20%.

Outputs:
  - Formatted table of all trade entry/exit points (printed to console)
  - Performance metrics summary
  - backtest_report.png   — price chart with trade markers + equity curve + drawdown

Usage:
    python examples/binance_test.py
    python examples/binance_test.py --interval 4h --limit 1000 --output /tmp/report.png
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd

from utils.data_loader import fetch_binance_ohlcv, train_test_split
from utils.metrics import compute_metrics
from utils.visualization import plot_backtest_report
from agents.sb3_agent import SB3Agent
from backtesting.backtest_engine import BacktestEngine


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _print_trades_table(trades: list, df: pd.DataFrame) -> None:
    """Print a formatted table of all entry/exit trade points."""
    if not trades:
        print("  (no trades executed)")
        return

    # Try to map bar index → datetime if df has a DatetimeIndex
    has_dt = isinstance(df.index, pd.DatetimeIndex)

    header = (
        f"{'#':>4}  {'Dir':>5}  "
        f"{'Entry Bar':>9}  {'Entry Date':>12}  {'Entry Price':>12}  "
        f"{'Exit Bar':>8}  {'Exit Date':>12}  {'Exit Price':>11}  "
        f"{'Exit':>6}  {'PnL %':>7}  {'Dur(bars)':>9}"
    )
    sep = "-" * len(header)

    print(sep)
    print(header)
    print(sep)

    for i, t in enumerate(trades, 1):
        entry_dt = str(df.index[t.entry_bar])[:10] if has_dt and t.entry_bar < len(df) else "—"
        exit_dt = (
            str(df.index[t.exit_bar])[:10]
            if has_dt and t.exit_bar is not None and t.exit_bar < len(df)
            else "—"
        )
        exit_bar_s = str(t.exit_bar) if t.exit_bar is not None else "—"
        exit_price_s = f"{t.exit_price:>11.2f}" if t.exit_price is not None else f"{'—':>11}"
        exit_reason_s = (t.exit_reason or "—").upper()[:6]
        pnl_s = f"{t.pnl_pct * 100:>+7.3f}"
        dur = (t.exit_bar - t.entry_bar) if t.exit_bar is not None else "—"

        print(
            f"{i:>4}  {t.direction.upper():>5}  "
            f"{t.entry_bar:>9}  {entry_dt:>12}  {t.entry_price:>12.2f}  "
            f"{exit_bar_s:>8}  {exit_dt:>12}  {exit_price_s}  "
            f"{exit_reason_s:>6}  {pnl_s}  {str(dur):>9}"
        )

    print(sep)

    # Summary row
    wins = sum(1 for t in trades if t.pnl_pct > 0)
    total_pnl = sum(t.pnl_pct for t in trades) * 100
    print(
        f"{'TOTAL':>4}  {len(trades):>5} trades | "
        f"{wins} wins / {len(trades) - wins} losses | "
        f"Net PnL sum: {total_pnl:+.3f}%"
    )
    print(sep)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run(
    symbol: str = "BTCUSDT",
    interval: str = "1d",
    limit: int = 500,
    train_ratio: float = 0.80,
    timesteps: int = 30_000,
    window_size: int = 20,
    output: str = "backtest_report.png",
    seed: int = 42,
    commission_pct: float = 0.001,
) -> None:
    """
    Main pipeline: fetch → train → backtest → report.

    Args:
        symbol:         Binance trading pair.
        interval:       Candle interval.  Use '1d' for daily, '1h' for hourly, etc.
        limit:          Number of candles to fetch (max 1000 per Binance request).
        train_ratio:    Fraction of data used for training.
        timesteps:      PPO training timesteps.
        window_size:    Observation window (number of bars).
        output:         Path to save the backtest chart PNG.
        seed:           Random seed.
        commission_pct: One-way commission fraction (verify current rate at
                        binance.com/en/fee/schedule; default 0.001 = 0.1%).
    """
    np.random.seed(seed)

    # ---- 1. Fetch data ----------------------------------------------------
    print("=" * 65)
    print(f"  Fetching {symbol} {interval} data from Binance …")
    print("=" * 65)
    df = fetch_binance_ohlcv(
        symbol=symbol,
        interval=interval,
        limit=limit,
        use_fallback_on_error=True,
    )
    print(f"\n  Total bars  : {len(df):,}")
    print(f"  Date range  : {df.index[0]} → {df.index[-1]}")
    print(f"  Price range : {df['close'].min():.2f} – {df['close'].max():.2f} USD")
    print(f"  Columns     : {list(df.columns)}")

    if len(df) < window_size * 3:
        raise ValueError(
            f"Not enough data ({len(df)} bars) for window_size={window_size}. "
            "Fetch more bars or reduce window_size."
        )

    # ---- 2. Train / test split --------------------------------------------
    train_df, test_df = train_test_split(df, train_ratio=train_ratio)
    print(f"\n  Train bars  : {len(train_df):,}")
    print(f"  Test bars   : {len(test_df):,}")

    # ---- 3. Build & train PPO agent --------------------------------------
    env_kwargs = {
        "window_size": window_size,
        "commission_pct": commission_pct,  # verify current Binance fees at binance.com/en/fee/schedule
        "slippage_pct": 0.0005,
        "initial_equity": 10_000.0,
        "max_episode_steps": min(len(train_df) - window_size - 1, 400),
        "reward_scaling": 1.0,
        "risk_penalty_coef": 0.05,
        "allow_short": True,
    }

    print(f"\n  Training PPO for {timesteps:,} timesteps …")
    agent = SB3Agent(
        algorithm="ppo",
        policy="MlpPolicy",
        env_kwargs=env_kwargs,
        algorithm_kwargs={
            "n_steps": 256,
            "batch_size": 64,
            "learning_rate": 3e-4,
            "ent_coef": 0.01,
            "gamma": 0.99,
        },
        normalize_obs=False,
        verbose=1,
    )
    agent.train(df=train_df, total_timesteps=timesteps, n_envs=1)

    # ---- 4. Backtest on test set -----------------------------------------
    print(f"\n  Running backtest on {len(test_df):,} test bars …")
    engine = BacktestEngine(agent=agent, env_kwargs=env_kwargs)
    result = engine.run(test_df, deterministic=True)

    # ---- 5. Print trade table --------------------------------------------
    print(f"\n{'=' * 65}")
    print(f"  TRADE LOG — {symbol} {interval.upper()} (test set)")
    print(f"{'=' * 65}")
    _print_trades_table(result.trades, test_df)

    # ---- 6. Print metrics ------------------------------------------------
    print(f"\n{'=' * 65}")
    print("  PERFORMANCE METRICS")
    print(f"{'=' * 65}")
    for k, v in result.metrics.items():
        bar = ""
        if k == "total_return_pct":
            sign = "+" if v >= 0 else ""
            bar = f"  ← {sign}{v:.2f}%"
        print(f"  {k:<35}: {v}{bar}")

    print(f"\n  Initial equity : $10,000.00")
    final_eq = result.equity_curve[-1] if result.equity_curve else 10_000.0
    pnl = final_eq - 10_000.0
    print(f"  Final equity   : ${final_eq:,.2f}  ({pnl:+,.2f})")

    # ---- 7. Save chart ---------------------------------------------------
    print(f"\n  Generating backtest chart → {output}")
    plot_backtest_report(
        df=test_df,
        trades=result.trades,
        equity_curve=result.equity_curve,
        metrics=result.metrics,
        title=f"RL Breakout Strategy — {symbol} {interval.upper()} (test set)",
        save_path=output,
    )
    print("  Done!")
    print("=" * 65)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args():
    p = argparse.ArgumentParser(
        description="Fetch Binance BTC data, train RL agent, run backtest."
    )
    p.add_argument("--symbol",   default="BTCUSDT", help="Binance trading pair.")
    p.add_argument("--interval", default="1d",      help="Candle interval (1d, 4h, 1h, …).")
    p.add_argument("--limit",    default=500, type=int,
                   help="Number of candles to fetch (max 1000).")
    p.add_argument("--train-ratio", default=0.80, type=float,
                   help="Fraction of data used for training.")
    p.add_argument("--timesteps", default=30_000, type=int,
                   help="PPO training timesteps.")
    p.add_argument("--window",   default=20, type=int,
                   help="Observation window size (bars).")
    p.add_argument("--output",   default="backtest_report.png",
                   help="Path to save the chart PNG.")
    p.add_argument("--seed",     default=42, type=int)
    p.add_argument("--commission", default=0.001, type=float,
                   help="One-way commission fraction (default 0.001 = 0.1%%). "
                        "Check current Binance fee schedule.")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    run(
        symbol=args.symbol,
        interval=args.interval,
        limit=args.limit,
        train_ratio=args.train_ratio,
        timesteps=args.timesteps,
        window_size=args.window,
        output=args.output,
        seed=args.seed,
        commission_pct=args.commission,
    )
