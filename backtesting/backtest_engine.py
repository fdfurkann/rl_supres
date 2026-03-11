"""
Backtesting engine — evaluates a trained RL agent over historical data.
"""

from __future__ import annotations

import os
import sys
import numpy as np
import pandas as pd
from typing import Optional, Dict, Any, List

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from environment.trading_env import TradingEnv
from utils.metrics import compute_metrics


class BacktestEngine:
    """
    Runs a trained RL agent over a historical OHLCV dataset and collects
    all trade records and equity curve data for analysis.

    Args:
        agent:      Trained SB3Agent.
        env_kwargs: Keyword arguments for TradingEnv (must match training config).
    """

    def __init__(self, agent, env_kwargs: Optional[Dict[str, Any]] = None):
        self.agent = agent
        self.env_kwargs = env_kwargs or {}

    def run(
        self,
        df: pd.DataFrame,
        deterministic: bool = True,
        start_step: Optional[int] = None,
    ) -> "BacktestResult":
        """
        Run the backtest over *df*.

        Args:
            df:           OHLCV DataFrame.
            deterministic: Use deterministic policy actions.
            start_step:   Override starting bar (0-based index into df).

        Returns:
            BacktestResult with equity curve, trades, and metrics.
        """
        env = TradingEnv(df=df, **self.env_kwargs)

        window_size = self.env_kwargs.get("window_size", 64)
        default_start = window_size
        options = {"start_step": start_step if start_step is not None else default_start}

        obs, info = env.reset(options=options)
        equity_curve = [env.pos_manager.equity]
        action_params_log = []
        done = False

        while not done:
            action, _ = self.agent.model.predict(obs, deterministic=deterministic)
            obs, reward, terminated, truncated, info = env.step(int(action))
            equity_curve.append(env.pos_manager.equity)
            if info.get("last_action_params"):
                action_params_log.append(info["last_action_params"])
            done = terminated or truncated

        trades = env.pos_manager.closed_trades.copy()
        metrics = compute_metrics(equity_curve, trades)

        return BacktestResult(
            equity_curve=equity_curve,
            trades=trades,
            metrics=metrics,
            action_params_log=action_params_log,
            df=df,
        )

    def walk_forward_backtest(
        self,
        df: pd.DataFrame,
        n_splits: int = 5,
        train_ratio: float = 0.7,
        deterministic: bool = True,
    ) -> List["BacktestResult"]:
        """
        Perform walk-forward backtesting.

        For each fold, the model is evaluated on the test portion.
        Note: this method does NOT re-train the model between folds; it
        evaluates the pre-trained model on each out-of-sample fold.
        For re-training per fold, use training.train.train() in a loop.

        Args:
            df:           Full OHLCV DataFrame.
            n_splits:     Number of walk-forward folds.
            train_ratio:  Fraction of each fold used as "training" (only the test
                          portion is evaluated).
            deterministic: Use deterministic actions.

        Returns:
            List of BacktestResult, one per fold.
        """
        from utils.data_loader import walk_forward_splits

        splits = walk_forward_splits(df, n_splits=n_splits, train_ratio=train_ratio)
        results = []
        for i, (_, test_df) in enumerate(splits):
            print(f"[backtest] Fold {i + 1}/{len(splits)} — {len(test_df):,} bars")
            result = self.run(test_df, deterministic=deterministic)
            results.append(result)
        return results


class BacktestResult:
    """Container for backtesting output."""

    def __init__(
        self,
        equity_curve: List[float],
        trades: list,
        metrics: Dict[str, Any],
        action_params_log: List[tuple],
        df: pd.DataFrame,
    ):
        self.equity_curve = equity_curve
        self.trades = trades
        self.metrics = metrics
        self.action_params_log = action_params_log
        self.df = df

    def print_summary(self) -> None:
        """Print a formatted metrics summary."""
        print("\n" + "=" * 50)
        print("  BACKTEST RESULTS")
        print("=" * 50)
        for k, v in self.metrics.items():
            print(f"  {k:35s}: {v}")
        print(f"  {'total_trades':35s}: {len(self.trades)}")
        print("=" * 50)

    def to_dict(self) -> Dict[str, Any]:
        """Return metrics and trade summary as a serializable dict."""
        trade_records = []
        for t in self.trades:
            trade_records.append(
                {
                    "direction": t.direction,
                    "entry_bar": t.entry_bar,
                    "exit_bar": t.exit_bar,
                    "entry_price": t.entry_price,
                    "exit_price": t.exit_price,
                    "exit_reason": t.exit_reason,
                    "pnl_pct": t.pnl_pct,
                    "pnl_points": t.pnl_points,
                }
            )
        return {
            "metrics": self.metrics,
            "trades": trade_records,
            "equity_curve": self.equity_curve,
        }

    def trades_dataframe(self) -> pd.DataFrame:
        """Return trades as a pandas DataFrame."""
        records = self.to_dict()["trades"]
        return pd.DataFrame(records) if records else pd.DataFrame()
