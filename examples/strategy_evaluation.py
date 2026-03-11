"""
Strategy evaluation example — runs a trained agent through the backtesting engine
and produces a detailed performance report with walk-forward validation.

Run from project root:
    python examples/strategy_evaluation.py
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.data_loader import generate_synthetic_ohlcv, train_test_split, walk_forward_splits
from utils.metrics import compute_metrics
from agents.sb3_agent import SB3Agent
from backtesting.backtest_engine import BacktestEngine
from backtesting.performance_analysis import (
    aggregate_walk_forward,
    analyze_trades,
    print_walk_forward_summary,
)
from training.hyperparams import fast_test_config


def main():
    print("=" * 60)
    print("  RL BREAKOUT STRATEGY — EVALUATION EXAMPLE")
    print("=" * 60)

    # ---- 1. Data ----------------------------------------------------------
    print("\n[1] Generating data …")
    df = generate_synthetic_ohlcv(n_bars=5_000, seed=7)
    train_df, test_df = train_test_split(df, train_ratio=0.75)
    print(f"    Train: {len(train_df):,} bars | Test: {len(test_df):,} bars")

    # ---- 2. Train (quick) -------------------------------------------------
    print("\n[2] Training PPO agent …")
    cfg = fast_test_config()
    cfg.total_timesteps = 20_000
    cfg.n_envs = 1
    cfg.env.window_size = 32
    cfg.env.max_episode_steps = 300

    agent = SB3Agent(
        algorithm="ppo",
        policy="MlpPolicy",
        env_kwargs=cfg.env.to_dict(),
        algorithm_kwargs=cfg.ppo.to_dict(),
        normalize_obs=False,
        verbose=0,
    )
    agent.train(df=train_df, total_timesteps=cfg.total_timesteps, n_envs=1)

    # ---- 3. Single backtest -----------------------------------------------
    print("\n[3] Running single backtest on held-out test set …")
    engine = BacktestEngine(agent=agent, env_kwargs=cfg.env.to_dict())
    result = engine.run(test_df, deterministic=True)
    result.print_summary()

    # ---- 4. Trade analysis ------------------------------------------------
    print("\n[4] Trade-level analysis:")
    trades_df = result.trades_dataframe()
    if not trades_df.empty:
        print(trades_df.describe(include="all").to_string())
    else:
        print("    No trades recorded.")

    # ---- 5. Walk-forward --------------------------------------------------
    print("\n[5] Walk-forward evaluation (5 folds) …")
    wf_results = engine.walk_forward_backtest(df, n_splits=5, train_ratio=0.7)
    agg = aggregate_walk_forward(wf_results)
    print_walk_forward_summary(agg)

    print("\nDone!")


if __name__ == "__main__":
    main()
