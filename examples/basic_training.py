"""
Basic training example for the RL support/resistance breakout strategy.

This script:
1. Generates synthetic OHLCV data.
2. Splits into train / test.
3. Trains a PPO agent for a short period.
4. Evaluates on the test set and prints performance metrics.

Run from project root:
    python examples/basic_training.py
"""

from __future__ import annotations

import os
import sys

# Ensure project root is importable
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.data_loader import generate_synthetic_ohlcv, train_test_split
from utils.metrics import compute_metrics
from agents.sb3_agent import SB3Agent
from training.hyperparams import fast_test_config


def main():
    print("=" * 60)
    print("  RL SUPPORT/RESISTANCE BREAKOUT — BASIC TRAINING EXAMPLE")
    print("=" * 60)

    # ---- 1. Generate data -------------------------------------------------
    print("\n[1] Generating synthetic OHLCV data …")
    df = generate_synthetic_ohlcv(n_bars=3_000, seed=42)
    train_df, test_df = train_test_split(df, train_ratio=0.8)
    print(f"    Train: {len(train_df):,} bars | Test: {len(test_df):,} bars")

    # ---- 2. Build & train agent -------------------------------------------
    print("\n[2] Building PPO agent …")
    cfg = fast_test_config()
    cfg.total_timesteps = 20_000   # small for quick demo
    cfg.n_envs = 1
    cfg.env.window_size = 32
    cfg.env.max_episode_steps = 200

    agent = SB3Agent(
        algorithm="ppo",
        policy="MlpPolicy",
        env_kwargs=cfg.env.to_dict(),
        algorithm_kwargs=cfg.ppo.to_dict(),
        normalize_obs=False,  # skip VecNormalize for simplicity
        verbose=1,
    )

    print(f"\n[3] Training for {cfg.total_timesteps:,} timesteps …")
    agent.train(
        df=train_df,
        total_timesteps=cfg.total_timesteps,
        n_envs=cfg.n_envs,
    )

    # ---- 3. Evaluate -------------------------------------------------------
    print("\n[4] Evaluating on test data …")
    result = agent.evaluate(test_df, deterministic=True)

    metrics = compute_metrics(result["equity_curve"], result["trades"])
    print("\nPerformance metrics:")
    for k, v in metrics.items():
        print(f"  {k:35s}: {v}")

    print(f"\n  Equity start : {result['equity_curve'][0]:,.2f}")
    print(f"  Equity end   : {result['equity_curve'][-1]:,.2f}")
    print(f"  Total trades : {result['n_trades']:.0f}")

    # ---- 4. Optionally save -----------------------------------------------
    save_dir = "/tmp/rl_supres_example"
    agent.save(save_dir)
    print(f"\nModel saved to: {save_dir}")
    print("Done!")


if __name__ == "__main__":
    main()
