"""
Main training script for the RL support/resistance breakout strategy.

Usage:
    python training/train.py --data path/to/data.csv --timesteps 200000 --save-dir ./runs
    python training/train.py --synthetic --timesteps 50000
"""

from __future__ import annotations

import argparse
import os
import sys
import json
from datetime import datetime
from typing import Optional

# Allow running from project root or from training/ directory
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agents.sb3_agent import SB3Agent
from training.hyperparams import TrainingConfig, default_ppo_config, default_a2c_config
from utils.data_loader import (
    load_csv,
    load_parquet,
    train_test_split,
    generate_synthetic_ohlcv,
)
from utils.metrics import compute_metrics


def train(
    config: Optional[TrainingConfig] = None,
    data_path: Optional[str] = None,
    save_dir: str = "./runs",
    synthetic: bool = False,
    n_synthetic_bars: int = 5_000,
    eval_split: float = 0.2,
    verbose: bool = True,
) -> SB3Agent:
    """
    Run a full training pipeline.

    Args:
        config:           TrainingConfig (defaults to PPO if None).
        data_path:        Path to OHLCV CSV or Parquet file.
                          Required unless synthetic=True.
        save_dir:         Directory to save model checkpoints and results.
        synthetic:        Use synthetically generated data for testing.
        n_synthetic_bars: Number of bars if synthetic=True.
        eval_split:       Fraction of data to reserve for evaluation.
        verbose:          Print progress messages.

    Returns:
        Trained SB3Agent.
    """
    if config is None:
        config = default_ppo_config()

    # ---- Load data --------------------------------------------------------
    if synthetic:
        if verbose:
            print("[train] Generating synthetic OHLCV data …")
        df = generate_synthetic_ohlcv(n_bars=n_synthetic_bars, seed=config.seed)
    elif data_path is not None:
        if verbose:
            print(f"[train] Loading data from {data_path} …")
        if data_path.endswith(".parquet"):
            df = load_parquet(data_path)
        else:
            df = load_csv(data_path)
    else:
        raise ValueError("Provide either data_path or set synthetic=True.")

    train_df, eval_df = train_test_split(df, train_ratio=1.0 - eval_split, gap_bars=0)
    if verbose:
        print(
            f"[train] Train bars: {len(train_df):,} | Eval bars: {len(eval_df):,}"
        )

    # ---- Build agent ------------------------------------------------------
    agent = SB3Agent(
        algorithm=config.algorithm,
        policy=config.policy,
        env_kwargs=config.env.to_dict(),
        algorithm_kwargs=config.get_algorithm_kwargs(),
        normalize_obs=config.normalize_obs,
        verbose=config.verbose if config.verbose else int(verbose),
    )

    # ---- Create run directory ---------------------------------------------
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(save_dir, f"{config.algorithm}_{timestamp}")
    os.makedirs(run_dir, exist_ok=True)

    # Save config for reproducibility
    config_dict = {
        "algorithm": config.algorithm,
        "policy": config.policy,
        "total_timesteps": config.total_timesteps,
        "n_envs": config.n_envs,
        "env": config.env.to_dict(),
        "algo_kwargs": config.get_algorithm_kwargs(),
    }
    with open(os.path.join(run_dir, "config.json"), "w") as f:
        json.dump(config_dict, f, indent=2)

    # ---- Train ------------------------------------------------------------
    if verbose:
        print(
            f"[train] Starting {config.algorithm.upper()} training "
            f"for {config.total_timesteps:,} steps …"
        )
    agent.train(
        df=train_df,
        total_timesteps=config.total_timesteps,
        n_envs=config.n_envs,
        eval_df=eval_df if len(eval_df) > config.env.window_size * 2 else None,
        eval_freq=config.eval_freq,
        save_path=run_dir,
        checkpoint_freq=config.checkpoint_freq,
    )

    # ---- Save final model -------------------------------------------------
    agent.save(run_dir)
    if verbose:
        print(f"[train] Model saved to {run_dir}")

    # ---- Evaluate on held-out set ----------------------------------------
    if len(eval_df) > config.env.window_size * 2:
        if verbose:
            print("[train] Evaluating on held-out data …")
        result = agent.evaluate(eval_df)
        metrics = compute_metrics(
            result["equity_curve"],
            result["trades"],
        )
        if verbose:
            print("[train] Evaluation metrics:")
            for k, v in metrics.items():
                print(f"  {k:30s}: {v}")
        with open(os.path.join(run_dir, "eval_metrics.json"), "w") as f:
            json.dump({k: (v if not isinstance(v, float) or not (v != v) else "nan")
                       for k, v in metrics.items()}, f, indent=2)

    return agent


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def _parse_args():
    p = argparse.ArgumentParser(
        description="Train RL support/resistance breakout trading agent."
    )
    p.add_argument("--data", type=str, default=None, help="Path to CSV or Parquet OHLCV file.")
    p.add_argument("--synthetic", action="store_true", help="Use synthetic data.")
    p.add_argument("--n-bars", type=int, default=5_000, help="Synthetic bar count.")
    p.add_argument("--algo", type=str, default="ppo", choices=["ppo", "a2c"], help="RL algorithm.")
    p.add_argument("--timesteps", type=int, default=200_000, help="Training timesteps.")
    p.add_argument("--n-envs", type=int, default=4, help="Parallel environments.")
    p.add_argument("--window", type=int, default=64, help="Observation window size.")
    p.add_argument("--save-dir", type=str, default="./runs", help="Output directory.")
    p.add_argument("--seed", type=int, default=42, help="Random seed.")
    p.add_argument("--no-short", action="store_true", help="Disable short trades.")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()

    if args.algo == "ppo":
        cfg = default_ppo_config()
    else:
        cfg = default_a2c_config()

    cfg.total_timesteps = args.timesteps
    cfg.n_envs = args.n_envs
    cfg.seed = args.seed
    cfg.env.window_size = args.window
    cfg.env.allow_short = not args.no_short

    train(
        config=cfg,
        data_path=args.data,
        save_dir=args.save_dir,
        synthetic=args.synthetic,
        n_synthetic_bars=args.n_bars,
    )
