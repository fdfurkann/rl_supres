"""
Stable-Baselines3 wrapper for the RL trading agent.
Provides a unified interface for training, evaluation, and saving/loading.
"""

from __future__ import annotations

import os
import numpy as np
import pandas as pd
from typing import Optional, Dict, Any, Type

from stable_baselines3 import PPO, A2C, SAC
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.callbacks import (
    EvalCallback,
    CheckpointCallback,
    BaseCallback,
)
from stable_baselines3.common.monitor import Monitor

from environment.trading_env import TradingEnv

ALGORITHM_MAP = {
    "ppo": PPO,
    "a2c": A2C,
    "sac": SAC,
}


class EarlyStopCallback(BaseCallback):
    """Stop training when mean reward has not improved for `patience` evaluations."""

    def __init__(self, patience: int = 10, verbose: int = 0):
        super().__init__(verbose)
        self.patience = patience
        self._no_improve_count = 0
        self._best_mean_reward = -np.inf

    def _on_step(self) -> bool:
        if self.locals.get("mean_reward") is not None:
            mean_reward = self.locals["mean_reward"]
            if mean_reward > self._best_mean_reward:
                self._best_mean_reward = mean_reward
                self._no_improve_count = 0
            else:
                self._no_improve_count += 1
            if self._no_improve_count >= self.patience:
                if self.verbose:
                    print(f"Early stopping: no improvement for {self.patience} evals.")
                return False
        return True


class SB3Agent:
    """
    Wrapper around Stable-Baselines3 for the TradingEnv.

    Args:
        algorithm:     RL algorithm name: "ppo", "a2c", or "sac".
        policy:        SB3 policy key ("MlpPolicy", "CnnPolicy", etc.).
        env_kwargs:    Keyword arguments passed to TradingEnv.__init__.
        algorithm_kwargs: Extra kwargs forwarded to the algorithm constructor.
        normalize_obs: Whether to wrap env with VecNormalize.
        verbose:       Verbosity level for SB3 (0=silent, 1=info, 2=debug).
    """

    def __init__(
        self,
        algorithm: str = "ppo",
        policy: str = "MlpPolicy",
        env_kwargs: Optional[Dict[str, Any]] = None,
        algorithm_kwargs: Optional[Dict[str, Any]] = None,
        normalize_obs: bool = True,
        verbose: int = 1,
    ):
        algo_name = algorithm.lower()
        if algo_name not in ALGORITHM_MAP:
            raise ValueError(
                f"Unknown algorithm '{algorithm}'. Choose from {list(ALGORITHM_MAP.keys())}."
            )
        self.algorithm_cls = ALGORITHM_MAP[algo_name]
        self.policy = policy
        self.env_kwargs = env_kwargs or {}
        self.algorithm_kwargs = algorithm_kwargs or {}
        self.normalize_obs = normalize_obs
        self.verbose = verbose

        self.model = None
        self.vec_env = None

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def build_env(self, df: pd.DataFrame, n_envs: int = 1) -> DummyVecEnv:
        """Create (and optionally normalize) a vectorised training environment."""

        def _make():
            env = TradingEnv(df=df, **self.env_kwargs)
            return Monitor(env)

        vec_env = make_vec_env(_make, n_envs=n_envs)
        if self.normalize_obs:
            vec_env = VecNormalize(vec_env, norm_obs=True, norm_reward=True, clip_obs=10.0)
        self.vec_env = vec_env
        return vec_env

    def train(
        self,
        df: pd.DataFrame,
        total_timesteps: int = 100_000,
        n_envs: int = 1,
        eval_df: Optional[pd.DataFrame] = None,
        eval_freq: int = 10_000,
        save_path: Optional[str] = None,
        checkpoint_freq: int = 50_000,
        callbacks: Optional[list] = None,
    ) -> "SB3Agent":
        """
        Train the agent.

        Args:
            df:              Training OHLCV DataFrame.
            total_timesteps: Total environment steps to train for.
            n_envs:          Number of parallel environments.
            eval_df:         Optional validation DataFrame for EvalCallback.
            eval_freq:       Evaluation frequency in steps.
            save_path:       Directory to save checkpoints and best model.
            checkpoint_freq: Frequency for saving model checkpoints.
            callbacks:       Additional SB3 callbacks.

        Returns:
            self (for chaining).
        """
        train_env = self.build_env(df, n_envs=n_envs)

        # Merge default algorithm kwargs
        algo_kwargs = {"verbose": self.verbose, **self.algorithm_kwargs}
        if self.algorithm_cls == SAC:
            # SAC requires continuous action space; not directly compatible with
            # Discrete actions unless wrapped — warn the user.
            raise ValueError(
                "SAC requires a continuous action space. "
                "Use 'ppo' or 'a2c' for the discrete action space in TradingEnv."
            )

        self.model = self.algorithm_cls(
            policy=self.policy, env=train_env, **algo_kwargs
        )

        cb_list = callbacks or []

        if eval_df is not None and save_path is not None:
            eval_env = self.build_eval_env(eval_df)
            eval_cb = EvalCallback(
                eval_env,
                best_model_save_path=save_path,
                log_path=save_path,
                eval_freq=max(eval_freq // n_envs, 1),
                deterministic=True,
                render=False,
            )
            cb_list.append(eval_cb)

        if save_path is not None:
            ckpt_cb = CheckpointCallback(
                save_freq=max(checkpoint_freq // n_envs, 1),
                save_path=save_path,
                name_prefix="trading_model",
            )
            cb_list.append(ckpt_cb)

        self.model.learn(total_timesteps=total_timesteps, callback=cb_list or None)
        return self

    def build_eval_env(self, df: pd.DataFrame) -> DummyVecEnv:
        """Create a deterministic (non-randomised start) evaluation environment."""

        def _make():
            env = TradingEnv(df=df, **self.env_kwargs)
            return Monitor(env)

        vec_env = make_vec_env(_make, n_envs=1)
        if self.normalize_obs and self.vec_env is not None:
            vec_env = VecNormalize(
                vec_env,
                norm_obs=True,
                norm_reward=False,
                clip_obs=10.0,
                training=False,
            )
        return vec_env

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

    def evaluate(
        self,
        df: pd.DataFrame,
        n_eval_episodes: int = 1,
        deterministic: bool = True,
        start_step: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Run the trained agent on *df* and return performance metrics.

        Args:
            df:              Evaluation OHLCV DataFrame.
            n_eval_episodes: Number of evaluation episodes to average.
            deterministic:   Use deterministic policy (recommended for eval).
            start_step:      Fix the start step (None = start of data).

        Returns:
            Dict with keys: total_reward, n_trades, equity_curve (list).
        """
        if self.model is None:
            raise RuntimeError("Model not trained. Call .train() first.")

        env = TradingEnv(df=df, **self.env_kwargs)
        results = []

        for _ in range(n_eval_episodes):
            options = {"start_step": start_step} if start_step is not None else None
            obs, _ = env.reset(options=options)
            total_reward = 0.0
            equity_curve = [env.pos_manager.equity]
            done = False

            while not done:
                action, _ = self.model.predict(obs, deterministic=deterministic)
                obs, reward, terminated, truncated, _ = env.step(int(action))
                total_reward += reward
                equity_curve.append(env.pos_manager.equity)
                done = terminated or truncated

            results.append(
                {
                    "total_reward": total_reward,
                    "n_trades": len(env.pos_manager.closed_trades),
                    "equity_curve": equity_curve,
                    "trades": env.pos_manager.closed_trades.copy(),
                }
            )

        # Average scalar metrics
        avg_reward = np.mean([r["total_reward"] for r in results])
        avg_trades = np.mean([r["n_trades"] for r in results])
        return {
            "total_reward": float(avg_reward),
            "n_trades": float(avg_trades),
            "equity_curve": results[0]["equity_curve"],
            "trades": results[0]["trades"],
        }

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str) -> None:
        """Save model (and normalizer stats if applicable) to *path*."""
        if self.model is None:
            raise RuntimeError("Nothing to save — model not trained.")
        os.makedirs(path, exist_ok=True)
        self.model.save(os.path.join(path, "model"))
        if self.normalize_obs and isinstance(self.vec_env, VecNormalize):
            self.vec_env.save(os.path.join(path, "vec_normalize.pkl"))

    def load(self, path: str, df: Optional[pd.DataFrame] = None) -> "SB3Agent":
        """
        Load a previously saved model.

        Args:
            path: Directory where model was saved.
            df:   Optional DataFrame to rebuild the environment (for VecNormalize).

        Returns:
            self.
        """
        model_path = os.path.join(path, "model")
        normalize_path = os.path.join(path, "vec_normalize.pkl")

        if df is not None and self.normalize_obs and os.path.exists(normalize_path):
            vec_env = self.build_env(df)
            vec_env = VecNormalize.load(normalize_path, vec_env)
            vec_env.training = False
            vec_env.norm_reward = False
            self.vec_env = vec_env
            self.model = self.algorithm_cls.load(model_path, env=vec_env)
        else:
            self.model = self.algorithm_cls.load(model_path)

        return self
