"""
Hyperparameter configurations for RL training.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Any, Optional


@dataclass
class EnvConfig:
    """TradingEnv hyperparameters."""
    window_size: int = 64
    commission_pct: float = 0.001
    slippage_pct: float = 0.0005
    initial_equity: float = 10_000.0
    max_episode_steps: Optional[int] = 500
    reward_scaling: float = 1.0
    risk_penalty_coef: float = 0.1
    allow_short: bool = True
    volume_lookback: int = 20

    def to_dict(self) -> Dict[str, Any]:
        return {
            "window_size": self.window_size,
            "commission_pct": self.commission_pct,
            "slippage_pct": self.slippage_pct,
            "initial_equity": self.initial_equity,
            "max_episode_steps": self.max_episode_steps,
            "reward_scaling": self.reward_scaling,
            "risk_penalty_coef": self.risk_penalty_coef,
            "allow_short": self.allow_short,
            "volume_lookback": self.volume_lookback,
        }


@dataclass
class PPOConfig:
    """PPO algorithm hyperparameters (stable-baselines3)."""
    learning_rate: float = 3e-4
    n_steps: int = 2048
    batch_size: int = 64
    n_epochs: int = 10
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_range: float = 0.2
    ent_coef: float = 0.01
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5

    def to_dict(self) -> Dict[str, Any]:
        return {
            "learning_rate": self.learning_rate,
            "n_steps": self.n_steps,
            "batch_size": self.batch_size,
            "n_epochs": self.n_epochs,
            "gamma": self.gamma,
            "gae_lambda": self.gae_lambda,
            "clip_range": self.clip_range,
            "ent_coef": self.ent_coef,
            "vf_coef": self.vf_coef,
            "max_grad_norm": self.max_grad_norm,
        }


@dataclass
class A2CConfig:
    """A2C algorithm hyperparameters."""
    learning_rate: float = 7e-4
    n_steps: int = 5
    gamma: float = 0.99
    gae_lambda: float = 1.0
    ent_coef: float = 0.01
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    rms_prop_eps: float = 1e-5

    def to_dict(self) -> Dict[str, Any]:
        return {
            "learning_rate": self.learning_rate,
            "n_steps": self.n_steps,
            "gamma": self.gamma,
            "gae_lambda": self.gae_lambda,
            "ent_coef": self.ent_coef,
            "vf_coef": self.vf_coef,
            "max_grad_norm": self.max_grad_norm,
            "rms_prop_eps": self.rms_prop_eps,
        }


@dataclass
class TrainingConfig:
    """Top-level training configuration."""
    algorithm: str = "ppo"          # "ppo" or "a2c"
    policy: str = "MlpPolicy"       # "MlpPolicy" or use CNN1DExtractor
    total_timesteps: int = 200_000
    n_envs: int = 4
    eval_freq: int = 20_000
    checkpoint_freq: int = 50_000
    normalize_obs: bool = True
    verbose: int = 1
    seed: Optional[int] = 42

    # Sub-configs
    env: EnvConfig = field(default_factory=EnvConfig)
    ppo: PPOConfig = field(default_factory=PPOConfig)
    a2c: A2CConfig = field(default_factory=A2CConfig)

    def get_algorithm_kwargs(self) -> Dict[str, Any]:
        if self.algorithm.lower() == "ppo":
            return self.ppo.to_dict()
        if self.algorithm.lower() == "a2c":
            return self.a2c.to_dict()
        return {}


# ---------------------------------------------------------------------------
# Pre-defined configs
# ---------------------------------------------------------------------------

def fast_test_config() -> TrainingConfig:
    """Minimal config for quick smoke-tests (low timesteps, small env)."""
    cfg = TrainingConfig(
        total_timesteps=10_000,
        n_envs=1,
        eval_freq=5_000,
        checkpoint_freq=5_000,
        verbose=0,
    )
    cfg.env.window_size = 32
    cfg.env.max_episode_steps = 100
    cfg.ppo.n_steps = 128
    cfg.ppo.batch_size = 32
    return cfg


def default_ppo_config() -> TrainingConfig:
    """Standard PPO training config."""
    return TrainingConfig(algorithm="ppo")


def default_a2c_config() -> TrainingConfig:
    """Standard A2C training config."""
    cfg = TrainingConfig(algorithm="a2c")
    cfg.a2c.learning_rate = 7e-4
    return cfg
