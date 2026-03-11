# RL Supres — Reinforcement Learning Trading Bot

A **Gymnasium-compatible reinforcement learning trading environment** that implements a support/resistance breakout strategy using only OHLCV (Open, High, Low, Close, Volume) data.

## Strategy Overview

The RL agent learns to:
1. **Select a lookback period** (X bars) and **direction** (track highs or lows).
2. **Detect breakouts** — when price closes above the rolling X-bar high (long) or below the rolling X-bar low (short).
3. **Set Stop-Loss (Y)** and **Take-Profit (Z)** distances as fractions of price at entry.

No external indicators are used. The agent operates on normalized OHLCV data only.

---

## Project Structure

```
rl_supres/
├── environment/
│   ├── trading_env.py          # Main Gymnasium environment
│   ├── position_manager.py     # Trade execution, SL/TP monitoring, P&L
│   └── data_processor.py       # OHLCV normalization (log-returns, z-score)
├── agents/
│   ├── sb3_agent.py            # Stable-Baselines3 training/eval wrapper
│   └── custom_policies.py      # CNN-1D feature extractor
├── utils/
│   ├── metrics.py              # Sharpe, drawdown, profit factor, win rate
│   ├── data_loader.py          # CSV/Parquet loading, resampling, splits
│   └── visualization.py        # Equity curves, drawdown, trade plots
├── training/
│   ├── train.py                # CLI training script
│   └── hyperparams.py          # Hyperparameter dataclasses
├── backtesting/
│   ├── backtest_engine.py      # Backtest runner + walk-forward
│   └── performance_analysis.py # Aggregation and reporting utilities
├── examples/
│   ├── basic_training.py       # Quick start: train + evaluate
│   └── strategy_evaluation.py  # Walk-forward backtest example
└── requirements.txt
```

---

## Installation

```bash
pip install -r requirements.txt
```

**Dependencies:** `stable-baselines3`, `gymnasium`, `torch`, `pandas`, `numpy`, `matplotlib`

---

## Quick Start

### 1 — Basic Training (synthetic data)

```bash
python examples/basic_training.py
```

### 2 — Train on your own data

```bash
python training/train.py --data path/to/ohlcv.csv --algo ppo --timesteps 200000
```

### 3 — Full strategy evaluation with walk-forward

```bash
python examples/strategy_evaluation.py
```

---

## Observation Space

Each observation is a flat vector of length `window_size × 5 + 5`:

| Slice | Contents |
|---|---|
| `[0 : window_size*5]` | Last N bars of OHLCV, normalized (see below) |
| `[-5:]` | Position state: sign, time-in-trade, entry ratio, SL dist, TP dist |

**Normalization:**
- Open / High / Low → `(price / close) - 1` (relative to current close)
- Close → log-return: `log(close_t / close_{t-1})`
- Volume → log-transform + rolling z-score

---

## Action Space

Discrete — each integer encodes a unique combination of:

| Parameter | Values |
|---|---|
| X (lookback bars) | 5, 10, 15, 20, 30, 50 |
| D (direction) | high, low |
| Y (SL %) | 0.5%, 1.0%, 1.5%, 2.0%, 3.0% |
| Z (TP %) | 1.0%, 1.5%, 2.0%, 3.0%, 5.0% |

Total actions: 6 × 2 × 5 × 5 = **300 discrete actions**

---

## Reward Function

```
reward_t = Δequity_t - risk_penalty
```

- **Δequity**: mark-to-market change each bar (dense, step-based).
- **risk_penalty**: `risk_penalty_coef × sl_pct` at trade entry — penalises excessively wide stop-losses.
- Commission and slippage are deducted automatically from equity.

---

## Using a Custom CNN Policy

```python
from stable_baselines3 import PPO
from agents.custom_policies import CNN1DExtractor
from environment.trading_env import TradingEnv

policy_kwargs = dict(
    features_extractor_class=CNN1DExtractor,
    features_extractor_kwargs=dict(window_size=64, features_dim=256),
)
env = TradingEnv(df=your_df)
model = PPO("MlpPolicy", env, policy_kwargs=policy_kwargs, verbose=1)
model.learn(200_000)
```

---

## Loading Real Data

```python
from utils.data_loader import load_csv, resample_ohlcv

df = load_csv("btcusdt_1h.csv")           # auto-detects datetime column
df_4h = resample_ohlcv(df, "4h")          # resample to 4-hour bars
```

---

## Walk-Forward Backtest

```python
from agents.sb3_agent import SB3Agent
from backtesting.backtest_engine import BacktestEngine
from backtesting.performance_analysis import aggregate_walk_forward

agent = SB3Agent(algorithm="ppo", env_kwargs={"window_size": 64})
agent.train(df=train_df, total_timesteps=200_000)

engine = BacktestEngine(agent=agent, env_kwargs={"window_size": 64})
results = engine.walk_forward_backtest(df, n_splits=5)
agg = aggregate_walk_forward(results)
```

---

## Performance Metrics

| Metric | Description |
|---|---|
| `total_return_pct` | Total return over the episode |
| `annualised_return_pct` | Annualised return |
| `sharpe_ratio` | Risk-adjusted return (annualised) |
| `max_drawdown_pct` | Peak-to-trough drawdown |
| `calmar_ratio` | Annualised return / max drawdown |
| `profit_factor` | Gross profit / gross loss |
| `win_rate_pct` | Percentage of winning trades |
| `n_trades` | Total trades executed |
| `avg_trade_pct` | Average trade P&L as % |
| `avg_trade_duration_bars` | Average bars per trade |

---

## Configuration

All hyperparameters are in `training/hyperparams.py`:

```python
from training.hyperparams import TrainingConfig, EnvConfig, PPOConfig

cfg = TrainingConfig(
    algorithm="ppo",
    total_timesteps=500_000,
    n_envs=8,
    env=EnvConfig(
        window_size=64,
        commission_pct=0.001,
        slippage_pct=0.0005,
        allow_short=True,
        risk_penalty_coef=0.1,
    ),
)
```
