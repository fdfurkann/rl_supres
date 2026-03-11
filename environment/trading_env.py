"""
Gymnasium-compatible trading environment for RL-based support/resistance breakout strategy.

Strategy:
    At each step the agent selects a discrete action that encodes four parameters:
        X  — lookback period (number of bars) for high/low identification.
        D  — direction: track highest (potential long breakout) or lowest (short breakout).
        Y  — stop-loss distance as a percentage of the current close price.
        Z  — take-profit distance as a percentage of the current close price.

    Breakout logic:
        - If D == "high" and current close > rolling_high(X), a long trade is entered.
        - If D == "low"  and current close < rolling_low(X),  a short trade is entered.
        - SL = close * Y; TP = close * Z (percentage of price).

Observation:
    - Last `window_size` bars of normalized OHLCV features (shape: window_size x 5).
    - Position state vector of length 5:
        [position_sign, time_in_trade_normalized, entry_price_ratio,
         sl_distance_pct, tp_distance_pct].
    - Flattened and concatenated into a 1-D vector.

Action Space:
    Discrete — each integer maps to a unique (X, D, Y, Z) combination.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import gymnasium as gym
from gymnasium import spaces
from typing import Optional, Tuple, Dict, Any, List

from environment.data_processor import DataProcessor
from environment.position_manager import PositionManager


# ---------------------------------------------------------------------------
# Action grid definitions
# ---------------------------------------------------------------------------
LOOKBACK_VALUES: List[int] = [5, 10, 15, 20, 30, 50]          # X: bars
DIRECTION_VALUES: List[str] = ["high", "low"]                   # D
SL_PCT_VALUES: List[float] = [0.005, 0.01, 0.015, 0.02, 0.03]  # Y: fraction of price
TP_PCT_VALUES: List[float] = [0.01, 0.015, 0.02, 0.03, 0.05]   # Z: fraction of price

N_ACTIONS = len(LOOKBACK_VALUES) * len(DIRECTION_VALUES) * len(SL_PCT_VALUES) * len(TP_PCT_VALUES)

# Pre-build mapping: action_id → (x, direction, sl_pct, tp_pct)
_ACTION_MAP: Dict[int, Tuple[int, str, float, float]] = {}
_idx = 0
for _x in LOOKBACK_VALUES:
    for _d in DIRECTION_VALUES:
        for _y in SL_PCT_VALUES:
            for _z in TP_PCT_VALUES:
                _ACTION_MAP[_idx] = (_x, _d, _y, _z)
                _idx += 1


class TradingEnv(gym.Env):
    """
    Gymnasium environment for the RL support/resistance breakout strategy.

    Args:
        df:               OHLCV DataFrame with datetime index and columns
                          ['open', 'high', 'low', 'close', 'volume'].
        window_size:      Number of historical bars in each observation.
        commission_pct:   One-way commission (e.g. 0.001 = 0.1 %).
        slippage_pct:     One-way slippage (e.g. 0.0005 = 0.05 %).
        initial_equity:   Starting account equity.
        max_episode_steps: Maximum number of steps per episode (None = full data).
        reward_scaling:   Scalar to multiply step rewards (default 1.0).
        risk_penalty_coef: Penalty coefficient for wide SL distances to discourage
                           excessive risk per trade (set to 0 to disable).
        allow_short:      If False, only long trades are permitted.
        volume_lookback:  Rolling window for volume z-score normalization.
    """

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        df: pd.DataFrame,
        window_size: int = 64,
        commission_pct: float = 0.001,
        slippage_pct: float = 0.0005,
        initial_equity: float = 10_000.0,
        max_episode_steps: Optional[int] = None,
        reward_scaling: float = 1.0,
        risk_penalty_coef: float = 0.1,
        allow_short: bool = True,
        volume_lookback: int = 20,
    ):
        super().__init__()

        self.processor = DataProcessor(window_size=window_size, volume_lookback=volume_lookback)
        df_clean = DataProcessor.validate_dataframe(df)
        self._raw_df = df_clean.reset_index(drop=True)
        self._features = self.processor.process(df_clean)

        self.window_size = window_size
        self.commission_pct = commission_pct
        self.slippage_pct = slippage_pct
        self.initial_equity = initial_equity
        self.max_episode_steps = max_episode_steps
        self.reward_scaling = reward_scaling
        self.risk_penalty_coef = risk_penalty_coef
        self.allow_short = allow_short

        # Position manager
        self.pos_manager = PositionManager(
            commission_pct=commission_pct,
            slippage_pct=slippage_pct,
            initial_equity=initial_equity,
        )

        # Spaces
        obs_dim = window_size * 5 + 5  # OHLCV window + position state
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )
        self.action_space = spaces.Discrete(N_ACTIONS)

        # Episode state
        self._current_step: int = 0
        self._start_step: int = window_size  # need enough history for max lookback
        self._end_step: int = len(self._raw_df) - 1
        self._episode_steps: int = 0
        self._prev_equity: float = initial_equity

        # Last decoded action (for info)
        self._last_action_params: Optional[Tuple] = None

    # ------------------------------------------------------------------
    # Gymnasium API
    # ------------------------------------------------------------------

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        super().reset(seed=seed)

        self.pos_manager.reset()
        self._prev_equity = self.initial_equity
        self._episode_steps = 0

        # Randomise starting point within valid range for training diversity
        max_start = self._end_step - (self.max_episode_steps or 0)
        max_start = max(self._start_step, max_start)
        if options and options.get("start_step") is not None:
            self._current_step = int(options["start_step"])
        else:
            self._current_step = int(self.np_random.integers(self._start_step, max_start + 1))

        obs = self._get_obs()
        info = self._get_info()
        return obs, info

    def step(
        self, action: int
    ) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        assert self.action_space.contains(action), f"Invalid action: {action}"

        x, direction, sl_pct, tp_pct = _ACTION_MAP[action]
        self._last_action_params = (x, direction, sl_pct, tp_pct)

        # Clamp short action if not allowed
        if not self.allow_short and direction == "low":
            direction = "high"

        bar = self._raw_df.iloc[self._current_step]
        close = float(bar["close"])
        high = float(bar["high"])
        low = float(bar["low"])
        bar_open = float(bar["open"])

        # --- Advance position (check SL/TP hits on current bar) ---
        step_pnl = self.pos_manager.step(
            bar_open=bar_open,
            bar_high=high,
            bar_low=low,
            bar_close=close,
            bar_index=self._current_step,
        )

        # --- Breakout detection & trade entry (if not already in position) ---
        risk_penalty = 0.0
        if not self.pos_manager.in_position:
            breakout, trade_direction = self._detect_breakout(
                x, direction, self._current_step, close
            )
            if breakout:
                sl_distance = close * sl_pct
                tp_distance = close * tp_pct
                try:
                    cost = self.pos_manager.enter_trade(
                        direction=trade_direction,
                        price=close,
                        sl_distance=sl_distance,
                        tp_distance=tp_distance,
                        bar_index=self._current_step,
                    )
                    # Optional risk penalty: penalise wide SL to encourage tight risk
                    risk_penalty = self.risk_penalty_coef * sl_pct
                    step_pnl -= cost  # cost already deducted from equity; align reward
                except (RuntimeError, ValueError):
                    pass

        # --- Reward ---
        current_equity = self.pos_manager.equity
        reward = float((current_equity - self._prev_equity) - risk_penalty)
        reward *= self.reward_scaling
        self._prev_equity = current_equity

        # --- Advance step ---
        self._current_step += 1
        self._episode_steps += 1

        # --- Termination conditions ---
        truncated = False
        terminated = False

        if self._current_step > self._end_step:
            terminated = True
        if self.max_episode_steps and self._episode_steps >= self.max_episode_steps:
            truncated = True

        # Close open trade at episode end
        if (terminated or truncated) and self.pos_manager.in_position:
            close_price = float(self._raw_df.iloc[self._current_step - 1]["close"])
            eod_pnl = self.pos_manager.close_trade(close_price, self._current_step - 1)
            reward += eod_pnl * self.reward_scaling

        obs = self._get_obs() if not (terminated or truncated) else np.zeros(
            self.observation_space.shape, dtype=np.float32
        )
        info = self._get_info()
        return obs, reward, terminated, truncated, info

    def render(self) -> None:
        t = self.pos_manager.trade
        print(
            f"Step {self._current_step:5d} | "
            f"Equity: {self.pos_manager.equity:10.2f} | "
            f"In position: {self.pos_manager.in_position} | "
            f"Trade: {t}"
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _get_obs(self) -> np.ndarray:
        window = self.processor.get_window(self._features, self._current_step)
        flat_ohlcv = window.flatten()

        # Position state
        close = float(self._raw_df.iloc[self._current_step]["close"])
        ps = self.pos_manager.get_position_state(close)
        if self.pos_manager.trade is not None:
            ps["time_in_trade"] = float(
                self._current_step - self.pos_manager.trade.entry_bar
            ) / max(self.window_size, 1)

        pos_state = np.array(
            [
                ps["position"],
                ps["time_in_trade"],
                ps["entry_price_ratio"],
                ps["sl_distance_pct"],
                ps["tp_distance_pct"],
            ],
            dtype=np.float32,
        )
        pos_state = np.clip(pos_state, -10.0, 10.0)
        return np.concatenate([flat_ohlcv, pos_state])

    def _get_info(self) -> Dict[str, Any]:
        return {
            "step": self._current_step,
            "equity": self.pos_manager.equity,
            "in_position": self.pos_manager.in_position,
            "n_trades": len(self.pos_manager.closed_trades),
            "last_action_params": self._last_action_params,
        }

    def _detect_breakout(
        self,
        lookback: int,
        direction: str,
        current_idx: int,
        current_close: float,
    ) -> Tuple[bool, str]:
        """
        Detect if current_close breaks the rolling high or low.

        Returns:
            (breakout_detected, trade_direction).
        """
        start = max(0, current_idx - lookback)
        # Exclude the current bar from the rolling window
        window_slice = self._raw_df.iloc[start:current_idx]
        if window_slice.empty:
            return False, ""

        if direction == "high":
            level = float(window_slice["high"].max())
            if current_close > level:
                return True, "long"
        else:  # direction == "low"
            level = float(window_slice["low"].min())
            if current_close < level:
                return True, "short"

        return False, ""
