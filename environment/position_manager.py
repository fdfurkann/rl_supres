"""
Position manager — handles trade execution, SL/TP monitoring, and P&L accounting.
"""

from dataclasses import dataclass
from typing import Optional
import numpy as np


@dataclass
class Trade:
    """Represents an open or closed trade."""

    direction: str          # "long" or "short"
    entry_price: float
    sl_price: float
    tp_price: float
    entry_bar: int
    exit_bar: Optional[int] = None
    exit_price: Optional[float] = None
    exit_reason: Optional[str] = None  # "sl", "tp", "manual", "eod"
    pnl_points: float = 0.0
    pnl_pct: float = 0.0


class PositionManager:
    """
    Manages open positions for the trading environment.

    Responsibilities:
    - Enter long/short positions with configurable SL/TP.
    - Monitor bar data to detect SL/TP hits (uses high/low of each bar).
    - Compute step-by-step unrealized P&L and realized P&L on close.
    - Apply commission and slippage per trade.

    Args:
        commission_pct: One-way commission as a fraction of trade value
                        (e.g. 0.001 = 0.1 %).
        slippage_pct:   One-way slippage as a fraction of price
                        (e.g. 0.0005 = 0.05 %).
        initial_equity: Starting equity (used only as a denominator for
                        pct-based reward scaling).
    """

    def __init__(
        self,
        commission_pct: float = 0.001,
        slippage_pct: float = 0.0005,
        initial_equity: float = 10_000.0,
    ):
        self.commission_pct = commission_pct
        self.slippage_pct = slippage_pct
        self.initial_equity = initial_equity

        self.equity: float = initial_equity
        self.trade: Optional[Trade] = None
        self.closed_trades: list = []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def in_position(self) -> bool:
        return self.trade is not None

    def reset(self):
        """Reset all state for a new episode."""
        self.equity = self.initial_equity
        self.trade = None
        self.closed_trades = []

    def enter_trade(
        self,
        direction: str,
        price: float,
        sl_distance: float,
        tp_distance: float,
        bar_index: int,
    ) -> float:
        """
        Open a new trade.

        Args:
            direction:   "long" or "short".
            price:       Current close price (execution price before slippage).
            sl_distance: Stop-loss distance in price units (positive value).
            tp_distance: Take-profit distance in price units (positive value).
            bar_index:   Current bar index.

        Returns:
            Transaction cost charged (positive number subtracted from equity).
        """
        if self.in_position:
            raise RuntimeError("Cannot enter a trade while already in position.")
        if sl_distance <= 0 or tp_distance <= 0:
            raise ValueError("sl_distance and tp_distance must be positive.")

        slip = price * self.slippage_pct
        if direction == "long":
            exec_price = price + slip
            sl_price = exec_price - sl_distance
            tp_price = exec_price + tp_distance
        elif direction == "short":
            exec_price = price - slip
            sl_price = exec_price + sl_distance
            tp_price = exec_price - tp_distance
        else:
            raise ValueError(f"Unknown direction: '{direction}'")

        cost = exec_price * self.commission_pct
        self.equity -= cost

        self.trade = Trade(
            direction=direction,
            entry_price=exec_price,
            sl_price=sl_price,
            tp_price=tp_price,
            entry_bar=bar_index,
        )
        return cost

    def step(
        self, bar_open: float, bar_high: float, bar_low: float, bar_close: float, bar_index: int
    ) -> float:
        """
        Advance the position by one bar.

        Checks if SL or TP is hit during the bar (using high/low).
        Returns the realized P&L for this bar (0 if no position, or realized
        PnL when trade closes, else unrealized mark-to-market change).

        Args:
            bar_open, bar_high, bar_low, bar_close: OHLC of current bar.
            bar_index: Current bar index.

        Returns:
            float: P&L change this step (in currency units).
        """
        if not self.in_position:
            return 0.0

        t = self.trade
        prev_close_pnl = self._unrealized_pnl(t, bar_open)  # approx start-of-bar

        # Check SL/TP hit (intra-bar using high/low)
        hit_sl = (t.direction == "long" and bar_low <= t.sl_price) or (
            t.direction == "short" and bar_high >= t.sl_price
        )
        hit_tp = (t.direction == "long" and bar_high >= t.tp_price) or (
            t.direction == "short" and bar_low <= t.tp_price
        )

        if hit_sl or hit_tp:
            # Use the better fill assumption: if both hit, TP wins (optimistic)
            if hit_tp:
                exit_price = t.tp_price
                exit_reason = "tp"
            else:
                exit_price = t.sl_price
                exit_reason = "sl"

            pnl = self._close_trade(exit_price, exit_reason, bar_index)
            return pnl

        # No close — mark-to-market change
        end_pnl = self._unrealized_pnl(t, bar_close)
        return end_pnl - prev_close_pnl

    def close_trade(self, price: float, bar_index: int) -> float:
        """
        Manually close the current trade at the given price.

        Returns:
            Realized P&L (after costs).
        """
        if not self.in_position:
            return 0.0
        return self._close_trade(price, "manual", bar_index)

    def get_position_state(self, current_price: float) -> dict:
        """
        Return a dict of current position state for use in observations.

        Keys:
            position:          0 (no pos), 1 (long), -1 (short)
            time_in_trade:     bars since entry (0 if not in position)
            entry_price_ratio: (current_price / entry_price) - 1  (0 if not in pos)
            sl_distance_pct:   SL distance as fraction of current price (0 if not in pos)
            tp_distance_pct:   TP distance as fraction of current price (0 if not in pos)
        """
        if not self.in_position:
            return {
                "position": 0.0,
                "time_in_trade": 0.0,
                "entry_price_ratio": 0.0,
                "sl_distance_pct": 0.0,
                "tp_distance_pct": 0.0,
            }
        t = self.trade
        pos_sign = 1.0 if t.direction == "long" else -1.0
        time_in = 0.0  # filled by caller using bar_index - t.entry_bar
        if current_price > 0:
            entry_ratio = current_price / t.entry_price - 1.0
            sl_dist = abs(t.sl_price - current_price) / current_price
            tp_dist = abs(t.tp_price - current_price) / current_price
        else:
            entry_ratio = sl_dist = tp_dist = 0.0
        return {
            "position": pos_sign,
            "time_in_trade": time_in,
            "entry_price_ratio": float(entry_ratio),
            "sl_distance_pct": float(sl_dist),
            "tp_distance_pct": float(tp_dist),
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _unrealized_pnl(self, trade: Trade, current_price: float) -> float:
        if trade.direction == "long":
            return current_price - trade.entry_price
        return trade.entry_price - current_price

    def _close_trade(self, exit_price: float, exit_reason: str, bar_index: int) -> float:
        t = self.trade
        slip = exit_price * self.slippage_pct
        if t.direction == "long":
            exec_exit = exit_price - slip
            pnl_points = exec_exit - t.entry_price
        else:
            exec_exit = exit_price + slip
            pnl_points = t.entry_price - exec_exit

        cost = exec_exit * self.commission_pct
        net_pnl = pnl_points - cost

        pnl_pct = net_pnl / t.entry_price if t.entry_price != 0 else 0.0
        t.exit_bar = bar_index
        t.exit_price = exec_exit
        t.exit_reason = exit_reason
        t.pnl_points = pnl_points
        t.pnl_pct = pnl_pct

        self.equity += net_pnl
        self.closed_trades.append(t)
        self.trade = None
        return net_pnl
