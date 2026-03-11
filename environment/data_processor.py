"""
Data processor for OHLCV data normalization and feature engineering.
Only OHLCV-based transformations are used — no external indicators.
"""

import numpy as np
import pandas as pd
from typing import Optional


class DataProcessor:
    """
    Processes raw OHLCV data into normalized observation windows.

    Normalization strategy:
    - Open/High/Low: expressed as ratio relative to Close (price-normalized).
    - Close: expressed as log-return from previous Close.
    - Volume: log(1 + volume) followed by rolling z-score normalization.
    """

    def __init__(self, window_size: int = 64, volume_lookback: int = 20):
        """
        Args:
            window_size: Number of bars to include in each observation window.
            volume_lookback: Rolling window size for volume z-score normalization.
        """
        self.window_size = window_size
        self.volume_lookback = volume_lookback

    def process(self, df: pd.DataFrame) -> np.ndarray:
        """
        Transform a raw OHLCV DataFrame into a normalized feature array.

        Args:
            df: DataFrame with columns ['open', 'high', 'low', 'close', 'volume']
                and a datetime index.  Must have at least (window_size + 1) rows.

        Returns:
            Normalized array of shape (len(df), 5).
            Columns: [open_ratio, high_ratio, low_ratio, close_logret, volume_zscore].
            The first row contains zeros (no previous close for log-return).
        """
        df = df.copy()
        df.columns = [c.lower() for c in df.columns]
        required = {"open", "high", "low", "close", "volume"}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"DataFrame missing columns: {missing}")

        close = df["close"].values.astype(np.float64)
        open_ = df["open"].values.astype(np.float64)
        high = df["high"].values.astype(np.float64)
        low = df["low"].values.astype(np.float64)
        volume = df["volume"].values.astype(np.float64)

        # Price features relative to current close (dimensionless, stationary-ish)
        open_ratio = np.where(close != 0, open_ / close - 1.0, 0.0)
        high_ratio = np.where(close != 0, high / close - 1.0, 0.0)
        low_ratio = np.where(close != 0, low / close - 1.0, 0.0)

        # Close: log-return (first value is 0)
        close_lr = np.zeros_like(close)
        prev_close = close[:-1]
        curr_close = close[1:]
        valid = prev_close > 0
        close_lr[1:] = np.where(valid, np.log(curr_close / prev_close), 0.0)

        # Volume: log-transform + rolling z-score
        log_vol = np.log1p(volume)
        vol_zscore = self._rolling_zscore(log_vol, self.volume_lookback)

        features = np.stack(
            [open_ratio, high_ratio, low_ratio, close_lr, vol_zscore], axis=1
        ).astype(np.float32)

        # Clip extreme values to prevent exploding observations
        features = np.clip(features, -10.0, 10.0)
        return features

    def get_window(self, features: np.ndarray, index: int) -> np.ndarray:
        """
        Extract a (window_size, 5) observation window ending at *index* (inclusive).

        Args:
            features: Full processed feature array of shape (T, 5).
            index: Current bar index (0-based).

        Returns:
            Array of shape (window_size, 5).  Rows before the start are zero-padded.
        """
        start = index - self.window_size + 1
        if start < 0:
            pad = np.zeros((-start, features.shape[1]), dtype=np.float32)
            window = features[: index + 1]
            return np.concatenate([pad, window], axis=0)
        return features[start : index + 1].copy()

    @staticmethod
    def _rolling_zscore(x: np.ndarray, window: int) -> np.ndarray:
        """Compute rolling z-score with NaN-safe fallback to 0."""
        result = np.zeros_like(x, dtype=np.float64)
        for i in range(len(x)):
            start = max(0, i - window + 1)
            subset = x[start : i + 1]
            std = np.std(subset)
            if std > 0:
                result[i] = (x[i] - np.mean(subset)) / std
        return result.astype(np.float32)

    @staticmethod
    def validate_dataframe(df: pd.DataFrame) -> pd.DataFrame:
        """
        Validate and clean a raw OHLCV DataFrame.

        - Ensures lowercase column names.
        - Drops rows with NaN or non-positive close price.
        - Sorts by index.

        Returns:
            Cleaned DataFrame.
        """
        df = df.copy()
        df.columns = [c.lower() for c in df.columns]
        required = ["open", "high", "low", "close", "volume"]
        for col in required:
            if col not in df.columns:
                raise ValueError(f"Missing required column: '{col}'")
        df = df.dropna(subset=required)
        df = df[df["close"] > 0]
        df = df.sort_index()
        return df
