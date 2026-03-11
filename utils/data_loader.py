"""
Data loading utilities for OHLCV data.
Supports CSV/Parquet files and multiple timeframes.
"""

from __future__ import annotations

import os
import numpy as np
import pandas as pd
from typing import Optional, Tuple, List, Union


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------

def load_csv(
    path: str,
    datetime_col: Optional[str] = None,
    lowercase_cols: bool = True,
) -> pd.DataFrame:
    """
    Load OHLCV data from a CSV file.

    Args:
        path:          Path to the CSV file.
        datetime_col:  Column name to use as the datetime index.
                       If None, the first column that looks like a datetime is used.
        lowercase_cols: Lowercase all column names (recommended).

    Returns:
        DataFrame with a DatetimeIndex and OHLCV columns.
    """
    df = pd.read_csv(path)
    if lowercase_cols:
        df.columns = [c.lower() for c in df.columns]

    # Detect datetime column
    if datetime_col is None:
        for col in df.columns:
            if col in ("date", "datetime", "timestamp", "time", "index"):
                datetime_col = col
                break

    if datetime_col and datetime_col in df.columns:
        df[datetime_col] = pd.to_datetime(df[datetime_col])
        df = df.set_index(datetime_col)
    elif not isinstance(df.index, pd.DatetimeIndex):
        # Try to parse index
        try:
            df.index = pd.to_datetime(df.index)
        except Exception:
            pass

    df = df.sort_index()
    return df


def load_parquet(path: str, lowercase_cols: bool = True) -> pd.DataFrame:
    """
    Load OHLCV data from a Parquet file.

    Args:
        path:          Path to the Parquet file.
        lowercase_cols: Lowercase all column names.

    Returns:
        DataFrame with a DatetimeIndex and OHLCV columns.
    """
    df = pd.read_parquet(path)
    if lowercase_cols:
        df.columns = [c.lower() for c in df.columns]
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)
    df = df.sort_index()
    return df


# ---------------------------------------------------------------------------
# Resampling
# ---------------------------------------------------------------------------

_RESAMPLE_RULES = {
    "1m": "1min",
    "5m": "5min",
    "15m": "15min",
    "30m": "30min",
    "1h": "1h",
    "4h": "4h",
    "1d": "1D",
    "1w": "1W",
}


def resample_ohlcv(df: pd.DataFrame, timeframe: str) -> pd.DataFrame:
    """
    Resample an OHLCV DataFrame to a coarser timeframe.

    Args:
        df:          DataFrame with DatetimeIndex and OHLCV columns.
        timeframe:   Target timeframe string ("1m", "5m", "15m", "1h", "4h", "1d").

    Returns:
        Resampled DataFrame.
    """
    rule = _RESAMPLE_RULES.get(timeframe, timeframe)
    ohlcv = df[["open", "high", "low", "close", "volume"]]
    resampled = ohlcv.resample(rule).agg(
        {
            "open": "first",
            "high": "max",
            "low": "min",
            "close": "last",
            "volume": "sum",
        }
    )
    return resampled.dropna(subset=["close"])


# ---------------------------------------------------------------------------
# Train / test splitting
# ---------------------------------------------------------------------------

def walk_forward_splits(
    df: pd.DataFrame,
    n_splits: int = 5,
    train_ratio: float = 0.7,
    gap_bars: int = 0,
) -> List[Tuple[pd.DataFrame, pd.DataFrame]]:
    """
    Generate walk-forward (time-series aware) train/test splits.

    Each split uses a contiguous training segment followed by a test segment.
    Splits are arranged so that later splits cover later time periods.

    Args:
        df:          Full OHLCV DataFrame.
        n_splits:    Number of train/test folds.
        train_ratio: Fraction of each fold assigned to training.
        gap_bars:    Number of bars to skip between train and test
                     (avoids look-ahead leakage).

    Returns:
        List of (train_df, test_df) tuples.
    """
    n = len(df)
    fold_size = n // n_splits
    splits = []
    for i in range(n_splits):
        start = i * fold_size
        end = start + fold_size if i < n_splits - 1 else n
        segment = df.iloc[start:end]
        train_end = int(len(segment) * train_ratio)
        train = segment.iloc[:train_end]
        test = segment.iloc[train_end + gap_bars :]
        if len(train) > 0 and len(test) > 0:
            splits.append((train.copy(), test.copy()))
    return splits


def train_test_split(
    df: pd.DataFrame,
    train_ratio: float = 0.8,
    gap_bars: int = 0,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Simple chronological train/test split.

    Args:
        df:          Full OHLCV DataFrame.
        train_ratio: Fraction of rows for training.
        gap_bars:    Number of bars to skip between train and test.

    Returns:
        (train_df, test_df).
    """
    split_idx = int(len(df) * train_ratio)
    train = df.iloc[:split_idx].copy()
    test = df.iloc[split_idx + gap_bars :].copy()
    return train, test


# ---------------------------------------------------------------------------
# Synthetic data generator (for testing / demonstration)
# ---------------------------------------------------------------------------

def generate_synthetic_ohlcv(
    n_bars: int = 1000,
    initial_price: float = 100.0,
    volatility: float = 0.01,
    drift: float = 0.0001,
    volume_base: float = 1_000_000.0,
    freq: str = "1h",
    seed: Optional[int] = 42,
) -> pd.DataFrame:
    """
    Generate a synthetic OHLCV DataFrame using a geometric random walk.

    Args:
        n_bars:        Number of bars to generate.
        initial_price: Starting price.
        volatility:    Per-bar return volatility (std).
        drift:         Per-bar return drift (mean).
        volume_base:   Base volume per bar (randomised ±50 %).
        freq:          Pandas frequency string for the datetime index.
        seed:          Random seed for reproducibility.

    Returns:
        DataFrame with columns ['open', 'high', 'low', 'close', 'volume']
        and a DatetimeIndex.
    """
    rng = np.random.default_rng(seed)
    returns = rng.normal(loc=drift, scale=volatility, size=n_bars)
    close = initial_price * np.exp(np.cumsum(returns))

    intra_vol = volatility * 0.5
    open_ = np.roll(close, 1)
    open_[0] = initial_price
    noise_high = np.abs(rng.normal(0, intra_vol, n_bars))
    noise_low = np.abs(rng.normal(0, intra_vol, n_bars))
    high = np.maximum(open_, close) + close * noise_high
    low = np.minimum(open_, close) - close * noise_low
    volume = volume_base * (1 + rng.uniform(-0.5, 0.5, n_bars))

    index = pd.date_range(start="2020-01-01", periods=n_bars, freq=freq)
    return pd.DataFrame(
        {
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume,
        },
        index=index,
    )
