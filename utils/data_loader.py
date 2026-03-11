"""
Data loading utilities for OHLCV data.
Supports CSV/Parquet files, Binance REST API, and multiple timeframes.
"""

from __future__ import annotations

import io
import os
import time
import urllib.request
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
# Binance REST API fetcher
# ---------------------------------------------------------------------------

_BINANCE_BASE = "https://api.binance.com/api/v3/klines"

_BINANCE_INTERVALS = {
    "1m", "3m", "5m", "15m", "30m",
    "1h", "2h", "4h", "6h", "8h", "12h",
    "1d", "3d", "1w", "1M",
}

# Fallback: publicly accessible BTC daily data (2021) hosted on GitHub
_BTCUSD_FALLBACK_URL = (
    "https://raw.githubusercontent.com/gagolews/teaching-data"
    "/master/marek/btcusd_ohlcv_2021_dates.csv"
)


def fetch_binance_ohlcv(
    symbol: str = "BTCUSDT",
    interval: str = "1h",
    limit: int = 1000,
    start_time: Optional[int] = None,
    end_time: Optional[int] = None,
    use_fallback_on_error: bool = True,
) -> pd.DataFrame:
    """
    Fetch OHLCV candlestick data from the Binance REST API.

    No API key is required for public market data.

    Args:
        symbol:               Trading pair symbol (e.g. "BTCUSDT", "ETHUSDT").
        interval:             Candlestick interval.  One of: 1m, 3m, 5m, 15m,
                              30m, 1h, 2h, 4h, 6h, 8h, 12h, 1d, 3d, 1w, 1M.
        limit:                Number of candles to return (max 1000 per request).
        start_time:           Optional start time as UTC millisecond timestamp.
        end_time:             Optional end time as UTC millisecond timestamp.
        use_fallback_on_error: If True and the Binance API is unreachable,
                              falls back to a cached 2021 daily BTC/USD dataset
                              from GitHub so the script still runs in restricted
                              network environments.  Set to False to raise an
                              error instead.

    Returns:
        DataFrame with columns ['open', 'high', 'low', 'close', 'volume']
        and a UTC DatetimeIndex.

    Raises:
        RuntimeError: If Binance is unreachable and use_fallback_on_error=False.
    """
    if interval not in _BINANCE_INTERVALS:
        raise ValueError(
            f"Invalid interval '{interval}'. "
            f"Valid options: {sorted(_BINANCE_INTERVALS)}"
        )

    params = f"?symbol={symbol.upper()}&interval={interval}&limit={limit}"
    if start_time is not None:
        params += f"&startTime={int(start_time)}"
    if end_time is not None:
        params += f"&endTime={int(end_time)}"

    url = _BINANCE_BASE + params

    try:
        req = urllib.request.Request(url, headers={"User-Agent": "rl_supres/1.0"})
        with urllib.request.urlopen(req, timeout=15) as resp:
            import json
            raw = json.loads(resp.read())
    except Exception as exc:
        if not use_fallback_on_error:
            raise RuntimeError(
                f"Binance API request failed: {exc}\n"
                "Check your internet connection or set use_fallback_on_error=True."
            ) from exc
        print(
            f"[fetch_binance_ohlcv] WARNING: Binance API unreachable ({exc}).\n"
            "  Falling back to cached 2021 BTC/USD daily dataset from GitHub."
        )
        return _load_fallback_btcusd()

    # Binance klines columns:
    # 0:open_time 1:open 2:high 3:low 4:close 5:volume
    # 6:close_time 7:quote_volume 8:n_trades 9:taker_buy_base 10:taker_buy_quote 11:ignore
    df = pd.DataFrame(raw, columns=[
        "open_time", "open", "high", "low", "close", "volume",
        "close_time", "quote_volume", "n_trades",
        "taker_buy_base", "taker_buy_quote", "ignore",
    ])
    df = df[["open_time", "open", "high", "low", "close", "volume"]].copy()
    df["open_time"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
    df = df.set_index("open_time")
    df.index.name = "datetime"
    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.sort_index()
    return df


def fetch_binance_bulk(
    symbol: str = "BTCUSDT",
    interval: str = "1h",
    total_bars: int = 5000,
) -> pd.DataFrame:
    """
    Fetch more than 1000 bars from Binance by making multiple paginated requests.

    Args:
        symbol:      Trading pair (e.g. "BTCUSDT").
        interval:    Candlestick interval (e.g. "1h", "1d").
        total_bars:  Approximate total number of bars to fetch.

    Returns:
        DataFrame with OHLCV data.
    """
    all_dfs = []
    limit = 1000
    end_time = None

    while True:
        needed = total_bars - sum(len(d) for d in all_dfs)
        if needed <= 0:
            break
        batch = min(limit, needed)
        df = fetch_binance_ohlcv(
            symbol=symbol,
            interval=interval,
            limit=batch,
            end_time=end_time,
            use_fallback_on_error=False,
        )
        if df.empty:
            break
        all_dfs.append(df)
        # Move end_time to just before the first bar of the current batch
        end_time = int(df.index[0].timestamp() * 1000) - 1
        if len(df) < batch:
            break
        time.sleep(0.1)  # rate-limit courtesy

    if not all_dfs:
        return pd.DataFrame()
    combined = pd.concat(all_dfs).sort_index().drop_duplicates()
    return combined


def _load_fallback_btcusd() -> pd.DataFrame:
    """
    Load a cached 2021 BTC/USD daily OHLCV dataset from GitHub.
    Used as a fallback when the Binance API is not accessible.
    """
    req = urllib.request.Request(
        _BTCUSD_FALLBACK_URL,
        headers={"User-Agent": "rl_supres/1.0"},
    )
    with urllib.request.urlopen(req, timeout=15) as resp:
        raw = resp.read().decode("utf-8")

    lines = [l for l in raw.splitlines() if not l.startswith("#") and l.strip()]
    df = pd.read_csv(io.StringIO("\n".join(lines)))
    df.columns = [c.lower() for c in df.columns]
    df["date"] = pd.to_datetime(df["date"], utc=True)
    df = df.set_index("date")
    df.index.name = "datetime"
    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.sort_index()
    print(
        f"[fetch_binance_ohlcv] Fallback data loaded: "
        f"{len(df)} daily BTC/USD bars "
        f"({str(df.index[0])[:10]} → {str(df.index[-1])[:10]})."
    )
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
