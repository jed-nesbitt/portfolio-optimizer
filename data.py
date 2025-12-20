from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
import yfinance as yf


def load_tickers(csv_path: Path) -> List[str]:
    df = pd.read_csv(csv_path)
    if "ticker" in df.columns:
        s = df["ticker"]
    else:
        s = df.iloc[:, 0]

    tickers = (
        s.astype(str)
        .str.strip()
        .replace({"": np.nan, "nan": np.nan, "None": np.nan})
        .dropna()
        .drop_duplicates()
        .tolist()
    )
    return tickers


def download_close_and_volume(
    tickers: List[str],
    start: str,
    end_exclusive: str,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Returns (close, volume) as DataFrames indexed by date.

    NOTE: yfinance end= is exclusive.
    """

    raw = yf.download(
        tickers=tickers,
        start=start,
        end=end_exclusive,
        auto_adjust=True,
        progress=True,     # <-- progress bar ON
        threads=True,
        group_by="column",
    )

    # Extract Close + Volume robustly
    if isinstance(raw, pd.DataFrame) and "Close" in raw.columns:
        close = raw["Close"]
        volume = raw["Volume"]
    elif isinstance(raw, pd.DataFrame) and isinstance(raw.columns, pd.MultiIndex):
        # MultiIndex case
        close = raw["Close"]
        volume = raw["Volume"]
    else:
        raise ValueError("Unexpected yfinance output structure. Can't find Close/Volume.")

    if isinstance(close, pd.Series):
        close = close.to_frame(name=tickers[0])
    if isinstance(volume, pd.Series):
        volume = volume.to_frame(name=tickers[0])

    # Basic clean: forward/back fill within each series, drop dead columns
    close = close.sort_index().ffill().bfill().dropna(axis=1, how="all")
    volume = volume.reindex(close.index).ffill().bfill()
    volume = volume[close.columns]  # align columns exactly

    return close, volume
