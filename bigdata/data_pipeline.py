"""Data pipeline utilities for downloading and preparing market data."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

import pandas as pd
import yfinance as yf


@dataclass(frozen=True)
class RollingWindow:
    """A single rolling train/test split."""

    train_start: pd.Timestamp
    train_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp
    train_data: pd.DataFrame
    test_data: pd.DataFrame


@dataclass(frozen=True)
class DataPipelineResult:
    """Standard output bundle for the cleaned market data pipeline."""

    clean_return_matrix: pd.DataFrame
    training_test_windows: List[RollingWindow]
    clean_price_matrix: pd.DataFrame
    missing_ratio_by_asset: pd.Series
    dropped_assets: List[str]


def download_adjusted_close_prices(
    tickers: List[str],
    start_date: str = "2020-01-01",
    end_date: str | None = None,
) -> pd.DataFrame:
    """Download daily adjusted close prices for a list of tickers."""

    raw = yf.download(
        tickers=tickers,
        start=start_date,
        end=end_date,
        auto_adjust=False,
        progress=False,
        group_by="column",
    )

    if raw.empty:
        raise ValueError("Downloaded price data is empty.")

    if isinstance(raw.columns, pd.MultiIndex):
        if "Adj Close" in raw.columns.get_level_values(0):
            prices = raw["Adj Close"].copy()
        elif "Close" in raw.columns.get_level_values(0):
            prices = raw["Close"].copy()
        else:
            raise ValueError("Neither 'Adj Close' nor 'Close' found in downloaded data.")
    else:
        if "Adj Close" in raw.columns:
            prices = raw[["Adj Close"]].copy()
            prices.columns = tickers if len(tickers) == 1 else ["Price"]
        elif "Close" in raw.columns:
            prices = raw[["Close"]].copy()
            prices.columns = tickers if len(tickers) == 1 else ["Price"]
        else:
            raise ValueError("Neither 'Adj Close' nor 'Close' found in downloaded data.")

    prices.index = pd.to_datetime(prices.index)
    prices = prices.sort_index()
    return prices


def align_and_clean_prices(
    prices: pd.DataFrame,
    missing_ratio_threshold: float = 0.10,
) -> tuple[pd.DataFrame, pd.Series, List[str]]:
    """Align prices to a unified business-day calendar and clean missing values."""

    if prices.empty:
        raise ValueError("Price matrix is empty.")

    full_calendar = pd.bdate_range(prices.index.min(), prices.index.max())
    aligned = prices.reindex(full_calendar).sort_index()

    missing_ratio = aligned.isna().mean().sort_values(ascending=False)
    keep_columns = missing_ratio[missing_ratio <= missing_ratio_threshold].index.tolist()
    dropped_assets = missing_ratio[missing_ratio > missing_ratio_threshold].index.tolist()

    if not keep_columns:
        raise ValueError("All assets were dropped after applying the missing data threshold.")

    cleaned = aligned[keep_columns].ffill()
    cleaned = cleaned.dropna(how="any")

    if cleaned.empty:
        raise ValueError("Price matrix is empty after alignment and missing value handling.")

    return cleaned, missing_ratio, dropped_assets


def compute_clean_return_matrix(
    cleaned_prices: pd.DataFrame,
    winsorize_limit: float = 0.01,
) -> pd.DataFrame:
    """Compute aligned daily simple returns and clip extreme values."""

    if cleaned_prices.empty:
        raise ValueError("Cleaned price matrix is empty.")

    returns = cleaned_prices.pct_change().dropna(how="any")

    if returns.empty:
        raise ValueError("Return matrix is empty after pct_change.")

    lower = returns.quantile(winsorize_limit)
    upper = returns.quantile(1 - winsorize_limit)
    returns = returns.clip(lower=lower, upper=upper, axis=1)

    if returns.isna().any().any():
        raise ValueError("Clean return matrix still contains NaN values.")

    return returns


def create_rolling_windows(
    clean_return_matrix: pd.DataFrame,
    train_window: int = 252,
    test_window: int = 21,
    step_size: int = 21,
) -> List[RollingWindow]:
    """Create rolling train/test windows from a clean return matrix."""

    if clean_return_matrix.empty:
        raise ValueError("Clean return matrix is empty.")

    total_rows = len(clean_return_matrix)
    windows: List[RollingWindow] = []

    start = 0
    while start + train_window + test_window <= total_rows:
        train_slice = clean_return_matrix.iloc[start : start + train_window]
        test_slice = clean_return_matrix.iloc[
            start + train_window : start + train_window + test_window
        ]

        windows.append(
            RollingWindow(
                train_start=train_slice.index[0],
                train_end=train_slice.index[-1],
                test_start=test_slice.index[0],
                test_end=test_slice.index[-1],
                train_data=train_slice,
                test_data=test_slice,
            )
        )

        start += step_size

    if not windows:
        raise ValueError("Not enough data to create at least one rolling train/test window.")

    return windows


def build_data_pipeline(
    tickers: List[str],
    start_date: str = "2020-01-01",
    end_date: str | None = None,
    missing_ratio_threshold: float = 0.10,
    winsorize_limit: float = 0.01,
    train_window: int = 252,
    test_window: int = 21,
    step_size: int = 21,
) -> DataPipelineResult:
    """Run the full market data pipeline for downstream risk modeling."""

    prices = download_adjusted_close_prices(
        tickers=tickers,
        start_date=start_date,
        end_date=end_date,
    )
    cleaned_prices, missing_ratio, dropped_assets = align_and_clean_prices(
        prices,
        missing_ratio_threshold=missing_ratio_threshold,
    )
    clean_return_matrix = compute_clean_return_matrix(
        cleaned_prices,
        winsorize_limit=winsorize_limit,
    )
    windows = create_rolling_windows(
        clean_return_matrix,
        train_window=train_window,
        test_window=test_window,
        step_size=step_size,
    )

    return DataPipelineResult(
        clean_return_matrix=clean_return_matrix,
        training_test_windows=windows,
        clean_price_matrix=cleaned_prices,
        missing_ratio_by_asset=missing_ratio,
        dropped_assets=dropped_assets,
    )
