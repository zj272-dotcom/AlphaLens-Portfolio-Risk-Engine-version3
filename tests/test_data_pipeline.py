from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from bigdata.data_pipeline import (
    DataPipelineResult,
    align_and_clean_prices,
    build_data_pipeline,
    compute_clean_return_matrix,
    create_rolling_windows,
    download_adjusted_close_prices,
)


# ========================
# download_adjusted_close_prices
# ========================

@patch("bigdata.data_pipeline.yf.download")
def test_download_adjusted_close_prices_multiindex(mock_download):
    columns = pd.MultiIndex.from_product([["Adj Close"], ["SPY", "QQQ"]])
    raw = pd.DataFrame(
        [[100.0, 200.0], [101.0, 202.0]],
        index=pd.to_datetime(["2024-01-01", "2024-01-02"]),
        columns=columns,
    )
    mock_download.return_value = raw

    prices = download_adjusted_close_prices(["SPY", "QQQ"])

    assert list(prices.columns) == ["SPY", "QQQ"]
    assert prices.index.is_monotonic_increasing


@patch("bigdata.data_pipeline.yf.download")
def test_download_adjusted_close_prices_multiindex_close_fallback(mock_download):
    columns = pd.MultiIndex.from_product([["Close"], ["SPY", "QQQ"]])
    raw = pd.DataFrame(
        [[100.0, 200.0], [101.0, 202.0]],
        index=pd.to_datetime(["2024-01-01", "2024-01-02"]),
        columns=columns,
    )
    mock_download.return_value = raw

    prices = download_adjusted_close_prices(["SPY", "QQQ"])

    assert list(prices.columns) == ["SPY", "QQQ"]
    assert prices.index.is_monotonic_increasing
    assert prices.loc[pd.Timestamp("2024-01-01"), "SPY"] == 100.0


@patch("bigdata.data_pipeline.yf.download")
def test_download_adjusted_close_prices_single_ticker(mock_download):
    raw = pd.DataFrame(
        {"Adj Close": [100.0, 101.0]},
        index=pd.to_datetime(["2024-01-01", "2024-01-02"]),
    )
    mock_download.return_value = raw

    prices = download_adjusted_close_prices(["SPY"])

    assert list(prices.columns) == ["SPY"]


@patch("bigdata.data_pipeline.yf.download")
def test_download_adjusted_close_prices_single_ticker_adj_close_branch(mock_download):
    raw = pd.DataFrame(
        {"Adj Close": [100.0, 101.0]},
        index=pd.to_datetime(["2024-01-01", "2024-01-02"]),
    )
    mock_download.return_value = raw

    prices = download_adjusted_close_prices(["SPY"])

    assert list(prices.columns) == ["SPY"]
    assert prices.iloc[0, 0] == 100.0


@patch("bigdata.data_pipeline.yf.download")
def test_download_adjusted_close_prices_single_ticker_close_branch(mock_download):
    raw = pd.DataFrame(
        {"Close": [100.0, 101.0]},
        index=pd.to_datetime(["2024-01-01", "2024-01-02"]),
    )
    mock_download.return_value = raw

    prices = download_adjusted_close_prices(["SPY"])

    assert list(prices.columns) == ["SPY"]
    assert prices.iloc[1, 0] == 101.0


@patch("bigdata.data_pipeline.yf.download")
def test_download_adjusted_close_prices_empty_raises(mock_download):
    mock_download.return_value = pd.DataFrame()

    with pytest.raises(ValueError, match="Downloaded price data is empty"):
        download_adjusted_close_prices(["SPY"])


@patch("bigdata.data_pipeline.yf.download")
def test_download_adjusted_close_prices_missing_columns_raises(mock_download):
    raw = pd.DataFrame(
        {"Open": [1, 2], "High": [2, 3]},
        index=pd.to_datetime(["2024-01-01", "2024-01-02"]),
    )
    mock_download.return_value = raw

    with pytest.raises(ValueError, match="Neither 'Adj Close' nor 'Close' found"):
        download_adjusted_close_prices(["SPY"])


@patch("bigdata.data_pipeline.yf.download")
def test_download_adjusted_close_prices_multiindex_missing_columns_raises(mock_download):
    columns = pd.MultiIndex.from_product([["Open"], ["SPY", "QQQ"]])
    raw = pd.DataFrame(
        [[100.0, 200.0], [101.0, 202.0]],
        index=pd.to_datetime(["2024-01-01", "2024-01-02"]),
        columns=columns,
    )
    mock_download.return_value = raw

    with pytest.raises(ValueError, match="Neither 'Adj Close' nor 'Close' found"):
        download_adjusted_close_prices(["SPY", "QQQ"])


# ========================
# align_and_clean_prices
# ========================

def test_align_and_clean_prices_drops_high_missing_assets(sample_prices_with_gaps):
    cleaned, missing_ratio, dropped_assets = align_and_clean_prices(
        sample_prices_with_gaps,
        missing_ratio_threshold=0.30,
    )

    assert "BAD" in dropped_assets
    assert "BAD" not in cleaned.columns
    assert not cleaned.isna().any().any()
    assert missing_ratio["BAD"] > 0.30


def test_align_and_clean_prices_forward_fills_internal_gaps(sample_prices_with_gaps):
    cleaned, _, _ = align_and_clean_prices(
        sample_prices_with_gaps[["SPY", "QQQ"]],
        missing_ratio_threshold=0.50,
    )

    expected_index = pd.bdate_range("2024-01-01", "2024-01-05")
    assert cleaned.index.equals(expected_index)
    assert cleaned.loc[pd.Timestamp("2024-01-03"), "QQQ"] == 200.0


def test_align_and_clean_prices_empty_input_raises():
    with pytest.raises(ValueError, match="Price matrix is empty"):
        align_and_clean_prices(pd.DataFrame())


def test_align_and_clean_prices_all_assets_dropped_raises(sample_prices_with_gaps):
    with pytest.raises(ValueError, match="All assets were dropped"):
        align_and_clean_prices(
            sample_prices_with_gaps,
            missing_ratio_threshold=0.0,
        )


def test_align_and_clean_prices_cleaned_empty_after_ffill_raises():
    prices = pd.DataFrame(
        {"A": [np.nan, np.nan]},
        index=pd.to_datetime(["2024-01-01", "2024-01-02"]),
    )

    with pytest.raises(
        ValueError,
        match="Price matrix is empty after alignment and missing value handling",
    ):
        align_and_clean_prices(prices, missing_ratio_threshold=1.0)


# ========================
# compute_clean_return_matrix
# ========================

def test_compute_clean_return_matrix_removes_nan_and_clips_outliers():
    prices = pd.DataFrame(
        {
            "SPY": [100.0, 101.0, 102.0, 300.0, 303.0],
            "QQQ": [200.0, 201.0, 202.0, 203.0, 204.0],
        },
        index=pd.bdate_range("2024-01-01", periods=5),
    )

    returns = compute_clean_return_matrix(prices, winsorize_limit=0.25)

    assert not returns.isna().any().any()
    assert len(returns) == 4
    assert returns["SPY"].max() < 2.0


def test_compute_clean_return_matrix_empty_input_raises():
    with pytest.raises(ValueError, match="Cleaned price matrix is empty"):
        compute_clean_return_matrix(pd.DataFrame())


def test_compute_clean_return_matrix_returns_empty_after_pct_change_raises():
    prices = pd.DataFrame(
        {"A": [100.0]},
        index=pd.to_datetime(["2024-01-01"]),
    )

    with pytest.raises(ValueError, match="Return matrix is empty after pct_change"):
        compute_clean_return_matrix(prices)


def test_compute_clean_return_matrix_clips_extreme_values():
    prices = pd.DataFrame(
        {
            "A": [100.0, 101.0, 102.0, 5000.0, 5100.0],
        },
        index=pd.bdate_range("2024-01-01", periods=5),
    )

    raw_returns = prices.pct_change().dropna()
    clipped_returns = compute_clean_return_matrix(prices, winsorize_limit=0.2)

    assert not clipped_returns.isna().any().any()
    assert clipped_returns.shape == raw_returns.shape
    assert clipped_returns["A"].max() <= raw_returns["A"].max()
    assert clipped_returns["A"].iloc[2] < raw_returns["A"].iloc[2]


# ========================
# create_rolling_windows
# ========================

def test_create_rolling_windows_returns_expected_number_of_windows(clean_return_matrix):
    windows = create_rolling_windows(
        clean_return_matrix,
        train_window=5,
        test_window=2,
        step_size=2,
    )

    assert len(windows) == 3
    assert len(windows[0].train_data) == 5
    assert len(windows[0].test_data) == 2
    assert windows[0].train_end < windows[0].test_start


def test_create_rolling_windows_raises_when_data_is_too_short(clean_return_matrix):
    with pytest.raises(ValueError):
        create_rolling_windows(
            clean_return_matrix.iloc[:5],
            train_window=5,
            test_window=2,
            step_size=1,
        )


def test_create_rolling_windows_empty_input_raises():
    with pytest.raises(ValueError, match="Clean return matrix is empty"):
        create_rolling_windows(pd.DataFrame())


def test_create_rolling_windows_basic_structure(clean_return_matrix):
    windows = create_rolling_windows(
        clean_return_matrix,
        train_window=5,
        test_window=2,
        step_size=2,
    )

    w = windows[0]
    assert w.train_start < w.train_end
    assert w.test_start < w.test_end
    assert w.train_end < w.test_start


# ========================
# build_data_pipeline
# ========================

@patch("bigdata.data_pipeline.download_adjusted_close_prices")
def test_build_data_pipeline_returns_standard_bundle(mock_download):
    prices = pd.DataFrame(
        {
            "SPY": [100.0, 101.0, 102.0, 103.0, 104.0, 105.0, 106.0, 107.0, 108.0, 109.0],
            "QQQ": [200.0, 201.0, 202.0, 203.0, 204.0, 205.0, 206.0, 207.0, 208.0, 209.0],
        },
        index=pd.bdate_range("2024-01-01", periods=10),
    )
    mock_download.return_value = prices

    result = build_data_pipeline(
        tickers=["SPY", "QQQ"],
        train_window=5,
        test_window=2,
        step_size=2,
    )

    assert isinstance(result, DataPipelineResult)
    assert list(result.clean_return_matrix.columns) == ["SPY", "QQQ"]
    assert len(result.training_test_windows) == 2
    assert result.dropped_assets == []


@patch("bigdata.data_pipeline.download_adjusted_close_prices")
def test_build_data_pipeline_empty_prices_propagates_error(mock_download):
    mock_download.return_value = pd.DataFrame()

    with pytest.raises(ValueError):
        build_data_pipeline(["SPY"])

