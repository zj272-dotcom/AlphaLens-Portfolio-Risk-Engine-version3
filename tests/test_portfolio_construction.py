import numpy as np
import pandas as pd
import pytest

from bigdata.data_pipeline import RollingWindow
from bigdata.portfolio_construction import (
    PortfolioConstructionResult,
    compute_equal_weight,
    compute_min_variance_weights,
    compute_turnover,
    estimate_covariance_matrix,
    run_rolling_portfolio_construction,
    shrink_covariance,
)


def test_estimate_covariance_matrix_returns_square_matrix(clean_return_matrix):
    covariance = estimate_covariance_matrix(clean_return_matrix)

    assert covariance.shape == (2, 2)
    assert list(covariance.index) == ["SPY", "QQQ"]
    assert list(covariance.columns) == ["SPY", "QQQ"]


def test_estimate_covariance_matrix_empty_raises():
    empty_df = pd.DataFrame()

    with pytest.raises(ValueError, match="Training return matrix is empty"):
        estimate_covariance_matrix(empty_df)


def test_estimate_covariance_matrix_with_nan_raises():
    df = pd.DataFrame(
        {
            "A": [0.01, np.nan, 0.03],
            "B": [0.02, 0.01, 0.00],
        }
    )

    with pytest.raises(ValueError, match="contains NaN values"):
        estimate_covariance_matrix(df)


def test_estimate_covariance_matrix_too_few_observations_raises():
    df = pd.DataFrame(
        {
            "A": [0.01],
            "B": [0.02],
        }
    )

    with pytest.raises(ValueError, match="At least two observations"):
        estimate_covariance_matrix(df)


def test_shrink_covariance_enforces_positive_eigenvalues():
    sample = pd.DataFrame(
        [[0.04, 0.05], [0.05, 0.04]],
        index=["A", "B"],
        columns=["A", "B"],
    )

    shrunk = shrink_covariance(sample, shrinkage_intensity=0.5, target="diagonal")
    eigenvalues = np.linalg.eigvalsh(shrunk.to_numpy())

    assert np.all(eigenvalues > 0)


def test_shrink_covariance_identity_target_returns_square_matrix():
    sample = pd.DataFrame(
        [[0.04, 0.01], [0.01, 0.09]],
        index=["A", "B"],
        columns=["A", "B"],
    )

    shrunk = shrink_covariance(sample, shrinkage_intensity=0.3, target="identity")

    assert shrunk.shape == (2, 2)
    assert list(shrunk.index) == ["A", "B"]
    assert list(shrunk.columns) == ["A", "B"]


def test_shrink_covariance_empty_raises():
    empty_cov = pd.DataFrame()

    with pytest.raises(ValueError, match="Sample covariance matrix is empty"):
        shrink_covariance(empty_cov)


def test_shrink_covariance_non_square_raises():
    nonsquare_cov = pd.DataFrame([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]])

    with pytest.raises(ValueError, match="must be square"):
        shrink_covariance(nonsquare_cov)


def test_shrink_covariance_invalid_intensity_raises():
    sample = pd.DataFrame([[0.1, 0.0], [0.0, 0.1]])

    with pytest.raises(ValueError, match="shrinkage_intensity must be between 0 and 1"):
        shrink_covariance(sample, shrinkage_intensity=1.5)


def test_shrink_covariance_invalid_target_raises():
    sample = pd.DataFrame([[0.1, 0.0], [0.0, 0.1]])

    with pytest.raises(ValueError, match="target must be either"):
        shrink_covariance(sample, target="unknown")


def test_compute_equal_weight_sums_to_one():
    weights = compute_equal_weight(["SPY", "QQQ", "TLT"])

    assert pytest.approx(weights.sum()) == 1.0
    assert np.allclose(weights.to_numpy(), np.repeat(1 / 3, 3))


def test_compute_equal_weight_empty_raises():
    with pytest.raises(ValueError, match="asset_names cannot be empty"):
        compute_equal_weight([])


def test_compute_min_variance_weights_respects_constraints():
    covariance = pd.DataFrame(
        [
            [0.04, 0.01, 0.00],
            [0.01, 0.09, 0.01],
            [0.00, 0.01, 0.02],
        ],
        index=["SPY", "QQQ", "TLT"],
        columns=["SPY", "QQQ", "TLT"],
    )

    weights = compute_min_variance_weights(covariance, max_weight=0.6)

    assert pytest.approx(weights.sum()) == 1.0
    assert (weights >= 0).all()
    assert (weights <= 0.6 + 1e-10).all()


def test_compute_min_variance_weights_empty_raises():
    empty_cov = pd.DataFrame()

    with pytest.raises(ValueError, match="Covariance matrix is empty"):
        compute_min_variance_weights(empty_cov)


def test_compute_min_variance_weights_non_square_raises():
    nonsquare_cov = pd.DataFrame([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]])

    with pytest.raises(ValueError, match="must be square"):
        compute_min_variance_weights(nonsquare_cov)


def test_compute_min_variance_weights_invalid_max_weight_raises():
    cov = pd.DataFrame([[0.1, 0.0], [0.0, 0.1]])

    with pytest.raises(ValueError, match="max_weight must be in the interval"):
        compute_min_variance_weights(cov, max_weight=1.5)


def test_compute_min_variance_weights_too_small_max_weight_raises():
    cov = pd.DataFrame([[0.1, 0.0], [0.0, 0.1]])

    with pytest.raises(ValueError, match="too small to allow a fully invested portfolio"):
        compute_min_variance_weights(cov, max_weight=0.4)


def test_compute_turnover_uses_absolute_weight_changes():
    previous = pd.Series({"SPY": 0.5, "QQQ": 0.5})
    current = pd.Series({"SPY": 0.3, "QQQ": 0.7})

    turnover = compute_turnover(previous, current)

    assert pytest.approx(turnover) == 0.4


def test_compute_turnover_none_previous_is_zero():
    current = pd.Series({"SPY": 0.5, "QQQ": 0.5})

    turnover = compute_turnover(None, current)

    assert turnover == 0.0


def test_compute_turnover_missing_previous_asset_treated_as_zero():
    previous = pd.Series({"SPY": 1.0})
    current = pd.Series({"SPY": 0.6, "QQQ": 0.4})

    turnover = compute_turnover(previous, current)

    assert pytest.approx(turnover) == 0.8


def test_run_rolling_portfolio_construction_min_variance_returns_consistent_outputs(
    rolling_windows_for_portfolios,
):
    result = run_rolling_portfolio_construction(
        rolling_windows_for_portfolios,
        strategy="min_variance",
        shrinkage_intensity=0.1,
        max_weight=0.6,
    )

    assert isinstance(result, PortfolioConstructionResult)
    assert result.rolling_weights.shape == (2, 3)
    assert len(result.portfolio_returns) == 4
    assert len(result.turnover_series) == 2
    assert pytest.approx(result.rolling_weights.iloc[0].sum()) == 1.0
    assert pytest.approx(result.rolling_weights.iloc[1].sum()) == 1.0
    assert result.portfolio_returns.index.is_monotonic_increasing
    assert len(result.covariance_by_rebalance) == 2


def test_run_rolling_portfolio_construction_equal_weight_has_zero_initial_turnover(
    rolling_windows_for_portfolios,
):
    result = run_rolling_portfolio_construction(
        rolling_windows_for_portfolios,
        strategy="equal_weight",
    )

    assert result.turnover_series.iloc[0] == 0.0
    assert np.allclose(result.rolling_weights.iloc[0].to_numpy(), np.repeat(1 / 3, 3))


def test_run_rolling_portfolio_construction_empty_windows_raises():
    with pytest.raises(ValueError, match="windows cannot be empty"):
        run_rolling_portfolio_construction([])


def test_run_rolling_portfolio_construction_invalid_strategy_raises(
    rolling_windows_for_portfolios,
):
    with pytest.raises(ValueError, match="strategy must be either"):
        run_rolling_portfolio_construction(
            rolling_windows_for_portfolios,
            strategy="random_strategy",
        )


def test_run_rolling_portfolio_construction_rejects_overlapping_test_windows():
    dates = pd.bdate_range("2024-01-01", periods=8)
    returns = pd.DataFrame(
        {
            "SPY": np.linspace(0.001, 0.008, 8),
            "QQQ": np.linspace(0.002, 0.009, 8),
        },
        index=dates,
    )
    windows = [
        RollingWindow(
            train_start=dates[0],
            train_end=dates[3],
            test_start=dates[4],
            test_end=dates[5],
            train_data=returns.iloc[0:4],
            test_data=returns.iloc[4:6],
        ),
        RollingWindow(
            train_start=dates[1],
            train_end=dates[4],
            test_start=dates[5],
            test_end=dates[6],
            train_data=returns.iloc[1:5],
            test_data=returns.iloc[5:7],
        ),
    ]

    with pytest.raises(ValueError, match="overlap"):
        run_rolling_portfolio_construction(windows, strategy="equal_weight")


def test_run_rolling_portfolio_construction_duplicate_test_dates_raise():
    dates = pd.bdate_range("2024-01-01", periods=10)
    returns = pd.DataFrame(
        {
            "SPY": np.linspace(0.001, 0.010, 10),
            "QQQ": np.linspace(0.002, 0.011, 10),
            "TLT": np.linspace(0.000, 0.009, 10),
        },
        index=dates,
    )

    windows = [
        RollingWindow(
            train_start=dates[0],
            train_end=dates[3],
            test_start=dates[4],
            test_end=dates[5],
            train_data=returns.iloc[0:4],
            test_data=returns.iloc[4:6],
        ),
        RollingWindow(
            train_start=dates[1],
            train_end=dates[4],
            test_start=dates[6],
            test_end=dates[7],
            train_data=returns.iloc[1:5],
            test_data=returns.loc[[dates[5], dates[6]]],
        ),
    ]

    with pytest.raises(ValueError, match="duplicate dates"):
        run_rolling_portfolio_construction(windows, strategy="equal_weight")
