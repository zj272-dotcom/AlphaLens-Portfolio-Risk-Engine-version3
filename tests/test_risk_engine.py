import numpy as np
import pandas as pd
import pytest

from bigdata.risk_engine import (
    RiskEngineResult,
    build_pnl_distribution,
    historical_var_es,
    monte_carlo_var_es,
    parametric_var_es,
    run_rolling_risk_engine,
)


def test_historical_var_es_returns_positive_loss_numbers():
    window = pd.Series([-0.04, -0.02, 0.00, 0.01, 0.02])

    var, es = historical_var_es(window, confidence_level=0.80)

    assert var >= 0
    assert es >= var


def test_historical_var_es_empty_raises():
    with pytest.raises(ValueError, match="portfolio_returns"):
        historical_var_es(pd.Series(dtype=float))


def test_historical_var_es_drops_nan_and_still_runs():
    window = pd.Series([-0.04, np.nan, -0.02, 0.01, 0.02])

    var, es = historical_var_es(window, confidence_level=0.80)

    assert var >= 0
    assert es >= var


def test_parametric_var_es_returns_es_greater_than_var():
    window = pd.Series([-0.02, -0.01, 0.00, 0.01, 0.02, 0.03])

    var, es = parametric_var_es(window, confidence_level=0.95)

    assert es >= var


def test_parametric_var_es_too_few_points_raises():
    window = pd.Series([0.01])

    with pytest.raises(ValueError, match="At least two observations"):
        parametric_var_es(window)


def test_monte_carlo_var_es_is_reproducible():
    window = pd.Series(np.linspace(-0.02, 0.03, 20))

    result_1 = monte_carlo_var_es(window, random_state=7, n_simulations=2000)
    result_2 = monte_carlo_var_es(window, random_state=7, n_simulations=2000)

    assert result_1[0] == pytest.approx(result_2[0])
    assert result_1[1] == pytest.approx(result_2[1])
    assert np.allclose(result_1[2], result_2[2])


def test_monte_carlo_var_es_invalid_simulations_raises():
    window = pd.Series(np.linspace(-0.02, 0.03, 20))

    with pytest.raises(ValueError, match="n_simulations"):
        monte_carlo_var_es(window, n_simulations=0)


def test_monte_carlo_var_es_too_few_points_raises():
    window = pd.Series([0.01])

    with pytest.raises(ValueError, match="At least two observations"):
        monte_carlo_var_es(window)


def test_monte_carlo_var_es_returns_sample_of_requested_size():
    window = pd.Series(np.linspace(-0.02, 0.03, 20))

    var, es, sample = monte_carlo_var_es(window, random_state=3, n_simulations=1234)

    assert len(sample) == 1234
    assert es >= var


def test_build_pnl_distribution_returns_plot_ready_dataframe():
    window = pd.Series(np.linspace(-0.03, 0.02, 25))
    monte_carlo_sample = np.linspace(-0.04, 0.03, 100)

    distribution = build_pnl_distribution(window, monte_carlo_sample, random_state=5)

    assert list(distribution.columns) == ["historical", "parametric", "monte_carlo"]
    assert distribution.shape == (100, 3)


def test_build_pnl_distribution_empty_window_raises():
    with pytest.raises(ValueError, match="portfolio_returns"):
        build_pnl_distribution(pd.Series(dtype=float), np.array([1, 2, 3]))


def test_build_pnl_distribution_uses_monte_carlo_sample_length():
    window = pd.Series(np.linspace(-0.03, 0.02, 25))
    monte_carlo_sample = np.linspace(-0.04, 0.03, 57)

    distribution = build_pnl_distribution(window, monte_carlo_sample, random_state=1)

    assert distribution.shape == (57, 3)


def test_run_rolling_risk_engine_outputs_aligned_frames():
    portfolio_returns = pd.Series(
        [
            -0.02,
            0.01,
            0.00,
            0.02,
            -0.01,
            0.01,
            -0.03,
            0.02,
            0.01,
            -0.01,
        ],
        index=pd.bdate_range("2024-01-01", periods=10),
        name="portfolio_return",
    )

    result = run_rolling_risk_engine(
        portfolio_returns,
        window_size=5,
        confidence_level=0.95,
        n_simulations=1000,
        random_state=11,
    )

    assert isinstance(result, RiskEngineResult)
    assert list(result.var_series.columns) == ["historical", "parametric", "monte_carlo"]
    assert list(result.es_series.columns) == ["historical", "parametric", "monte_carlo"]
    assert result.var_series.index.equals(result.es_series.index)
    assert result.var_series.index.equals(result.rolling_moments.index)
    assert len(result.var_series) == 5
    assert result.distribution_date == result.var_series.index[-1]
    assert result.pnl_distribution.shape == (1000, 3)


def test_run_rolling_risk_engine_uses_only_past_data():
    portfolio_returns = pd.Series(
        [-0.10, -0.02, 0.01, 0.03, 0.02, -0.50],
        index=pd.bdate_range("2024-01-01", periods=6),
        name="portfolio_return",
    )

    result = run_rolling_risk_engine(
        portfolio_returns,
        window_size=5,
        confidence_level=0.80,
        n_simulations=500,
        random_state=3,
    )

    expected_var, _ = historical_var_es(portfolio_returns.iloc[:5], confidence_level=0.80)
    realized_first_var = float(result.var_series.iloc[0]["historical"])

    assert realized_first_var == pytest.approx(expected_var)


def test_run_rolling_risk_engine_empty_returns_raises():
    with pytest.raises(ValueError, match="portfolio_returns"):
        run_rolling_risk_engine(pd.Series(dtype=float))


def test_run_rolling_risk_engine_window_too_small_raises():
    returns = pd.Series(np.random.randn(10))

    with pytest.raises(ValueError, match="window_size"):
        run_rolling_risk_engine(returns, window_size=1)


def test_run_rolling_risk_engine_not_enough_data_raises():
    returns = pd.Series(np.random.randn(5))

    with pytest.raises(ValueError, match="Not enough data"):
        run_rolling_risk_engine(returns, window_size=5)


def test_run_rolling_risk_engine_invalid_confidence_level_raises():
    returns = pd.Series(np.random.randn(10))

    with pytest.raises(ValueError):
        run_rolling_risk_engine(returns, confidence_level=1.5)


def test_run_rolling_risk_engine_handles_nan_input():
    returns = pd.Series([0.01, np.nan, 0.02, 0.01, 0.00, 0.01])

    result = run_rolling_risk_engine(returns, window_size=3)

    assert isinstance(result, RiskEngineResult)


def test_run_rolling_risk_engine_sets_default_series_name_if_missing():
    returns = pd.Series(
        np.linspace(-0.02, 0.02, 8),
        index=pd.bdate_range("2024-01-01", periods=8),
    )

    result = run_rolling_risk_engine(returns, window_size=3, n_simulations=200, random_state=4)

    assert isinstance(result, RiskEngineResult)
    assert len(result.var_series) == 5


def test_run_rolling_risk_engine_distribution_date_matches_last_index():
    returns = pd.Series(
        np.linspace(-0.02, 0.02, 10),
        index=pd.bdate_range("2024-01-01", periods=10),
        name="portfolio_return",
    )

    result = run_rolling_risk_engine(
        returns,
        window_size=4,
        confidence_level=0.95,
        n_simulations=250,
        random_state=9,
    )

    assert result.distribution_date == result.var_series.index.max()
    assert result.distribution_date == result.es_series.index.max()
    assert result.distribution_date == result.rolling_moments.index.max()
