import numpy as np
import pandas as pd
import pytest

from bigdata.macro_factor import (
    MacroFactorRiskResult,
    compute_factor_covariance,
    estimate_factor_beta,
    run_macro_factor_risk_decomposition,
)


# ========================
# estimate_factor_beta
# ========================

def test_estimate_factor_beta_recovers_simple_linear_exposure():
    factor_window = pd.DataFrame(
        {
            "SPY": [0.01, 0.02, -0.01, 0.00, 0.03],
            "TLT": [0.00, 0.01, 0.01, 0.02, -0.01],
        }
    )
    portfolio_window = 2.0 * factor_window["SPY"] + 0.5 * factor_window["TLT"]

    beta = estimate_factor_beta(portfolio_window, factor_window)

    assert beta["SPY"] == pytest.approx(2.0, abs=1e-10)
    assert beta["TLT"] == pytest.approx(0.5, abs=1e-10)


def test_estimate_factor_beta_length_mismatch_raises():
    y = pd.Series([0.01, 0.02])
    x = pd.DataFrame({"SPY": [0.01]})

    with pytest.raises(ValueError, match="same length"):
        estimate_factor_beta(y, x)


def test_estimate_factor_beta_too_few_points_raises():
    y = pd.Series([0.01])
    x = pd.DataFrame({"SPY": [0.01]})

    with pytest.raises(ValueError, match="At least two observations"):
        estimate_factor_beta(y, x)


# ========================
# compute_factor_covariance
# ========================

def test_compute_factor_covariance_returns_square_matrix(macro_factor_inputs):
    _, factor_returns = macro_factor_inputs
    covariance = compute_factor_covariance(factor_returns.iloc[:10])

    assert covariance.shape[0] == covariance.shape[1]


def test_compute_factor_covariance_empty_raises():
    with pytest.raises(ValueError, match="empty"):
        compute_factor_covariance(pd.DataFrame())


# ========================
# run_macro_factor_risk_decomposition (核心)
# ========================

def test_run_macro_factor_risk_decomposition_outputs_aligned_series(macro_factor_inputs):
    portfolio_returns, factor_returns = macro_factor_inputs

    result = run_macro_factor_risk_decomposition(
        portfolio_returns=portfolio_returns,
        asset_returns=factor_returns,
        factor_set=["SPY", "TLT", "GLD", "UUP"],
        window_size=12,
    )

    assert isinstance(result, MacroFactorRiskResult)
    assert list(result.rolling_beta.columns) == ["SPY", "TLT", "GLD", "UUP"]
    assert result.rolling_beta.index.equals(result.systematic_risk_series.index)


def test_run_macro_factor_risk_decomposition_shares_sum_to_one(macro_factor_inputs):
    portfolio_returns, factor_returns = macro_factor_inputs

    result = run_macro_factor_risk_decomposition(
        portfolio_returns=portfolio_returns,
        asset_returns=factor_returns,
        factor_set=["SPY", "TLT", "GLD", "UUP"],
        window_size=12,
    )

    shares = result.systematic_share_series + result.idiosyncratic_share_series
    assert np.allclose(shares.to_numpy(), np.ones(len(shares)))


# ========================
# 🔥 关键：_validate_macro_inputs 分支
# ========================

def test_run_macro_factor_empty_portfolio_raises(macro_factor_inputs):
    _, factor_returns = macro_factor_inputs

    with pytest.raises(ValueError, match="portfolio_returns is empty"):
        run_macro_factor_risk_decomposition(
            portfolio_returns=pd.Series(dtype=float),
            asset_returns=factor_returns,
            factor_set=["SPY"],
            window_size=5,
        )


def test_run_macro_factor_empty_asset_returns_raises(macro_factor_inputs):
    portfolio_returns, _ = macro_factor_inputs

    with pytest.raises(ValueError, match="asset_returns is empty"):
        run_macro_factor_risk_decomposition(
            portfolio_returns=portfolio_returns,
            asset_returns=pd.DataFrame(),
            factor_set=["SPY"],
            window_size=5,
        )


def test_run_macro_factor_empty_factor_set_raises(macro_factor_inputs):
    portfolio_returns, factor_returns = macro_factor_inputs

    with pytest.raises(ValueError, match="cannot be empty"):
        run_macro_factor_risk_decomposition(
            portfolio_returns,
            factor_returns,
            factor_set=[],
            window_size=5,
        )


def test_run_macro_factor_invalid_window_size_raises(macro_factor_inputs):
    portfolio_returns, factor_returns = macro_factor_inputs

    with pytest.raises(ValueError, match="at least 2"):
        run_macro_factor_risk_decomposition(
            portfolio_returns,
            factor_returns,
            factor_set=["SPY"],
            window_size=1,
        )


def test_run_macro_factor_missing_factor_raises(macro_factor_inputs):
    portfolio_returns, factor_returns = macro_factor_inputs

    with pytest.raises(ValueError, match="not found"):
        run_macro_factor_risk_decomposition(
            portfolio_returns,
            factor_returns,
            factor_set=["FAKE"],
            window_size=5,
        )


def test_run_macro_factor_duplicate_factors_raises(macro_factor_inputs):
    portfolio_returns, factor_returns = macro_factor_inputs

    with pytest.raises(ValueError, match="duplicates"):
        run_macro_factor_risk_decomposition(
            portfolio_returns,
            factor_returns,
            factor_set=["SPY", "SPY"],
            window_size=5,
        )


def test_run_macro_factor_not_enough_data_raises(macro_factor_inputs):
    portfolio_returns, factor_returns = macro_factor_inputs

    with pytest.raises(ValueError, match="Not enough aligned data"):
        run_macro_factor_risk_decomposition(
            portfolio_returns.iloc[:5],
            factor_returns.iloc[:5],
            factor_set=["SPY"],
            window_size=5,
        )


def test_run_macro_factor_handles_zero_total_risk_case():
    dates = pd.bdate_range("2024-01-01", periods=10)

    portfolio_returns = pd.Series([0.0] * 10, index=dates)
    factor_returns = pd.DataFrame(
        {"SPY": [0.0] * 10},
        index=dates,
    )

    result = run_macro_factor_risk_decomposition(
        portfolio_returns,
        factor_returns,
        factor_set=["SPY"],
        window_size=5,
    )

    assert np.all(result.systematic_share_series == 0.0)
    assert np.all(result.idiosyncratic_share_series == 0.0)
