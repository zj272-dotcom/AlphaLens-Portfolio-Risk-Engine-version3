import numpy as np
import pandas as pd
import pytest

from bigdata.macro_factor import (
    MacroFactorRiskResult,
    compute_factor_covariance,
    estimate_factor_beta,
    run_macro_factor_risk_decomposition,
)


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


def test_compute_factor_covariance_returns_square_matrix(macro_factor_inputs):
    _, factor_returns = macro_factor_inputs
    covariance = compute_factor_covariance(factor_returns.iloc[:10])

    assert covariance.shape == (4, 4)
    assert list(covariance.index) == ["SPY", "TLT", "GLD", "UUP"]


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
    assert result.rolling_beta.index.equals(result.idiosyncratic_risk_series.index)
    assert result.rolling_beta.index.equals(result.systematic_share_series.index)
    assert result.rolling_beta.index.equals(result.idiosyncratic_share_series.index)
    assert result.rolling_beta.index.equals(result.factor_risk_contribution_by_date.index)


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


def test_run_macro_factor_risk_decomposition_uses_only_past_window(macro_factor_inputs):
    portfolio_returns, factor_returns = macro_factor_inputs
    shocked_portfolio = portfolio_returns.copy()
    shocked_portfolio.iloc[-1] = shocked_portfolio.iloc[-1] + 10.0

    base = run_macro_factor_risk_decomposition(
        portfolio_returns=portfolio_returns,
        asset_returns=factor_returns,
        factor_set=["SPY", "TLT", "GLD", "UUP"],
        window_size=12,
    )
    shocked = run_macro_factor_risk_decomposition(
        portfolio_returns=shocked_portfolio,
        asset_returns=factor_returns,
        factor_set=["SPY", "TLT", "GLD", "UUP"],
        window_size=12,
    )

    pd.testing.assert_series_equal(base.rolling_beta.iloc[0], shocked.rolling_beta.iloc[0])
