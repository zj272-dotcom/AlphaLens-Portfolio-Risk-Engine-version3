import pandas as pd
import pytest

from bigdata.data_pipeline import RollingWindow
from bigdata.macro_factor import run_macro_factor_risk_decomposition
from bigdata.portfolio_construction import run_rolling_portfolio_construction
from bigdata.risk_engine import run_rolling_risk_engine
from bigdata.stress_testing import find_worst_historical_periods


@pytest.fixture
def sample_prices_with_gaps():
    index = pd.to_datetime(
        [
            "2024-01-01",
            "2024-01-02",
            "2024-01-04",
            "2024-01-05",
        ]
    )
    return pd.DataFrame(
        {
            "SPY": [100.0, 101.0, 103.0, 104.0],
            "QQQ": [200.0, None, 202.0, 203.0],
            "BAD": [None, None, 50.0, 51.0],
        },
        index=index,
    )


@pytest.fixture
def clean_return_matrix():
    index = pd.bdate_range("2024-01-01", periods=12)
    return pd.DataFrame(
        {
            "SPY": [0.01, 0.00, -0.01, 0.02, 0.01, 0.00, 0.01, -0.02, 0.01, 0.00, 0.01, 0.02],
            "QQQ": [0.02, -0.01, 0.00, 0.01, 0.03, -0.01, 0.00, 0.01, 0.02, -0.02, 0.01, 0.00],
        },
        index=index,
    )


@pytest.fixture
def rolling_windows_for_portfolios():
    index = pd.bdate_range("2024-01-01", periods=10)
    returns = pd.DataFrame(
        {
            "SPY": [0.01, 0.00, 0.02, -0.01, 0.01, 0.00, 0.01, 0.02, -0.01, 0.01],
            "QQQ": [0.015, -0.005, 0.01, 0.00, 0.02, -0.01, 0.01, 0.015, -0.005, 0.00],
            "TLT": [0.002, 0.003, 0.001, 0.004, 0.002, 0.003, 0.001, 0.002, 0.003, 0.002],
        },
        index=index,
    )

    return [
        RollingWindow(
            train_start=index[0],
            train_end=index[3],
            test_start=index[4],
            test_end=index[5],
            train_data=returns.iloc[0:4],
            test_data=returns.iloc[4:6],
        ),
        RollingWindow(
            train_start=index[2],
            train_end=index[5],
            test_start=index[6],
            test_end=index[7],
            train_data=returns.iloc[2:6],
            test_data=returns.iloc[6:8],
        ),
    ]


@pytest.fixture
def stress_asset_returns():
    index = pd.bdate_range("2020-02-03", periods=25)
    return pd.DataFrame(
        {
            "SPY": [
                -0.02, -0.03, -0.01, 0.01, 0.00,
                -0.04, -0.05, -0.03, 0.02, 0.01,
                -0.06, -0.04, -0.02, 0.01, 0.00,
                -0.03, -0.02, 0.01, -0.01, 0.00,
                -0.02, -0.01, 0.00, 0.01, 0.00,
            ],
            "QQQ": [
                -0.03, -0.04, -0.02, 0.01, 0.00,
                -0.05, -0.06, -0.04, 0.02, 0.01,
                -0.07, -0.05, -0.03, 0.01, 0.00,
                -0.04, -0.03, 0.01, -0.01, 0.00,
                -0.02, -0.01, 0.00, 0.01, 0.00,
            ],
            "TLT": [
                0.01, 0.00, 0.01, 0.00, 0.00,
                0.02, 0.03, 0.01, -0.01, 0.00,
                0.02, 0.01, 0.01, 0.00, 0.00,
                0.01, 0.00, 0.00, 0.00, 0.00,
                0.01, 0.00, 0.00, 0.00, 0.00,
            ],
        },
        index=index,
    )


@pytest.fixture
def current_weights():
    return pd.Series({"SPY": 0.4, "QQQ": 0.3, "TLT": 0.3})


@pytest.fixture
def macro_factor_inputs():
    index = pd.bdate_range("2024-01-01", periods=20)
    factor_returns = pd.DataFrame(
        {
            "SPY": [0.010, 0.002, -0.004, 0.006, 0.003, 0.004, -0.002, 0.005, 0.001, -0.003,
                    0.004, 0.002, -0.001, 0.003, 0.005, -0.004, 0.006, 0.002, -0.002, 0.004],
            "TLT": [0.001, 0.003, 0.002, -0.001, 0.000, 0.002, 0.003, -0.002, 0.001, 0.002,
                    0.001, -0.001, 0.002, 0.003, 0.001, 0.002, -0.002, 0.001, 0.002, 0.001],
            "GLD": [0.002, 0.001, 0.000, 0.003, -0.001, 0.002, 0.001, 0.000, 0.002, 0.001,
                    0.003, -0.001, 0.002, 0.001, 0.000, 0.002, 0.001, 0.003, -0.001, 0.002],
            "UUP": [0.001, 0.000, 0.002, 0.001, 0.003, -0.001, 0.002, 0.001, 0.000, 0.002,
                    0.001, 0.003, -0.001, 0.002, 0.001, 0.000, 0.002, 0.001, 0.003, -0.001],
        },
        index=index,
    )
    residual = pd.Series(
        [0.0005, -0.0004, 0.0003, -0.0002, 0.0001, 0.0002, -0.0001, 0.0003, -0.0002, 0.0001,
         0.0002, -0.0003, 0.0001, -0.0002, 0.0004, -0.0001, 0.0002, -0.0002, 0.0001, 0.0003],
        index=index,
    )
    portfolio_returns = (
        0.6 * factor_returns["SPY"]
        + 0.2 * factor_returns["TLT"]
        + 0.1 * factor_returns["GLD"]
        - 0.1 * factor_returns["UUP"]
        + residual
    )
    portfolio_returns.name = "portfolio_return"
    return portfolio_returns, factor_returns


@pytest.fixture
def reporting_inputs(clean_return_matrix):
    portfolio_windows = [
        RollingWindow(
            train_start=clean_return_matrix.index[0],
            train_end=clean_return_matrix.index[5],
            test_start=clean_return_matrix.index[6],
            test_end=clean_return_matrix.index[7],
            train_data=clean_return_matrix.iloc[0:6],
            test_data=clean_return_matrix.iloc[6:8],
        ),
        RollingWindow(
            train_start=clean_return_matrix.index[2],
            train_end=clean_return_matrix.index[7],
            test_start=clean_return_matrix.index[8],
            test_end=clean_return_matrix.index[9],
            train_data=clean_return_matrix.iloc[2:8],
            test_data=clean_return_matrix.iloc[8:10],
        ),
    ]

    portfolio_result = run_rolling_portfolio_construction(
        portfolio_windows,
        strategy="equal_weight",
    )
    risk_result = run_rolling_risk_engine(
        portfolio_returns=portfolio_result.portfolio_returns,
        window_size=2,
        confidence_level=0.95,
        n_simulations=500,
        random_state=7,
    )
    stress_result = find_worst_historical_periods(
        asset_returns=clean_return_matrix,
        current_weights=portfolio_result.rolling_weights.iloc[-1],
        windows={"Worst Day": 1, "Worst Week": 3},
    )
    macro_result = run_macro_factor_risk_decomposition(
        portfolio_returns=portfolio_result.portfolio_returns,
        asset_returns=clean_return_matrix,
        factor_set=["SPY", "QQQ"],
        window_size=2,
    )
    return portfolio_result, risk_result, stress_result, macro_result
