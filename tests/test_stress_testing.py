import numpy as np
import pandas as pd
import pytest

from bigdata.stress_testing import (
    StressTestResult,
    aggregate_period_returns,
    calculate_stress_loss,
    find_worst_historical_periods,
    run_historical_stress_scenarios,
)


# ========================
# aggregate_period_returns
# ========================

def test_aggregate_period_returns_uses_compounding():
    returns = pd.DataFrame(
        {
            "SPY": [-0.10, 0.05],
            "TLT": [0.02, 0.01],
        },
        index=pd.bdate_range("2024-01-01", periods=2),
    )

    aggregated = aggregate_period_returns(returns)

    assert aggregated["SPY"] == pytest.approx((1 - 0.10) * (1 + 0.05) - 1)
    assert aggregated["TLT"] == pytest.approx((1 + 0.02) * (1 + 0.01) - 1)


def test_aggregate_period_returns_empty_raises():
    with pytest.raises(ValueError, match="empty"):
        aggregate_period_returns(pd.DataFrame())


# ========================
# calculate_stress_loss
# ========================

def test_calculate_stress_loss_returns_positive_loss_and_contributions():
    weights = pd.Series({"SPY": 0.6, "TLT": 0.4})
    scenario_returns = pd.Series({"SPY": -0.10, "TLT": 0.02})

    scenario_return, loss, contributions = calculate_stress_loss(weights, scenario_returns)

    assert scenario_return == pytest.approx(-0.052)
    assert loss == pytest.approx(0.052)
    assert contributions["SPY"] == pytest.approx(0.06)
    assert contributions["TLT"] == pytest.approx(-0.008)


def test_calculate_stress_loss_missing_asset_raises():
    weights = pd.Series({"SPY": 0.6, "TLT": 0.4})
    scenario_returns = pd.Series({"SPY": -0.10})  # 缺 TLT

    with pytest.raises(ValueError, match="missing required assets"):
        calculate_stress_loss(weights, scenario_returns)


# ========================
# run_historical_stress_scenarios
# ========================

def test_run_historical_stress_scenarios_builds_tables(stress_asset_returns, current_weights):
    scenarios = {
        "Crisis A": ("2020-02-03", "2020-02-05"),
        "Crisis B": ("2020-03-02", "2020-03-04"),
    }

    result = run_historical_stress_scenarios(
        stress_asset_returns,
        current_weights,
        scenarios=scenarios,
    )

    assert isinstance(result, StressTestResult)
    assert "Loss" in result.scenario_loss_table.columns
    assert "SPY" in result.stress_contribution_table.columns
    assert result.stress_comparison.iloc[0]["Loss"] >= result.stress_comparison.iloc[-1]["Loss"]


def test_run_historical_stress_scenarios_empty_returns_raises(current_weights):
    with pytest.raises(ValueError, match="empty"):
        run_historical_stress_scenarios(pd.DataFrame(), current_weights)


def test_run_historical_stress_scenarios_empty_weights_raises(stress_asset_returns):
    with pytest.raises(ValueError, match="empty"):
        run_historical_stress_scenarios(stress_asset_returns, pd.Series())


def test_run_historical_stress_scenarios_no_overlap_assets_raises(stress_asset_returns):
    bad_weights = pd.Series({"ABC": 1.0})

    with pytest.raises(ValueError, match="No overlapping assets"):
        run_historical_stress_scenarios(stress_asset_returns, bad_weights)


def test_run_historical_stress_scenarios_weights_not_sum_to_one_raises(stress_asset_returns):
    bad_weights = pd.Series({"SPY": 0.8, "QQQ": 0.3})

    with pytest.raises(ValueError, match="must sum to 1"):
        run_historical_stress_scenarios(stress_asset_returns, bad_weights)


def test_run_historical_stress_scenarios_nan_in_returns_raises(stress_asset_returns, current_weights):
    bad_returns = stress_asset_returns.copy()
    bad_returns.iloc[0, 0] = np.nan

    with pytest.raises(ValueError, match="contains NaN"):
        run_historical_stress_scenarios(bad_returns, current_weights)


def test_run_historical_stress_scenarios_no_matching_scenarios_raises(stress_asset_returns, current_weights):
    scenarios = {
        "Future Crisis": ("2100-01-01", "2100-01-10")
    }

    with pytest.raises(ValueError, match="No stress scenarios"):
        run_historical_stress_scenarios(
            stress_asset_returns,
            current_weights,
            scenarios=scenarios,
        )


# ========================
# find_worst_historical_periods
# ========================

def test_find_worst_historical_periods_basic(stress_asset_returns, current_weights):
    result = find_worst_historical_periods(
        stress_asset_returns,
        current_weights,
        windows={"Worst Day": 1, "Worst Week": 5},
    )

    assert isinstance(result, StressTestResult)
    assert "Worst Day" in result.scenario_loss_table.index
    assert all(result.scenario_loss_table["Loss"] >= 0)


def test_find_worst_historical_periods_invalid_window_raises(stress_asset_returns, current_weights):
    with pytest.raises(ValueError, match="must be positive"):
        find_worst_historical_periods(
            stress_asset_returns,
            current_weights,
            windows={"Bad": 0},
        )


def test_find_worst_historical_periods_not_enough_data_raises(stress_asset_returns, current_weights):
    with pytest.raises(ValueError, match="Not enough data"):
        find_worst_historical_periods(
            stress_asset_returns.iloc[:2],
            current_weights,
            windows={"Month": 10},
        )
