import pandas as pd
import pytest

from bigdata.stress_testing import (
    StressTestResult,
    aggregate_period_returns,
    calculate_stress_loss,
    find_worst_historical_periods,
    run_historical_stress_scenarios,
)


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


def test_calculate_stress_loss_returns_positive_loss_and_contributions():
    weights = pd.Series({"SPY": 0.6, "TLT": 0.4})
    scenario_returns = pd.Series({"SPY": -0.10, "TLT": 0.02})

    scenario_return, loss, contributions = calculate_stress_loss(weights, scenario_returns)

    assert scenario_return == pytest.approx(-0.052)
    assert loss == pytest.approx(0.052)
    assert contributions["SPY"] == pytest.approx(0.06)
    assert contributions["TLT"] == pytest.approx(-0.008)


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
    assert list(result.scenario_loss_table.columns) == ["Start Date", "End Date", "Scenario Return", "Loss"]
    assert "Crisis A" in result.stress_contribution_table.index
    assert "SPY" in result.stress_contribution_table.columns
    assert result.stress_comparison.iloc[0]["Loss"] >= result.stress_comparison.iloc[-1]["Loss"]


def test_find_worst_historical_periods_finds_day_week_month(stress_asset_returns, current_weights):
    result = find_worst_historical_periods(
        stress_asset_returns,
        current_weights,
        windows={"Worst Day": 1, "Worst Week": 5, "Worst Month": 7},
    )

    assert isinstance(result, StressTestResult)
    assert {"Worst Day", "Worst Week", "Worst Month"} == set(result.scenario_loss_table.index)
    assert "Start Date" in result.scenario_loss_table.columns
    assert "End Date" in result.scenario_loss_table.columns
    assert "Scenario Return" in result.scenario_loss_table.columns
    assert all(result.scenario_loss_table["Loss"] >= 0)


def test_run_historical_stress_scenarios_rejects_missing_weight_sum(stress_asset_returns):
    bad_weights = pd.Series({"SPY": 0.8, "QQQ": 0.3})

    with pytest.raises(ValueError):
        run_historical_stress_scenarios(stress_asset_returns, bad_weights)
