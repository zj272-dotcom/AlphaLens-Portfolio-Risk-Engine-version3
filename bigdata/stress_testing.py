"""Historical stress testing utilities for portfolio loss analysis."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import pandas as pd


DEFAULT_HISTORICAL_SCENARIOS: Dict[str, tuple[str, str]] = {
    "GFC": ("2008-09-01", "2009-03-31"),
    "COVID": ("2020-02-20", "2020-03-23"),
}


@dataclass(frozen=True)
class StressTestResult:
    """Structured output for historical stress testing."""

    scenario_loss_table: pd.DataFrame
    stress_contribution_table: pd.DataFrame
    stress_comparison: pd.DataFrame
    scenario_asset_returns: pd.DataFrame


def _validate_inputs(
    asset_returns: pd.DataFrame,
    current_weights: pd.Series,
) -> tuple[pd.DataFrame, pd.Series]:
    """Validate and align return and weight inputs."""

    if asset_returns.empty:
        raise ValueError("asset_returns is empty.")
    if current_weights.empty:
        raise ValueError("current_weights is empty.")

    returns = asset_returns.copy()
    returns.index = pd.to_datetime(returns.index)
    returns = returns.sort_index()

    weights = current_weights.astype(float)
    common_assets = [asset for asset in returns.columns if asset in weights.index]
    if not common_assets:
        raise ValueError("No overlapping assets between returns and weights.")

    returns = returns[common_assets]
    weights = weights.reindex(common_assets)

    if returns.isna().any().any():
        raise ValueError("asset_returns contains NaN values.")
    if weights.isna().any():
        raise ValueError("current_weights contains NaN values after alignment.")
    if abs(float(weights.sum()) - 1.0) > 1e-6:
        raise ValueError("current_weights must sum to 1.")

    return returns, weights


def aggregate_period_returns(period_returns: pd.DataFrame) -> pd.Series:
    """Compound daily returns into a single multi-period asset return vector."""

    if period_returns.empty:
        raise ValueError("period_returns is empty.")

    compounded = (1.0 + period_returns).prod(axis=0) - 1.0
    return compounded.astype(float)


def calculate_stress_loss(
    current_weights: pd.Series,
    scenario_asset_returns: pd.Series,
) -> tuple[float, float, pd.Series]:
    """Calculate portfolio loss and asset-level contributions for one scenario."""

    aligned_returns = scenario_asset_returns.reindex(current_weights.index)
    if aligned_returns.isna().any():
        raise ValueError("Scenario asset returns are missing required assets.")

    contributions = current_weights * aligned_returns
    portfolio_return = float(contributions.sum())
    loss = max(0.0, -portfolio_return)
    contribution_loss = -contributions
    return portfolio_return, loss, contribution_loss


def run_historical_stress_scenarios(
    asset_returns: pd.DataFrame,
    current_weights: pd.Series,
    scenarios: Dict[str, tuple[str, str]] | None = None,
) -> StressTestResult:
    """Replay historical crisis periods using current portfolio weights."""

    returns, weights = _validate_inputs(asset_returns, current_weights)
    scenario_periods = scenarios or DEFAULT_HISTORICAL_SCENARIOS

    loss_rows = []
    contribution_rows = []
    scenario_return_rows = []

    for scenario_name, (start_date, end_date) in scenario_periods.items():
        scenario_slice = returns.loc[start_date:end_date]
        if scenario_slice.empty:
            continue

        scenario_asset_returns = aggregate_period_returns(scenario_slice)
        scenario_return, loss, contribution_loss = calculate_stress_loss(weights, scenario_asset_returns)

        loss_rows.append(
            {
                "Scenario": scenario_name,
                "Start Date": pd.Timestamp(start_date),
                "End Date": pd.Timestamp(end_date),
                "Scenario Return": scenario_return,
                "Loss": loss,
            }
        )

        contribution_row = {"Scenario": scenario_name}
        contribution_row.update(contribution_loss.to_dict())
        contribution_rows.append(contribution_row)

        return_row = {"Scenario": scenario_name}
        return_row.update(scenario_asset_returns.to_dict())
        scenario_return_rows.append(return_row)

    if not loss_rows:
        raise ValueError("No stress scenarios matched the return history.")

    scenario_loss_table = pd.DataFrame(loss_rows).set_index("Scenario")
    stress_contribution_table = pd.DataFrame(contribution_rows).set_index("Scenario")
    scenario_asset_returns = pd.DataFrame(scenario_return_rows).set_index("Scenario")
    stress_comparison = scenario_loss_table.sort_values("Loss", ascending=False)

    return StressTestResult(
        scenario_loss_table=scenario_loss_table,
        stress_contribution_table=stress_contribution_table,
        stress_comparison=stress_comparison,
        scenario_asset_returns=scenario_asset_returns,
    )


def find_worst_historical_periods(
    asset_returns: pd.DataFrame,
    current_weights: pd.Series,
    windows: Dict[str, int] | None = None,
) -> StressTestResult:
    """Find the worst day, week, and month based on current portfolio weights."""

    returns, weights = _validate_inputs(asset_returns, current_weights)
    period_windows = windows or {"Worst Day": 1, "Worst Week": 5, "Worst Month": 21}

    loss_rows = []
    contribution_rows = []
    scenario_return_rows = []

    for scenario_name, window_size in period_windows.items():
        if window_size <= 0:
            raise ValueError("window sizes must be positive.")
        if len(returns) < window_size:
            continue

        rolling_asset_returns = (1.0 + returns).rolling(window_size).apply(
            lambda values: values.prod(),
            raw=True,
        ) - 1.0
        rolling_asset_returns = rolling_asset_returns.dropna(how="any")
        if rolling_asset_returns.empty:
            continue

        rolling_portfolio_returns = rolling_asset_returns[weights.index].dot(weights)
        worst_end_date = rolling_portfolio_returns.idxmin()
        worst_asset_returns = rolling_asset_returns.loc[worst_end_date]

        scenario_return, loss, contribution_loss = calculate_stress_loss(weights, worst_asset_returns)
        start_date = returns.loc[:worst_end_date].tail(window_size).index[0]
        loss_rows.append(
            {
                "Scenario": scenario_name,
                "Start Date": start_date,
                "End Date": worst_end_date,
                "Scenario Return": scenario_return,
                "Loss": loss,
            }
        )

        contribution_row = {"Scenario": scenario_name}
        contribution_row.update(contribution_loss.to_dict())
        contribution_rows.append(contribution_row)

        return_row = {"Scenario": scenario_name}
        return_row.update(worst_asset_returns.to_dict())
        scenario_return_rows.append(return_row)

    if not loss_rows:
        raise ValueError("Not enough data to compute worst historical periods.")

    scenario_loss_table = pd.DataFrame(loss_rows).set_index("Scenario")
    stress_contribution_table = pd.DataFrame(contribution_rows).set_index("Scenario")
    scenario_asset_returns = pd.DataFrame(scenario_return_rows).set_index("Scenario")
    stress_comparison = scenario_loss_table.sort_values("Loss", ascending=False)

    return StressTestResult(
        scenario_loss_table=scenario_loss_table,
        stress_contribution_table=stress_contribution_table,
        stress_comparison=stress_comparison,
        scenario_asset_returns=scenario_asset_returns,
    )
