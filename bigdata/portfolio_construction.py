"""Rolling portfolio construction utilities focused on stable risk allocation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from .data_pipeline import RollingWindow


@dataclass(frozen=True)
class PortfolioConstructionResult:
    """Standard output bundle for rolling portfolio construction."""

    rolling_weights: pd.DataFrame
    portfolio_returns: pd.Series
    turnover_series: pd.Series
    covariance_by_rebalance: Dict[pd.Timestamp, pd.DataFrame]


def estimate_covariance_matrix(train_returns: pd.DataFrame) -> pd.DataFrame:
    """Estimate a sample covariance matrix from a train window."""

    if train_returns.empty:
        raise ValueError("Training return matrix is empty.")
    if train_returns.isna().any().any():
        raise ValueError("Training return matrix contains NaN values.")
    if len(train_returns) < 2:
        raise ValueError("At least two observations are required to estimate covariance.")

    covariance = train_returns.cov()
    return covariance.astype(float)


def shrink_covariance(
    sample_covariance: pd.DataFrame,
    shrinkage_intensity: float = 0.10,
    target: str = "diagonal",
    min_eigenvalue: float = 1e-8,
) -> pd.DataFrame:
    """Shrink a covariance matrix toward a stable target and enforce PSD."""

    if sample_covariance.empty:
        raise ValueError("Sample covariance matrix is empty.")
    if sample_covariance.shape[0] != sample_covariance.shape[1]:
        raise ValueError("Covariance matrix must be square.")
    if not 0.0 <= shrinkage_intensity <= 1.0:
        raise ValueError("shrinkage_intensity must be between 0 and 1.")

    sample = sample_covariance.to_numpy(dtype=float)
    columns = sample_covariance.columns

    if target == "diagonal":
        shrink_target = np.diag(np.diag(sample))
    elif target == "identity":
        average_variance = float(np.trace(sample) / sample.shape[0])
        shrink_target = np.eye(sample.shape[0]) * average_variance
    else:
        raise ValueError("target must be either 'diagonal' or 'identity'.")

    shrunk = (1.0 - shrinkage_intensity) * sample + shrinkage_intensity * shrink_target
    shrunk = (shrunk + shrunk.T) / 2.0

    eigenvalues, eigenvectors = np.linalg.eigh(shrunk)
    clipped_eigenvalues = np.clip(eigenvalues, min_eigenvalue, None)
    stabilized = eigenvectors @ np.diag(clipped_eigenvalues) @ eigenvectors.T
    stabilized = (stabilized + stabilized.T) / 2.0

    return pd.DataFrame(stabilized, index=columns, columns=columns)


def compute_equal_weight(asset_names: List[str]) -> pd.Series:
    """Compute a fully invested equal-weight portfolio."""

    if not asset_names:
        raise ValueError("asset_names cannot be empty.")

    n_assets = len(asset_names)
    weights = np.repeat(1.0 / n_assets, n_assets)
    return pd.Series(weights, index=asset_names, name="weight")


def compute_min_variance_weights(
    covariance_matrix: pd.DataFrame,
    max_weight: float = 0.30,
) -> pd.Series:
    """Compute long-only minimum-variance weights with a max-weight cap."""

    if covariance_matrix.empty:
        raise ValueError("Covariance matrix is empty.")
    if covariance_matrix.shape[0] != covariance_matrix.shape[1]:
        raise ValueError("Covariance matrix must be square.")

    n_assets = covariance_matrix.shape[0]
    if max_weight <= 0 or max_weight > 1:
        raise ValueError("max_weight must be in the interval (0, 1].")
    if n_assets * max_weight < 1.0:
        raise ValueError("max_weight is too small to allow a fully invested portfolio.")

    covariance = covariance_matrix.to_numpy(dtype=float)
    initial_guess = np.repeat(1.0 / n_assets, n_assets)
    bounds = [(0.0, max_weight) for _ in range(n_assets)]
    constraints = [{"type": "eq", "fun": lambda weights: np.sum(weights) - 1.0}]

    def objective(weights: np.ndarray) -> float:
        return float(weights @ covariance @ weights)

    result = minimize(
        objective,
        x0=initial_guess,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
        options={"ftol": 1e-12, "maxiter": 500},
    )

    if not result.success:
        raise ValueError(f"Minimum-variance optimization failed: {result.message}")

    weights = np.clip(result.x, 0.0, max_weight)
    weights = weights / weights.sum()

    return pd.Series(weights, index=covariance_matrix.index, name="weight")


def compute_turnover(previous_weights: pd.Series | None, current_weights: pd.Series) -> float:
    """Compute portfolio turnover between two rebalance weights."""

    current = current_weights.astype(float).sort_index()

    if previous_weights is None:
        return 0.0

    previous = previous_weights.astype(float).reindex(current.index).fillna(0.0)
    return float((current - previous).abs().sum())


def _validate_non_overlapping_test_windows(windows: List[RollingWindow]) -> None:
    """Ensure test windows are strictly increasing and non-overlapping."""

    for previous, current in zip(windows, windows[1:]):
        if current.test_start <= previous.test_end:
            raise ValueError("Rolling test windows overlap; choose a larger step_size.")


def run_rolling_portfolio_construction(
    windows: List[RollingWindow],
    strategy: str = "min_variance",
    shrinkage_intensity: float = 0.10,
    shrinkage_target: str = "diagonal",
    max_weight: float = 0.30,
) -> PortfolioConstructionResult:
    """Run rolling portfolio construction without look-ahead bias."""

    if not windows:
        raise ValueError("windows cannot be empty.")

    _validate_non_overlapping_test_windows(windows)

    weights_records = []
    portfolio_return_parts = []
    turnover_records = []
    covariance_by_rebalance: Dict[pd.Timestamp, pd.DataFrame] = {}
    previous_weights: pd.Series | None = None

    for window in windows:
        sample_covariance = estimate_covariance_matrix(window.train_data)
        shrunk_covariance = shrink_covariance(
            sample_covariance,
            shrinkage_intensity=shrinkage_intensity,
            target=shrinkage_target,
        )
        rebalance_date = window.test_start
        covariance_by_rebalance[rebalance_date] = shrunk_covariance

        if strategy == "equal_weight":
            weights = compute_equal_weight(list(window.train_data.columns))
        elif strategy == "min_variance":
            weights = compute_min_variance_weights(
                shrunk_covariance,
                max_weight=max_weight,
            )
        else:
            raise ValueError("strategy must be either 'equal_weight' or 'min_variance'.")

        weights.name = rebalance_date
        weights_records.append(weights)

        turnover = compute_turnover(previous_weights, weights)
        turnover_records.append((rebalance_date, turnover))

        test_returns = window.test_data[weights.index].dot(weights)
        test_returns.name = "portfolio_return"
        portfolio_return_parts.append(test_returns)

        previous_weights = weights

    rolling_weights = pd.DataFrame(weights_records)
    rolling_weights.index.name = "rebalance_date"

    portfolio_returns = pd.concat(portfolio_return_parts).sort_index()
    if portfolio_returns.index.has_duplicates:
        raise ValueError("Portfolio return index contains duplicate dates; test windows overlap.")

    turnover_series = pd.Series(
        {rebalance_date: turnover for rebalance_date, turnover in turnover_records},
        name="turnover",
    ).sort_index()
    turnover_series.index.name = "rebalance_date"

    return PortfolioConstructionResult(
        rolling_weights=rolling_weights,
        portfolio_returns=portfolio_returns,
        turnover_series=turnover_series,
        covariance_by_rebalance=covariance_by_rebalance,
    )
