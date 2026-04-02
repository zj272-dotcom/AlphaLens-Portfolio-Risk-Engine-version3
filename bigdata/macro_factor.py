"""Rolling macro factor risk decomposition utilities."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class MacroFactorRiskResult:
    """Structured output for rolling macro factor risk decomposition."""

    rolling_beta: pd.DataFrame
    factor_covariance_by_date: Dict[pd.Timestamp, pd.DataFrame]
    systematic_risk_series: pd.Series
    idiosyncratic_risk_series: pd.Series
    systematic_share_series: pd.Series
    idiosyncratic_share_series: pd.Series
    factor_risk_contribution_by_date: pd.DataFrame


def _validate_macro_inputs(
    portfolio_returns: pd.Series,
    asset_returns: pd.DataFrame,
    factor_set: List[str],
    window_size: int,
) -> tuple[pd.Series, pd.DataFrame]:
    """Validate and align inputs for rolling factor decomposition."""

    if portfolio_returns.empty:
        raise ValueError("portfolio_returns is empty.")
    if asset_returns.empty:
        raise ValueError("asset_returns is empty.")
    if not factor_set:
        raise ValueError("factor_set cannot be empty.")
    if window_size < 2:
        raise ValueError("window_size must be at least 2.")

    missing_factors = [factor for factor in factor_set if factor not in asset_returns.columns]
    if missing_factors:
        raise ValueError(f"Factors not found in asset_returns: {missing_factors}")
    if len(set(factor_set)) != len(factor_set):
        raise ValueError("factor_set contains duplicates.")

    aligned = pd.concat(
        [portfolio_returns.rename("portfolio_return"), asset_returns[factor_set]],
        axis=1,
        join="inner",
    ).dropna()

    if len(aligned) <= window_size:
        raise ValueError("Not enough aligned data for rolling macro factor decomposition.")

    y = aligned["portfolio_return"].astype(float)
    x = aligned[factor_set].astype(float)
    return y, x


def estimate_factor_beta(
    portfolio_window: pd.Series,
    factor_window: pd.DataFrame,
) -> pd.Series:
    """Estimate factor betas by stable least squares without look-ahead."""

    y = portfolio_window.to_numpy(dtype=float)
    x = factor_window.to_numpy(dtype=float)

    if len(y) != len(x):
        raise ValueError("portfolio_window and factor_window must have the same length.")
    if len(y) < 2:
        raise ValueError("At least two observations are required to estimate betas.")

    beta, *_ = np.linalg.lstsq(x, y, rcond=None)
    return pd.Series(beta, index=factor_window.columns, name="beta")


def compute_factor_covariance(factor_window: pd.DataFrame) -> pd.DataFrame:
    """Compute the factor covariance matrix within a rolling window."""

    if factor_window.empty:
        raise ValueError("factor_window is empty.")

    covariance = factor_window.cov().astype(float)
    return covariance


def decompose_risk(
    beta: pd.Series,
    factor_covariance: pd.DataFrame,
    residuals: pd.Series,
) -> tuple[float, float, pd.Series]:
    """Split total variance into systematic and idiosyncratic pieces."""

    beta_vector = beta.to_numpy(dtype=float)
    covariance_matrix = factor_covariance.reindex(index=beta.index, columns=beta.index).to_numpy(dtype=float)

    systematic_risk = float(beta_vector @ covariance_matrix @ beta_vector.T)
    systematic_risk = max(systematic_risk, 0.0)

    idiosyncratic_risk = float(residuals.var(ddof=1)) if len(residuals) > 1 else 0.0
    idiosyncratic_risk = max(idiosyncratic_risk, 0.0)

    marginal_factor_risk = factor_covariance.reindex(index=beta.index, columns=beta.index).dot(beta)
    contribution = beta * marginal_factor_risk

    return systematic_risk, idiosyncratic_risk, contribution


def run_macro_factor_risk_decomposition(
    portfolio_returns: pd.Series,
    asset_returns: pd.DataFrame,
    factor_set: List[str],
    window_size: int = 252,
) -> MacroFactorRiskResult:
    """Run rolling macro factor decomposition using user-selected factor returns."""

    y, x = _validate_macro_inputs(
        portfolio_returns=portfolio_returns,
        asset_returns=asset_returns,
        factor_set=factor_set,
        window_size=window_size,
    )

    beta_rows = []
    factor_covariance_by_date: Dict[pd.Timestamp, pd.DataFrame] = {}
    systematic_rows = []
    idiosyncratic_rows = []
    contribution_rows = []

    for index_position in range(window_size, len(y)):
        evaluation_date = y.index[index_position]
        y_window = y.iloc[index_position - window_size : index_position]
        x_window = x.iloc[index_position - window_size : index_position]

        beta = estimate_factor_beta(y_window, x_window)
        factor_covariance = compute_factor_covariance(x_window)
        fitted = x_window.dot(beta)
        residuals = y_window - fitted

        systematic_risk, idiosyncratic_risk, contribution = decompose_risk(
            beta=beta,
            factor_covariance=factor_covariance,
            residuals=residuals,
        )

        total_risk = systematic_risk + idiosyncratic_risk
        if total_risk <= 0:
            systematic_share = 0.0
            idiosyncratic_share = 0.0
        else:
            systematic_share = systematic_risk / total_risk
            idiosyncratic_share = idiosyncratic_risk / total_risk

        beta.name = evaluation_date
        beta_rows.append(beta)
        factor_covariance_by_date[evaluation_date] = factor_covariance
        systematic_rows.append((evaluation_date, systematic_risk, systematic_share))
        idiosyncratic_rows.append((evaluation_date, idiosyncratic_risk, idiosyncratic_share))

        contribution.name = evaluation_date
        contribution_rows.append(contribution)

    rolling_beta = pd.DataFrame(beta_rows)
    rolling_beta.index.name = "date"

    systematic_risk_series = pd.Series(
        {date: value for date, value, _ in systematic_rows},
        name="systematic_risk",
    ).sort_index()
    idiosyncratic_risk_series = pd.Series(
        {date: value for date, value, _ in idiosyncratic_rows},
        name="idiosyncratic_risk",
    ).sort_index()
    systematic_share_series = pd.Series(
        {date: value for date, _, value in systematic_rows},
        name="systematic_share",
    ).sort_index()
    idiosyncratic_share_series = pd.Series(
        {date: value for date, _, value in idiosyncratic_rows},
        name="idiosyncratic_share",
    ).sort_index()

    factor_risk_contribution_by_date = pd.DataFrame(contribution_rows)
    factor_risk_contribution_by_date.index.name = "date"

    return MacroFactorRiskResult(
        rolling_beta=rolling_beta,
        factor_covariance_by_date=factor_covariance_by_date,
        systematic_risk_series=systematic_risk_series,
        idiosyncratic_risk_series=idiosyncratic_risk_series,
        systematic_share_series=systematic_share_series,
        idiosyncratic_share_series=idiosyncratic_share_series,
        factor_risk_contribution_by_date=factor_risk_contribution_by_date,
    )
