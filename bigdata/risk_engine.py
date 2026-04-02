"""Rolling portfolio risk engine focused on stable daily risk estimates."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import numpy as np
import pandas as pd
from scipy.stats import norm


@dataclass(frozen=True)
class RiskEngineResult:
    """Standard output bundle for rolling portfolio risk estimation."""

    var_series: pd.DataFrame
    es_series: pd.DataFrame
    pnl_distribution: pd.DataFrame
    distribution_date: pd.Timestamp
    rolling_moments: pd.DataFrame


def _validate_return_series(portfolio_returns: pd.Series) -> pd.Series:
    """Validate and normalize a portfolio return series."""

    if portfolio_returns.empty:
        raise ValueError("portfolio_returns is empty.")

    clean = portfolio_returns.dropna().astype(float).sort_index()
    if clean.empty:
        raise ValueError("portfolio_returns contains no valid observations.")

    clean.name = portfolio_returns.name or "portfolio_return"
    return clean


def historical_var_es(window_returns: pd.Series, confidence_level: float = 0.95) -> tuple[float, float]:
    """Compute historical VaR and ES from a rolling window."""

    window = _validate_return_series(window_returns)
    tail_probability = 1.0 - confidence_level
    quantile = float(window.quantile(tail_probability))
    tail_losses = window[window <= quantile]
    expected_shortfall = float(tail_losses.mean()) if not tail_losses.empty else quantile

    var = max(0.0, -quantile)
    es = max(var, -expected_shortfall)
    return var, es


def parametric_var_es(window_returns: pd.Series, confidence_level: float = 0.95) -> tuple[float, float]:
    """Compute normal-theory VaR and ES from a rolling window."""

    window = _validate_return_series(window_returns)
    if len(window) < 2:
        raise ValueError("At least two observations are required for parametric VaR/ES.")

    mu = float(window.mean())
    sigma = float(window.std(ddof=1))
    sigma = max(sigma, 1e-12)

    z = float(norm.ppf(1.0 - confidence_level))
    pdf_at_z = float(norm.pdf(z))

    var = -(mu + sigma * z)
    es = -(mu - sigma * pdf_at_z / (1.0 - confidence_level))
    return max(0.0, var), max(var, es)


def monte_carlo_var_es(
    window_returns: pd.Series,
    confidence_level: float = 0.95,
    n_simulations: int = 10_000,
    random_state: int = 42,
) -> tuple[float, float, np.ndarray]:
    """Compute Monte Carlo VaR and ES using a normal return model."""

    window = _validate_return_series(window_returns)
    if len(window) < 2:
        raise ValueError("At least two observations are required for Monte Carlo VaR/ES.")
    if n_simulations <= 0:
        raise ValueError("n_simulations must be positive.")

    mu = float(window.mean())
    sigma = float(window.std(ddof=1))
    sigma = max(sigma, 1e-12)

    rng = np.random.default_rng(random_state)
    simulated_returns = rng.normal(loc=mu, scale=sigma, size=n_simulations)
    simulated_series = pd.Series(simulated_returns)
    var, es = historical_var_es(simulated_series, confidence_level=confidence_level)

    return var, es, simulated_returns


def build_pnl_distribution(
    window_returns: pd.Series,
    monte_carlo_sample: np.ndarray,
    random_state: int = 42,
) -> pd.DataFrame:
    """Build a plot-ready PnL distribution snapshot for the latest risk date."""

    historical = _validate_return_series(window_returns)
    rng = np.random.default_rng(random_state)

    bootstrap_sample = rng.choice(historical.to_numpy(), size=len(monte_carlo_sample), replace=True)
    mu = float(historical.mean())
    sigma = max(float(historical.std(ddof=1)), 1e-12)
    parametric_sample = rng.normal(loc=mu, scale=sigma, size=len(monte_carlo_sample))

    return pd.DataFrame(
        {
            "historical": bootstrap_sample,
            "parametric": parametric_sample,
            "monte_carlo": monte_carlo_sample,
        }
    )


def run_rolling_risk_engine(
    portfolio_returns: pd.Series,
    window_size: int = 252,
    confidence_level: float = 0.95,
    n_simulations: int = 10_000,
    random_state: int = 42,
) -> RiskEngineResult:
    """Run rolling historical, parametric, and Monte Carlo VaR/ES estimates."""

    returns = _validate_return_series(portfolio_returns)
    if window_size < 2:
        raise ValueError("window_size must be at least 2.")
    if len(returns) <= window_size:
        raise ValueError("Not enough data to compute rolling risk estimates.")
    if not 0.0 < confidence_level < 1.0:
        raise ValueError("confidence_level must lie strictly between 0 and 1.")

    var_rows = []
    es_rows = []
    moment_rows = []
    latest_distribution = None
    latest_distribution_date = None

    for index_position in range(window_size, len(returns)):
        evaluation_date = returns.index[index_position]
        window = returns.iloc[index_position - window_size : index_position]

        hist_var, hist_es = historical_var_es(window, confidence_level=confidence_level)
        param_var, param_es = parametric_var_es(window, confidence_level=confidence_level)
        mc_var, mc_es, mc_sample = monte_carlo_var_es(
            window,
            confidence_level=confidence_level,
            n_simulations=n_simulations,
            random_state=random_state + index_position,
        )

        var_rows.append(
            {
                "date": evaluation_date,
                "historical": hist_var,
                "parametric": param_var,
                "monte_carlo": mc_var,
            }
        )
        es_rows.append(
            {
                "date": evaluation_date,
                "historical": hist_es,
                "parametric": param_es,
                "monte_carlo": mc_es,
            }
        )
        moment_rows.append(
            {
                "date": evaluation_date,
                "mean": float(window.mean()),
                "volatility": float(window.std(ddof=1)),
            }
        )

        latest_distribution = build_pnl_distribution(
            window,
            monte_carlo_sample=mc_sample,
            random_state=random_state + index_position,
        )
        latest_distribution_date = evaluation_date

    var_series = pd.DataFrame(var_rows).set_index("date")
    es_series = pd.DataFrame(es_rows).set_index("date")
    rolling_moments = pd.DataFrame(moment_rows).set_index("date")

    return RiskEngineResult(
        var_series=var_series,
        es_series=es_series,
        pnl_distribution=latest_distribution,
        distribution_date=latest_distribution_date,
        rolling_moments=rolling_moments,
    )
