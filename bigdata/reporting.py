"""Integrated daily risk reporting and limit monitoring utilities."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import pandas as pd

from .macro_factor import MacroFactorRiskResult
from .portfolio_construction import PortfolioConstructionResult
from .risk_engine import RiskEngineResult
from .stress_testing import StressTestResult


DEFAULT_LIMITS: Dict[str, float] = {
    "var_hist": 0.03,
    "es_hist": 0.04,
    "beta_abs": 0.8,
    "turnover": 0.25,
    "equity_beta": 0.6,
    "stress_loss": 0.12,
}


@dataclass(frozen=True)
class RiskReport:
    """Structured output for integrated risk reporting."""

    summary_table: pd.DataFrame
    factor_summary_table: pd.DataFrame
    top_positions_table: pd.DataFrame
    limit_check_table: pd.DataFrame
    breach_log: pd.DataFrame
    alert_summary: pd.Series


def _last_row(dataframe: pd.DataFrame, label: str) -> pd.Series:
    """Return the latest row from a DataFrame with validation."""

    if dataframe.empty:
        raise ValueError(f"{label} is empty.")
    return dataframe.iloc[-1]


def build_daily_risk_summary(
    portfolio_result: PortfolioConstructionResult,
    risk_result: RiskEngineResult,
    stress_result: StressTestResult,
    macro_result: MacroFactorRiskResult,
) -> pd.DataFrame:
    """Build a one-row daily summary table from rolling module outputs."""

    latest_weights = _last_row(portfolio_result.rolling_weights, "rolling_weights")
    latest_return = float(portfolio_result.portfolio_returns.iloc[-1])
    latest_turnover = float(portfolio_result.turnover_series.iloc[-1])
    latest_var = _last_row(risk_result.var_series, "var_series")
    latest_es = _last_row(risk_result.es_series, "es_series")
    latest_stress = _last_row(stress_result.scenario_loss_table, "scenario_loss_table")
    latest_beta = _last_row(macro_result.rolling_beta, "rolling_beta")
    latest_factor_contribution = _last_row(
        macro_result.factor_risk_contribution_by_date,
        "factor_risk_contribution_by_date",
    )
    stress_lookup = stress_result.scenario_loss_table

    def stress_loss_for(name: str) -> float:
        if name in stress_lookup.index:
            return float(stress_lookup.loc[name, "Loss"])
        return 0.0

    summary = pd.DataFrame(
        [
            {
                "date": portfolio_result.portfolio_returns.index[-1],
                "portfolio_return": latest_return,
                "volatility": float(risk_result.rolling_moments.iloc[-1]["volatility"]),
                "VaR_hist": float(latest_var["historical"]),
                "VaR_param": float(latest_var["parametric"]),
                "VaR_mc": float(latest_var["monte_carlo"]),
                "ES_hist": float(latest_es["historical"]),
                "ES_param": float(latest_es["parametric"]),
                "ES_mc": float(latest_es["monte_carlo"]),
                "stress_loss": float(latest_stress["Loss"]),
                "stress_scenario": stress_result.scenario_loss_table.index[-1],
                "worst_day_loss": stress_loss_for("Worst Day"),
                "worst_week_loss": stress_loss_for("Worst Week"),
                "worst_month_loss": stress_loss_for("Worst Month"),
                "equity_beta": float(latest_beta.get("SPY", 0.0)),
                "rates_beta": float(latest_beta.get("TLT", 0.0)),
                "commodity_beta": float(latest_beta.get("GLD", 0.0) + latest_beta.get("USO", 0.0)),
                "fx_beta": float(latest_beta.get("UUP", 0.0)),
                "systematic_share": float(macro_result.systematic_share_series.iloc[-1]),
                "idiosyncratic_share": float(macro_result.idiosyncratic_share_series.iloc[-1]),
                "top_position": latest_weights.abs().idxmax(),
                "top_position_weight": float(latest_weights.loc[latest_weights.abs().idxmax()]),
                "top_factor": latest_factor_contribution.abs().idxmax(),
                "top_factor_contribution": float(
                    latest_factor_contribution.loc[latest_factor_contribution.abs().idxmax()]
                ),
                "turnover": latest_turnover,
            }
        ]
    ).set_index("date")

    return summary


def build_factor_summary(macro_result: MacroFactorRiskResult) -> pd.DataFrame:
    """Summarize latest factor exposures and contributions."""

    latest_beta = _last_row(macro_result.rolling_beta, "rolling_beta")
    latest_contribution = _last_row(
        macro_result.factor_risk_contribution_by_date,
        "factor_risk_contribution_by_date",
    )

    return pd.DataFrame(
        {
            "beta": latest_beta,
            "risk_contribution": latest_contribution.reindex(latest_beta.index),
        }
    )


def build_top_positions_table(
    rolling_weights: pd.DataFrame,
    top_n: int = 3,
) -> pd.DataFrame:
    """Extract the largest current portfolio positions by absolute weight."""

    latest_weights = _last_row(rolling_weights, "rolling_weights")
    ordered = latest_weights.abs().sort_values(ascending=False).head(top_n).index

    return pd.DataFrame(
        {
            "weight": latest_weights.reindex(ordered),
            "abs_weight": latest_weights.abs().reindex(ordered),
        }
    )


def _metric_value_map(
    summary_table: pd.DataFrame,
    factor_summary_table: pd.DataFrame,
) -> Dict[str, float]:
    """Map monitored metrics to scalar values."""

    latest_summary = summary_table.iloc[-1]
    return {
        "var_hist": float(latest_summary["VaR_hist"]),
        "es_hist": float(latest_summary["ES_hist"]),
        "equity_beta": abs(float(latest_summary["equity_beta"])),
        "turnover": float(latest_summary["turnover"]),
        "stress_loss": float(latest_summary["stress_loss"]),
        "beta_abs": float(factor_summary_table["beta"].abs().max()),
    }


def monitor_risk_limits(
    summary_table: pd.DataFrame,
    factor_summary_table: pd.DataFrame,
    limits: Dict[str, float] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    """Evaluate monitored metrics against configurable limits."""

    limits_config = limits or DEFAULT_LIMITS
    metrics = _metric_value_map(summary_table, factor_summary_table)
    report_date = summary_table.index[-1]

    check_rows: List[Dict[str, object]] = []
    breach_rows: List[Dict[str, object]] = []
    alert_lines: List[str] = []

    for metric_name, limit_value in limits_config.items():
        if metric_name not in metrics:
            continue

        value = metrics[metric_name]
        if value > limit_value:
            status = "BREACH"
        elif value > 0.9 * limit_value:
            status = "WARNING"
        else:
            status = "OK"

        row = {
            "date": report_date,
            "metric_name": metric_name,
            "value": value,
            "limit": limit_value,
            "status": status,
        }
        check_rows.append(row)

        if status in {"WARNING", "BREACH"}:
            breach_rows.append(row)
            alert_lines.append(f"{metric_name}: {status} ({value:.4f} vs {limit_value:.4f})")

    limit_check_table = pd.DataFrame(check_rows).set_index("metric_name")
    breach_log = pd.DataFrame(breach_rows)
    alert_summary = pd.Series(
        {
            "date": report_date,
            "message": " | ".join(alert_lines) if alert_lines else "All monitored metrics within limits.",
        }
    )

    return limit_check_table, breach_log, alert_summary


def build_integrated_risk_report(
    portfolio_result: PortfolioConstructionResult,
    risk_result: RiskEngineResult,
    stress_result: StressTestResult,
    macro_result: MacroFactorRiskResult,
    limits: Dict[str, float] | None = None,
    top_n_positions: int = 3,
) -> RiskReport:
    """Build an integrated daily risk report with limit monitoring."""

    summary_table = build_daily_risk_summary(
        portfolio_result=portfolio_result,
        risk_result=risk_result,
        stress_result=stress_result,
        macro_result=macro_result,
    )
    factor_summary_table = build_factor_summary(macro_result)
    top_positions_table = build_top_positions_table(
        portfolio_result.rolling_weights,
        top_n=top_n_positions,
    )
    limit_check_table, breach_log, alert_summary = monitor_risk_limits(
        summary_table=summary_table,
        factor_summary_table=factor_summary_table,
        limits=limits,
    )

    return RiskReport(
        summary_table=summary_table,
        factor_summary_table=factor_summary_table,
        top_positions_table=top_positions_table,
        limit_check_table=limit_check_table,
        breach_log=breach_log,
        alert_summary=alert_summary,
    )
