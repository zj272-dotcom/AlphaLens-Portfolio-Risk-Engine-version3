"""Daily risk dashboard assembly and lightweight HTML rendering."""

from __future__ import annotations

from dataclasses import dataclass
from html import escape
from typing import Dict, List

import pandas as pd


@dataclass(frozen=True)
class Dashboard:
    """Structured dashboard object for latest-date risk display."""

    headline: Dict[str, str]
    risk_metrics: Dict[str, str]
    stress: Dict[str, str]
    factor_exposure: Dict[str, str]
    top_positions: pd.DataFrame
    limit_status: Dict[str, object]


def _format_pct(value: float) -> str:
    """Format a scalar as a percentage string."""

    if pd.isna(value):
        raise ValueError("Dashboard field contains NaN.")
    return f"{100.0 * float(value):.2f}%"


def _latest_date_from(summary_table: pd.DataFrame) -> pd.Timestamp:
    """Return and validate the latest dashboard date."""

    if summary_table.empty:
        raise ValueError("risk_summary_table is empty.")
    return summary_table.index[-1]


def _overall_risk_status(limit_check_table: pd.DataFrame) -> str:
    """Collapse limit statuses into a single dashboard risk state."""

    statuses = set(limit_check_table["status"])
    if "BREACH" in statuses:
        return "BREACH"
    if "WARNING" in statuses:
        return "WARNING"
    return "OK"


def _validate_required_columns(dataframe: pd.DataFrame, columns: List[str], label: str) -> None:
    """Ensure required columns exist before dashboard extraction."""

    missing = [column for column in columns if column not in dataframe.columns]
    if missing:
        raise ValueError(f"{label} is missing required columns: {missing}")


def build_daily_risk_dashboard(
    risk_summary_table: pd.DataFrame,
    factor_summary_table: pd.DataFrame,
    top_positions_table: pd.DataFrame,
    limit_check_table: pd.DataFrame,
    alert_summary: pd.Series,
) -> Dashboard:
    """Build a latest-date dashboard object from reporting outputs only."""

    _validate_required_columns(
        risk_summary_table,
        [
            "portfolio_return",
            "VaR_hist",
            "ES_hist",
            "VaR_param",
            "VaR_mc",
            "ES_param",
            "ES_mc",
            "volatility",
            "worst_day_loss",
            "worst_week_loss",
            "worst_month_loss",
            "equity_beta",
            "rates_beta",
            "commodity_beta",
            "fx_beta",
        ],
        "risk_summary_table",
    )
    _validate_required_columns(factor_summary_table, ["beta", "risk_contribution"], "factor_summary_table")
    _validate_required_columns(top_positions_table, ["weight", "abs_weight"], "top_positions_table")
    _validate_required_columns(limit_check_table, ["date", "value", "limit", "status"], "limit_check_table")

    latest_date = _latest_date_from(risk_summary_table)
    latest_summary = risk_summary_table.loc[latest_date]

    if isinstance(alert_summary, pd.Series) and "date" in alert_summary.index:
        alert_date = pd.Timestamp(alert_summary["date"])
        if alert_date != latest_date:
            raise ValueError("alert_summary is not aligned to the latest summary date.")

    risk_status = _overall_risk_status(limit_check_table)

    headline = {
        "date": str(latest_date.date()),
        "portfolio_return": _format_pct(latest_summary["portfolio_return"]),
        "VaR_hist": _format_pct(latest_summary["VaR_hist"]),
        "ES_hist": _format_pct(latest_summary["ES_hist"]),
        "risk_status": risk_status,
    }

    risk_metrics = {
        "volatility": _format_pct(latest_summary["volatility"]),
        "VaR_hist": _format_pct(latest_summary["VaR_hist"]),
        "VaR_param": _format_pct(latest_summary["VaR_param"]),
        "VaR_mc": _format_pct(latest_summary["VaR_mc"]),
        "ES_hist": _format_pct(latest_summary["ES_hist"]),
        "ES_param": _format_pct(latest_summary["ES_param"]),
        "ES_mc": _format_pct(latest_summary["ES_mc"]),
    }

    stress = {
        "worst_day_loss": _format_pct(latest_summary["worst_day_loss"]),
        "worst_week_loss": _format_pct(latest_summary["worst_week_loss"]),
        "worst_month_loss": _format_pct(latest_summary["worst_month_loss"]),
    }

    factor_exposure = {
        "equity_exposure": _format_pct(latest_summary["equity_beta"]),
        "rates_exposure": _format_pct(latest_summary["rates_beta"]),
        "commodity_exposure": _format_pct(latest_summary["commodity_beta"]),
        "fx_exposure": _format_pct(latest_summary["fx_beta"]),
    }

    formatted_positions = top_positions_table.copy()
    formatted_positions["weight"] = formatted_positions["weight"].map(_format_pct)
    formatted_positions["abs_weight"] = formatted_positions["abs_weight"].map(_format_pct)

    limit_status = {
        "statuses": limit_check_table["status"].to_dict(),
        "table": limit_check_table.copy(),
        "alert_summary": alert_summary["message"] if "message" in alert_summary.index else str(alert_summary),
    }

    return Dashboard(
        headline=headline,
        risk_metrics=risk_metrics,
        stress=stress,
        factor_exposure=factor_exposure,
        top_positions=formatted_positions,
        limit_status=limit_status,
    )


def render_dashboard_html(dashboard: Dashboard) -> str:
    """Render a lightweight HTML dashboard without a front-end framework."""

    def dict_table(title: str, data: Dict[str, str]) -> str:
        rows = "".join(
            f"<tr><th>{escape(str(key))}</th><td>{escape(str(value))}</td></tr>"
            for key, value in data.items()
        )
        return f"<section><h2>{escape(title)}</h2><table>{rows}</table></section>"

    top_positions_html = dashboard.top_positions.to_html(escape=True)
    limit_table = dashboard.limit_status["table"][["value", "limit", "status"]].copy()
    limit_table["value"] = limit_table["value"].map(_format_pct)
    limit_table["limit"] = limit_table["limit"].map(_format_pct)
    limit_html = limit_table.to_html(escape=True)

    return (
        "<html><body>"
        f"{dict_table('Headline', dashboard.headline)}"
        f"{dict_table('Risk Metrics', dashboard.risk_metrics)}"
        f"{dict_table('Stress', dashboard.stress)}"
        f"{dict_table('Factor Exposure', dashboard.factor_exposure)}"
        f"<section><h2>Top Positions</h2>{top_positions_html}</section>"
        f"<section><h2>Limit Monitoring</h2>{limit_html}"
        f"<p>{escape(str(dashboard.limit_status['alert_summary']))}</p></section>"
        "</body></html>"
    )
