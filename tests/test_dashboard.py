import pandas as pd
import pytest

from bigdata.dashboard import Dashboard, build_daily_risk_dashboard, render_dashboard_html
from bigdata.reporting import build_integrated_risk_report


def test_build_daily_risk_dashboard_returns_all_sections(reporting_inputs):
    portfolio_result, risk_result, stress_result, macro_result = reporting_inputs
    report = build_integrated_risk_report(
        portfolio_result=portfolio_result,
        risk_result=risk_result,
        stress_result=stress_result,
        macro_result=macro_result,
        top_n_positions=3,
    )

    dashboard = build_daily_risk_dashboard(
        risk_summary_table=report.summary_table,
        factor_summary_table=report.factor_summary_table,
        top_positions_table=report.top_positions_table,
        limit_check_table=report.limit_check_table,
        alert_summary=report.alert_summary,
    )

    assert isinstance(dashboard, Dashboard)
    assert set(dashboard.headline) == {"date", "portfolio_return", "VaR_hist", "ES_hist", "risk_status"}
    assert "volatility" in dashboard.risk_metrics
    assert "worst_day_loss" in dashboard.stress
    assert "equity_exposure" in dashboard.factor_exposure
    assert dashboard.top_positions.shape[0] == report.top_positions_table.shape[0]
    assert "statuses" in dashboard.limit_status


def test_render_dashboard_html_contains_sections(reporting_inputs):
    portfolio_result, risk_result, stress_result, macro_result = reporting_inputs
    report = build_integrated_risk_report(
        portfolio_result=portfolio_result,
        risk_result=risk_result,
        stress_result=stress_result,
        macro_result=macro_result,
        top_n_positions=3,
    )
    dashboard = build_daily_risk_dashboard(
        risk_summary_table=report.summary_table,
        factor_summary_table=report.factor_summary_table,
        top_positions_table=report.top_positions_table,
        limit_check_table=report.limit_check_table,
        alert_summary=report.alert_summary,
    )

    html = render_dashboard_html(dashboard)

    assert "Headline" in html
    assert "Risk Metrics" in html
    assert "Stress" in html
    assert "Factor Exposure" in html
    assert "Limit Monitoring" in html


def test_build_daily_risk_dashboard_empty_summary_table_raises(reporting_inputs):
    _, _, _, macro_result = reporting_inputs
    empty_summary = pd.DataFrame()
    factor_summary = pd.DataFrame(
        {"beta": [0.1], "risk_contribution": [0.2]},
        index=["equity"],
    )
    top_positions = pd.DataFrame(
        {"weight": [0.3], "abs_weight": [0.3]},
        index=["SPY"],
    )
    limit_check = pd.DataFrame(
        {
            "date": [pd.Timestamp("2024-01-01")],
            "value": [0.01],
            "limit": [0.02],
            "status": ["OK"],
        },
        index=["VaR_hist"],
    )
    alert_summary = pd.Series({"date": pd.Timestamp("2024-01-01"), "message": "All clear"})

    with pytest.raises(ValueError, match="risk_summary_table"):
        build_daily_risk_dashboard(
            risk_summary_table=empty_summary,
            factor_summary_table=factor_summary,
            top_positions_table=top_positions,
            limit_check_table=limit_check,
            alert_summary=alert_summary,
        )


def test_build_daily_risk_dashboard_missing_required_columns_in_summary_raises():
    risk_summary = pd.DataFrame(
        {"portfolio_return": [0.01]},
        index=[pd.Timestamp("2024-01-01")],
    )
    factor_summary = pd.DataFrame(
        {"beta": [0.1], "risk_contribution": [0.2]},
        index=["equity"],
    )
    top_positions = pd.DataFrame(
        {"weight": [0.3], "abs_weight": [0.3]},
        index=["SPY"],
    )
    limit_check = pd.DataFrame(
        {
            "date": [pd.Timestamp("2024-01-01")],
            "value": [0.01],
            "limit": [0.02],
            "status": ["OK"],
        },
        index=["VaR_hist"],
    )
    alert_summary = pd.Series({"date": pd.Timestamp("2024-01-01"), "message": "All clear"})

    with pytest.raises(ValueError, match="risk_summary_table"):
        build_daily_risk_dashboard(
            risk_summary_table=risk_summary,
            factor_summary_table=factor_summary,
            top_positions_table=top_positions,
            limit_check_table=limit_check,
            alert_summary=alert_summary,
        )


def test_build_daily_risk_dashboard_missing_required_columns_in_factor_table_raises():
    risk_summary = pd.DataFrame(
        {
            "portfolio_return": [0.01],
            "VaR_hist": [0.02],
            "ES_hist": [0.03],
            "VaR_param": [0.02],
            "VaR_mc": [0.02],
            "ES_param": [0.03],
            "ES_mc": [0.03],
            "volatility": [0.01],
            "worst_day_loss": [0.04],
            "worst_week_loss": [0.05],
            "worst_month_loss": [0.06],
            "equity_beta": [0.8],
            "rates_beta": [0.1],
            "commodity_beta": [0.05],
            "fx_beta": [0.05],
        },
        index=[pd.Timestamp("2024-01-01")],
    )
    factor_summary = pd.DataFrame({"beta": [0.1]}, index=["equity"])
    top_positions = pd.DataFrame(
        {"weight": [0.3], "abs_weight": [0.3]},
        index=["SPY"],
    )
    limit_check = pd.DataFrame(
        {
            "date": [pd.Timestamp("2024-01-01")],
            "value": [0.01],
            "limit": [0.02],
            "status": ["OK"],
        },
        index=["VaR_hist"],
    )
    alert_summary = pd.Series({"date": pd.Timestamp("2024-01-01"), "message": "All clear"})

    with pytest.raises(ValueError, match="factor_summary_table"):
        build_daily_risk_dashboard(
            risk_summary_table=risk_summary,
            factor_summary_table=factor_summary,
            top_positions_table=top_positions,
            limit_check_table=limit_check,
            alert_summary=alert_summary,
        )


def test_build_daily_risk_dashboard_alert_date_misaligned_raises(reporting_inputs):
    portfolio_result, risk_result, stress_result, macro_result = reporting_inputs
    report = build_integrated_risk_report(
        portfolio_result=portfolio_result,
        risk_result=risk_result,
        stress_result=stress_result,
        macro_result=macro_result,
        top_n_positions=3,
    )

    bad_alert = report.alert_summary.copy()
    bad_alert["date"] = pd.Timestamp("1999-01-01")

    with pytest.raises(ValueError, match="alert_summary"):
        build_daily_risk_dashboard(
            risk_summary_table=report.summary_table,
            factor_summary_table=report.factor_summary_table,
            top_positions_table=report.top_positions_table,
            limit_check_table=report.limit_check_table,
            alert_summary=bad_alert,
        )


def test_build_daily_risk_dashboard_overall_status_warning():
    risk_summary = pd.DataFrame(
        {
            "portfolio_return": [0.01],
            "VaR_hist": [0.02],
            "ES_hist": [0.03],
            "VaR_param": [0.02],
            "VaR_mc": [0.02],
            "ES_param": [0.03],
            "ES_mc": [0.03],
            "volatility": [0.01],
            "worst_day_loss": [0.04],
            "worst_week_loss": [0.05],
            "worst_month_loss": [0.06],
            "equity_beta": [0.8],
            "rates_beta": [0.1],
            "commodity_beta": [0.05],
            "fx_beta": [0.05],
        },
        index=[pd.Timestamp("2024-01-01")],
    )
    factor_summary = pd.DataFrame(
        {"beta": [0.1], "risk_contribution": [0.2]},
        index=["equity"],
    )
    top_positions = pd.DataFrame(
        {"weight": [0.3], "abs_weight": [0.3]},
        index=["SPY"],
    )
    limit_check = pd.DataFrame(
        {
            "date": [pd.Timestamp("2024-01-01"), pd.Timestamp("2024-01-01")],
            "value": [0.01, 0.03],
            "limit": [0.02, 0.02],
            "status": ["OK", "WARNING"],
        },
        index=["VaR_hist", "ES_hist"],
    )
    alert_summary = pd.Series({"date": pd.Timestamp("2024-01-01"), "message": "Watch list"})

    dashboard = build_daily_risk_dashboard(
        risk_summary_table=risk_summary,
        factor_summary_table=factor_summary,
        top_positions_table=top_positions,
        limit_check_table=limit_check,
        alert_summary=alert_summary,
    )

    assert dashboard.headline["risk_status"] == "WARNING"


def test_build_daily_risk_dashboard_overall_status_breach():
    risk_summary = pd.DataFrame(
        {
            "portfolio_return": [0.01],
            "VaR_hist": [0.02],
            "ES_hist": [0.03],
            "VaR_param": [0.02],
            "VaR_mc": [0.02],
            "ES_param": [0.03],
            "ES_mc": [0.03],
            "volatility": [0.01],
            "worst_day_loss": [0.04],
            "worst_week_loss": [0.05],
            "worst_month_loss": [0.06],
            "equity_beta": [0.8],
            "rates_beta": [0.1],
            "commodity_beta": [0.05],
            "fx_beta": [0.05],
        },
        index=[pd.Timestamp("2024-01-01")],
    )
    factor_summary = pd.DataFrame(
        {"beta": [0.1], "risk_contribution": [0.2]},
        index=["equity"],
    )
    top_positions = pd.DataFrame(
        {"weight": [0.3], "abs_weight": [0.3]},
        index=["SPY"],
    )
    limit_check = pd.DataFrame(
        {
            "date": [pd.Timestamp("2024-01-01"), pd.Timestamp("2024-01-01")],
            "value": [0.01, 0.03],
            "limit": [0.02, 0.02],
            "status": ["WARNING", "BREACH"],
        },
        index=["VaR_hist", "ES_hist"],
    )
    alert_summary = pd.Series({"date": pd.Timestamp("2024-01-01"), "message": "Escalate"})

    dashboard = build_daily_risk_dashboard(
        risk_summary_table=risk_summary,
        factor_summary_table=factor_summary,
        top_positions_table=top_positions,
        limit_check_table=limit_check,
        alert_summary=alert_summary,
    )

    assert dashboard.headline["risk_status"] == "BREACH"


def test_build_daily_risk_dashboard_nan_field_raises():
    risk_summary = pd.DataFrame(
        {
            "portfolio_return": [0.01],
            "VaR_hist": [0.02],
            "ES_hist": [0.03],
            "VaR_param": [0.02],
            "VaR_mc": [0.02],
            "ES_param": [0.03],
            "ES_mc": [0.03],
            "volatility": [float("nan")],
            "worst_day_loss": [0.04],
            "worst_week_loss": [0.05],
            "worst_month_loss": [0.06],
            "equity_beta": [0.8],
            "rates_beta": [0.1],
            "commodity_beta": [0.05],
            "fx_beta": [0.05],
        },
        index=[pd.Timestamp("2024-01-01")],
    )
    factor_summary = pd.DataFrame(
        {"beta": [0.1], "risk_contribution": [0.2]},
        index=["equity"],
    )
    top_positions = pd.DataFrame(
        {"weight": [0.3], "abs_weight": [0.3]},
        index=["SPY"],
    )
    limit_check = pd.DataFrame(
        {
            "date": [pd.Timestamp("2024-01-01")],
            "value": [0.01],
            "limit": [0.02],
            "status": ["OK"],
        },
        index=["VaR_hist"],
    )
    alert_summary = pd.Series({"date": pd.Timestamp("2024-01-01"), "message": "All clear"})

    with pytest.raises(ValueError, match="NaN"):
        build_daily_risk_dashboard(
            risk_summary_table=risk_summary,
            factor_summary_table=factor_summary,
            top_positions_table=top_positions,
            limit_check_table=limit_check,
            alert_summary=alert_summary,
        )


def test_render_dashboard_html_escapes_html_characters():
    dashboard = Dashboard(
        headline={"date": "2024-01-01", "risk_status": "<BREACH>"},
        risk_metrics={"volatility": "1.00%"},
        stress={"worst_day_loss": "2.00%"},
        factor_exposure={"equity_exposure": "50.00%"},
        top_positions=pd.DataFrame(
            {"weight": ["10.00%"], "abs_weight": ["10.00%"]},
            index=["<SPY>"],
        ),
        limit_status={
            "statuses": {"VaR": "BREACH"},
            "table": pd.DataFrame(
                {"value": [0.01], "limit": [0.02], "status": ["BREACH"]},
                index=["<VaR_hist>"],
            ),
            "alert_summary": "<script>alert(1)</script>",
        },
    )

    html = render_dashboard_html(dashboard)

    assert "&lt;BREACH&gt;" in html
    assert "&lt;script&gt;alert(1)&lt;/script&gt;" in html
    assert "<script>alert(1)</script>" not in html
