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
