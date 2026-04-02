import pandas as pd

from bigdata.reporting import (
    DEFAULT_LIMITS,
    RiskReport,
    build_daily_risk_summary,
    build_factor_summary,
    build_integrated_risk_report,
    build_top_positions_table,
    monitor_risk_limits,
)


def test_build_daily_risk_summary_returns_one_row(reporting_inputs):
    portfolio_result, risk_result, stress_result, macro_result = reporting_inputs

    summary = build_daily_risk_summary(
        portfolio_result=portfolio_result,
        risk_result=risk_result,
        stress_result=stress_result,
        macro_result=macro_result,
    )

    assert summary.shape[0] == 1
    assert "VaR_hist" in summary.columns
    assert "stress_loss" in summary.columns
    assert "equity_beta" in summary.columns
    assert "worst_day_loss" in summary.columns
    assert "volatility" in summary.columns


def test_build_factor_summary_contains_beta_and_contribution(reporting_inputs):
    _, _, _, macro_result = reporting_inputs

    factor_summary = build_factor_summary(macro_result)

    assert list(factor_summary.columns) == ["beta", "risk_contribution"]
    assert "SPY" in factor_summary.index


def test_build_top_positions_table_returns_largest_weights(reporting_inputs):
    portfolio_result, _, _, _ = reporting_inputs

    top_positions = build_top_positions_table(portfolio_result.rolling_weights, top_n=2)

    assert top_positions.shape[0] == 2
    assert "weight" in top_positions.columns
    assert "abs_weight" in top_positions.columns


def test_monitor_risk_limits_flags_breaches(reporting_inputs):
    portfolio_result, risk_result, stress_result, macro_result = reporting_inputs
    summary = build_daily_risk_summary(
        portfolio_result=portfolio_result,
        risk_result=risk_result,
        stress_result=stress_result,
        macro_result=macro_result,
    )
    factor_summary = build_factor_summary(macro_result)

    limits = DEFAULT_LIMITS | {"var_hist": 0.001}
    limit_check_table, breach_log, alert_summary = monitor_risk_limits(
        summary_table=summary,
        factor_summary_table=factor_summary,
        limits=limits,
    )

    assert "var_hist" in limit_check_table.index
    assert limit_check_table.loc["var_hist", "status"] in {"WARNING", "BREACH"}
    assert isinstance(alert_summary["message"], str)
    assert isinstance(breach_log, pd.DataFrame)


def test_build_integrated_risk_report_returns_all_sections(reporting_inputs):
    portfolio_result, risk_result, stress_result, macro_result = reporting_inputs

    report = build_integrated_risk_report(
        portfolio_result=portfolio_result,
        risk_result=risk_result,
        stress_result=stress_result,
        macro_result=macro_result,
        top_n_positions=2,
    )

    assert isinstance(report, RiskReport)
    assert report.summary_table.shape[0] == 1
    assert report.factor_summary_table.shape[0] >= 1
    assert report.top_positions_table.shape[0] == 2
    assert "status" in report.limit_check_table.columns
