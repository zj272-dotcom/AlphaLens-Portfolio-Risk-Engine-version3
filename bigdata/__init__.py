"""Core package for the new portfolio risk engine project."""

from .data_pipeline import (
    DataPipelineResult,
    RollingWindow,
    align_and_clean_prices,
    build_data_pipeline,
    compute_clean_return_matrix,
    create_rolling_windows,
    download_adjusted_close_prices,
)
from .dashboard import Dashboard, build_daily_risk_dashboard, render_dashboard_html
from .portfolio_construction import (
    PortfolioConstructionResult,
    compute_equal_weight,
    compute_min_variance_weights,
    compute_turnover,
    estimate_covariance_matrix,
    run_rolling_portfolio_construction,
    shrink_covariance,
)
from .macro_factor import (
    MacroFactorRiskResult,
    compute_factor_covariance,
    estimate_factor_beta,
    run_macro_factor_risk_decomposition,
)
from .risk_engine import (
    RiskEngineResult,
    build_pnl_distribution,
    historical_var_es,
    monte_carlo_var_es,
    parametric_var_es,
    run_rolling_risk_engine,
)
from .reporting import (
    DEFAULT_LIMITS,
    RiskReport,
    build_daily_risk_summary,
    build_factor_summary,
    build_integrated_risk_report,
    build_top_positions_table,
    monitor_risk_limits,
)
from .stress_testing import (
    DEFAULT_HISTORICAL_SCENARIOS,
    StressTestResult,
    aggregate_period_returns,
    calculate_stress_loss,
    find_worst_historical_periods,
    run_historical_stress_scenarios,
)
from .universe import ASSET_CLASS_MAPPING, ASSET_LIST, ASSET_UNIVERSE

__all__ = [
    "ASSET_LIST",
    "ASSET_CLASS_MAPPING",
    "ASSET_UNIVERSE",
    "download_adjusted_close_prices",
    "align_and_clean_prices",
    "compute_clean_return_matrix",
    "create_rolling_windows",
    "build_data_pipeline",
    "RollingWindow",
    "DataPipelineResult",
    "build_daily_risk_dashboard",
    "render_dashboard_html",
    "Dashboard",
    "estimate_covariance_matrix",
    "shrink_covariance",
    "compute_equal_weight",
    "compute_min_variance_weights",
    "compute_turnover",
    "run_rolling_portfolio_construction",
    "PortfolioConstructionResult",
    "estimate_factor_beta",
    "compute_factor_covariance",
    "run_macro_factor_risk_decomposition",
    "MacroFactorRiskResult",
    "historical_var_es",
    "parametric_var_es",
    "monte_carlo_var_es",
    "build_pnl_distribution",
    "run_rolling_risk_engine",
    "RiskEngineResult",
    "DEFAULT_LIMITS",
    "build_daily_risk_summary",
    "build_factor_summary",
    "build_top_positions_table",
    "monitor_risk_limits",
    "build_integrated_risk_report",
    "RiskReport",
    "DEFAULT_HISTORICAL_SCENARIOS",
    "aggregate_period_returns",
    "calculate_stress_loss",
    "run_historical_stress_scenarios",
    "find_worst_historical_periods",
    "StressTestResult",
]
