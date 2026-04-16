# Portfolio Risk Engine

`bigdata` is a Python package for multi-asset portfolio risk analysis. It builds
an end-to-end workflow for:

- defining a liquid ETF universe
- cleaning and aligning return data
- constructing rolling portfolios without look-ahead bias
- estimating VaR / ES with multiple methods
- replaying historical stress scenarios
- decomposing risk into macro-factor and residual components
- assembling daily risk reports and a lightweight dashboard

The project is organized as a reusable Python package with unit tests, notebook
examples, and a structure that can be extended as new analytics are added.

## Purpose

The goal of this project is to provide a practical risk-engine template for
portfolio monitoring. It is designed to answer questions like:

- What is the portfolio's current daily risk?
- How much risk is systematic versus idiosyncratic?
- What happens under historical crisis conditions?
- Which positions and exposures are driving the current risk state?
- Are any internal risk limits in warning or breach territory?

The implementation prioritizes:

- numerical stability
- no look-ahead bias
- realistic portfolio constraints
- clear, structured outputs for monitoring and reporting

## Dataset

The current version uses liquid ETF proxies from Yahoo Finance via `yfinance`.

### Universe

- Equities: `SPY`, `QQQ`, `IWM`, `EEM`
- Fixed Income: `TLT`, `IEF`
- Commodities: `GLD`, `USO`
- FX Proxy: `UUP`

### Data Used

- Daily adjusted close prices
- Daily simple returns computed from cleaned adjusted close data
- Historical scenario windows for stress testing:
  - `GFC`: `2008-09-01` to `2009-03-31`
  - `COVID`: `2020-02-20` to `2020-03-23`

### Default Sample Design

- Default start date: `2020-01-01`
- Unified business-day calendar
- Forward fill for internal missing observations
- Removal of leading missing observations that cannot be filled
- Rolling windows:
  - training: `252` trading days
  - test: `21` trading days

## Package Setup

This project is set up as a Python package.

Core package directory:

- [`bigdata/`](./bigdata)

Main packaging/config files:

- [`pyproject.toml`](./pyproject.toml)
- [`conda.yaml`](./conda.yaml)

## Installation

### Option 1: Pip

Install the package in editable mode with test dependencies:

```bash
pip install -e ".[test]"
```

If you also want notebook support:

```bash
pip install -e ".[test,notebook]"
```

### Option 2: Conda

Create the environment from the provided file:

```bash
conda env create -f conda.yaml
conda activate bigdata
```

## Importing the Package

You can import the main modules directly from the package:

```python
from bigdata import (
    ASSET_LIST,
    build_data_pipeline,
    run_rolling_portfolio_construction,
    run_rolling_risk_engine,
    run_historical_stress_scenarios,
    run_macro_factor_risk_decomposition,
    build_integrated_risk_report,
    build_daily_risk_dashboard,
)
```

## Quick Start

### 1. Build the Data Pipeline

```python
from bigdata import ASSET_LIST, build_data_pipeline

pipeline_result = build_data_pipeline(
    tickers=ASSET_LIST,
    start_date="2020-01-01",
)
```

### 2. Construct Rolling Portfolio Weights

```python
from bigdata import run_rolling_portfolio_construction

portfolio_result = run_rolling_portfolio_construction(
    windows=pipeline_result.training_test_windows,
    strategy="min_variance",
)
```

### 3. Run the Risk Engine

```python
from bigdata import run_rolling_risk_engine

risk_result = run_rolling_risk_engine(
    portfolio_returns=portfolio_result.portfolio_returns,
)
```

### 4. Run Stress Testing

```python
from bigdata import run_historical_stress_scenarios

stress_result = run_historical_stress_scenarios(
    asset_returns=pipeline_result.clean_return_matrix,
    current_weights=portfolio_result.rolling_weights.iloc[-1],
)
```

### 5. Run Macro Factor Decomposition

```python
from bigdata import run_macro_factor_risk_decomposition

macro_result = run_macro_factor_risk_decomposition(
    portfolio_returns=portfolio_result.portfolio_returns,
    asset_returns=pipeline_result.clean_return_matrix,
    factor_set=["SPY", "TLT", "GLD", "UUP"],
)
```

### 6. Build the Daily Report and Dashboard

```python
from bigdata import build_integrated_risk_report, build_daily_risk_dashboard

report = build_integrated_risk_report(
    portfolio_result=portfolio_result,
    risk_result=risk_result,
    stress_result=stress_result,
    macro_result=macro_result,
)

dashboard = build_daily_risk_dashboard(
    risk_summary_table=report.summary_table,
    factor_summary_table=report.factor_summary_table,
    top_positions_table=report.top_positions_table,
    limit_check_table=report.limit_check_table,
    alert_summary=report.alert_summary,
)
```

## Running Useful Scripts and Examples

This repository currently provides a notebook-based walkthrough rather than a
CLI script.

### Example Notebook

The main example is:

- [`notebooks/portfolio_risk_engine_example.ipynb`](./notebooks/portfolio_risk_engine_example.ipynb)

Run it with:

```bash
jupyter lab
```

Then open `notebooks/portfolio_risk_engine_example.ipynb`.

The notebook walks through:

- universe definition
- data pipeline
- portfolio construction
- risk engine
- stress testing
- macro factor decomposition
- integrated reporting
- daily dashboard

## Testing

The project includes a dedicated `tests/` directory covering the implemented
pipeline modules.

Run the test suite with:

```bash
pytest
```

Install test dependencies with:

```bash
pip install -e ".[test]"
```

Current test modules:

- [`tests/test_data_pipeline.py`](./tests/test_data_pipeline.py)
- [`tests/test_portfolio_construction.py`](./tests/test_portfolio_construction.py)
- [`tests/test_risk_engine.py`](./tests/test_risk_engine.py)
- [`tests/test_stress_testing.py`](./tests/test_stress_testing.py)
- [`tests/test_macro_factor.py`](./tests/test_macro_factor.py)
- [`tests/test_reporting.py`](./tests/test_reporting.py)
- [`tests/test_dashboard.py`](./tests/test_dashboard.py)

## Project Structure

```text
bigdata/
├── bigdata/
│   ├── __init__.py
│   ├── universe.py
│   ├── data_pipeline.py
│   ├── portfolio_construction.py
│   ├── risk_engine.py
│   ├── stress_testing.py
│   ├── macro_factor.py
│   ├── reporting.py
│   └── dashboard.py
├── notebooks/
│   └── portfolio_risk_engine_example.ipynb
├── tests/
│   ├── conftest.py
│   ├── test_data_pipeline.py
│   ├── test_portfolio_construction.py
│   ├── test_risk_engine.py
│   ├── test_stress_testing.py
│   ├── test_macro_factor.py
│   ├── test_reporting.py
│   └── test_dashboard.py
├── pyproject.toml
├── conda.yaml
└── README.md
```

## Documentation Notes

Important design choices in this package:

- Portfolio construction uses rolling training windows only.
- Risk estimation uses trailing data only.
- Stress testing uses current weights and compounded scenario returns.
- Macro decomposition uses user-selected ETF factors instead of Fama-French factors.
- Dashboard and reporting layers do not perform new risk calculations; they only summarize existing outputs.

## Current Status

The package is ready for repository upload as a documented Python project with:

- package structure
- installation instructions
- import examples
- notebook walkthrough
- unit tests across all implemented modules
