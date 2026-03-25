# Windsurf Prompts for fecon235

> This file contains ready-to-use prompts for [Windsurf](https://windsurf.com), Cognition's AI-powered IDE. Open this repo in Windsurf, pick a prompt below, and paste it into Cascade. Each prompt includes context on what problem it solves and what Windsurf will do.
>
> Prompts are ordered from simple → complex so you can start easy and build confidence.

---

## Prompt 1 — Explore the Notebook Collection and Find What You Need (Beginner)
**What this solves:** fecon235 has dozens of Jupyter notebooks covering different financial economics topics — inflation, GDP, interest rates, forecasting, and more. It's hard to know which notebook does what and where to start.

**What Windsurf will do:** Catalog every notebook, summarize what each one analyzes, and recommend the best starting points based on your interests.

**Paste this into Cascade:**
```
This repo has a collection of Jupyter notebooks for financial economics. I need a catalog:

1. List every .ipynb notebook file in the repo with:
   - File path
   - Title (from the first heading or cell)
   - One-sentence summary of what it analyzes
   - Key FRED data series it uses (e.g., GDP, CPIAUCSL, UNRATE)
   - Last meaningful update date

2. Group the notebooks by topic:
   - Inflation & prices
   - GDP & economic growth
   - Labor market & employment
   - Interest rates & monetary policy
   - Financial markets & forecasting
   - Other

3. For someone who wants to understand recent inflation trends, which notebook should they start with? Which one is best for understanding Fed policy?

4. Are there any notebooks that are broken or use deprecated APIs? Flag any that might not run on current Python/pandas versions.

Format this as a table I can reference quickly.
```

---

## Prompt 2 — Fix a Broken Notebook (Intermediate)
**What this solves:** Many notebooks were written for older versions of pandas, matplotlib, and the FRED data access libraries. They throw deprecation warnings or outright errors on current Python environments. You need them working for a presentation tomorrow.

**What Windsurf will do:** Pick a specific notebook, run through it cell by cell, find every error and deprecation, and fix them all while preserving the analysis.

**Paste this into Cascade:**
```
I need to fix the notebooks in this repo to work with current Python (3.11+), pandas (2.x), and matplotlib (3.x). Start with the most important/popular notebook you can find (likely related to inflation, GDP, or Fed funds rate).

For that notebook:

1. Read every cell and identify issues:
   - Deprecated pandas APIs: `.append()`, `pd.datetime`, `inplace=True` patterns
   - Deprecated matplotlib APIs: old-style plotting syntax
   - Broken data access: URLs that changed, APIs that moved
   - Missing imports or undefined variables
   - Any hardcoded date ranges that should be updated to include recent data

2. Fix each issue:
   - Replace `.append()` with `pd.concat()`
   - Replace `pd.datetime` with `pd.Timestamp` or `datetime.datetime`
   - Update any broken data URLs or API endpoints
   - Add missing imports
   - Update date ranges to include data through 2024

3. Verify the notebook runs end-to-end without errors or warnings

4. Add a "Last updated" cell at the top noting the Python and pandas versions tested

Show me every change with before/after so I understand what was broken.
```

---

## Prompt 3 — Create a New Analysis: Fed Funds Rate vs Inflation Dashboard (Advanced)
**What this solves:** One of the most important questions in economics is the relationship between Fed policy (interest rates) and inflation. There's no notebook in this collection that directly visualizes this relationship with current data. You need one for tomorrow's presentation.

**What Windsurf will do:** Build a complete, polished Jupyter notebook from scratch — data retrieval, analysis, and publication-quality visualizations.

**Paste this into Cascade:**
```
Create a new Jupyter notebook: notebooks/fed_policy_vs_inflation.ipynb

This notebook should analyze the relationship between Federal Reserve monetary policy and inflation. Make it presentation-quality — this will be shown to senior economists.

Sections:

1. **Data Collection** (use fredapi or direct FRED API):
   - Federal Funds Rate (FEDFUNDS) — monthly
   - CPI Year-over-Year (CPIAUCSL) — monthly
   - Core PCE Year-over-Year (PCEPILFE) — monthly, the Fed's preferred inflation measure
   - 10-Year Breakeven Inflation (T10YIE) — market inflation expectations
   - Date range: 2000 to present

2. **Visualization 1 — The Dual Mandate Dashboard**:
   - Dual-axis time series: Fed Funds Rate on left axis, inflation (CPI and PCE) on right axis
   - Shade recession periods (use FRED USREC series)
   - Annotate key events: 2008 crisis, COVID, 2022 rate hikes
   - Use a clean, professional color scheme (not default matplotlib)

3. **Visualization 2 — Taylor Rule Comparison**:
   - Calculate the Taylor Rule implied rate: r = r* + 0.5*(inflation - 2%) + 0.5*(output_gap)
   - Plot actual Fed Funds Rate vs Taylor Rule implied rate
   - Highlight periods where the Fed was "too loose" or "too tight" vs the rule

4. **Visualization 3 — Rate Hike Cycles**:
   - Identify the last 4 rate hike cycles from the data
   - For each cycle: start date, end date, total rate increase, how long it lasted
   - Show how inflation responded during and after each cycle (lag analysis)

5. **Key Takeaways** (markdown cell):
   - Summary statistics and observations
   - What the current data suggests about the inflation-policy relationship

Use matplotlib with a professional theme (seaborn-v0_8-whitegrid or similar). Add axis labels, titles, legends, and source citations. Every chart should be presentation-ready.
```

---

## Prompt 4 — Build a Reusable Data Pipeline Module (Expert)
**What this solves:** Every notebook in this repo independently fetches and processes FRED data, leading to duplicated code, inconsistent data handling, and no caching. When the FRED API is slow or rate-limited, all notebooks suffer. You need a shared data layer.

**What Windsurf will do:** Design and build a reusable Python module that handles data fetching, caching, validation, and common transformations — so all notebooks can import it instead of reinventing the wheel.

**Paste this into Cascade:**
```
Create a reusable data pipeline module for this repo: fecon235/data_pipeline.py (or similar path following repo conventions)

This module should solve the common problems across all notebooks in this repo:

1. **FREDDataManager class**:
   - `__init__(api_key=None, cache_dir=".fred_cache")`: Initialize with optional API key (falls back to environment variable), set up local file cache
   - `get_series(series_id, start_date=None, end_date=None, transform=None)`: Fetch a series, automatically cache to local parquet files with TTL (default 24 hours for monthly data, 1 hour for daily)
   - `get_multiple_series(series_ids, ...)`: Fetch multiple series in parallel, return aligned DataFrame
   - `clear_cache()`: Clear all cached data

2. **Built-in transformations**:
   - `yoy_change(series)`: Year-over-year percent change (most common for CPI, GDP)
   - `mom_change(series)`: Month-over-month percent change
   - `rolling_average(series, window)`: Smoothed series
   - `normalize(series, base_date)`: Index to 100 at a base date
   - `real_values(nominal_series, cpi_series)`: Deflate nominal to real values

3. **Data validation**:
   - Check for gaps, staleness, NaN patterns
   - Warn if series frequency doesn't match expected (e.g., got annual when expected monthly)
   - Validate date ranges

4. **Pre-built data bundles** (convenience functions):
   - `get_inflation_data()`: Returns DataFrame with CPI, Core CPI, PCE, Core PCE, PPI
   - `get_labor_data()`: Returns DataFrame with unemployment, nonfarm payrolls, participation rate
   - `get_rates_data()`: Returns DataFrame with Fed Funds, 2Y/10Y/30Y Treasury, mortgage rates
   - `get_gdp_data()`: Returns DataFrame with real GDP, nominal GDP, GDP deflator

5. **Tests**: Add tests using mocked FRED responses — the module should be testable without an API key

Add a short example notebook (notebooks/using_data_pipeline.ipynb) showing how notebooks can use this module instead of raw API calls. Show the before/after: 10 lines of boilerplate reduced to 2 lines.
```

---
