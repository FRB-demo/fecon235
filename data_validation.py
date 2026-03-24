#  Python Module for import                           Date : 2026-03-24
#  vim: set fileencoding=utf-8 ff=unix tw=78 ai syn=python : per Python PEP 0263
'''
_______________|  data_validation.py : Data quality validation for economic series.

- Validates pandas Series for common data quality issues:
    NaN patterns, duplicate dates, non-monotonic dates, outliers.
- Checks data frequency and gaps.
- Detects stale data.
- Compares two vintages of the same series for revisions.

USAGE
    from fecon235.data_validation import validate_all, validate_series

CHANGE LOG  For latest version, see https://github.com/rsvp/fecon235
2026-03-24  First version with validate_series, validate_frequency,
               check_staleness, detect_revisions, validate_all.
'''

from __future__ import absolute_import, print_function

import pandas as pd
import numpy as np
from dataclasses import dataclass, field


@dataclass
class ValidationResult:
    """Structured result from a validation check."""
    series_name: str
    passed: bool
    warnings: list = field(default_factory=list)
    errors: list = field(default_factory=list)


def _extract_series(data):
    """Extract a pd.Series from data that may be a DataFrame or Series."""
    if isinstance(data, pd.DataFrame):
        if data.shape[1] == 1:
            return data.iloc[:, 0]
        raise ValueError("DataFrame has multiple columns; pass a single-column "
                         "DataFrame or a Series.")
    return data


def validate_series(series, name="unnamed"):
    """Check for unexpected NaN patterns, duplicate dates, non-monotonic
    dates, and suspicious outliers (>5 sigma from 60-day rolling mean).

    Parameters
    ----------
    series : pd.Series or single-column pd.DataFrame
    name   : str, label for the series

    Returns
    -------
    ValidationResult
    """
    series = _extract_series(series)
    warnings = []
    errors = []

    # --- Empty series ---
    if len(series) == 0:
        errors.append("Series is empty (0 observations).")
        return ValidationResult(series_name=name, passed=False,
                                warnings=warnings, errors=errors)

    # --- Single observation ---
    if len(series) == 1:
        warnings.append("Series has only 1 observation.")
        return ValidationResult(series_name=name, passed=True,
                                warnings=warnings, errors=errors)

    # --- Check index is DatetimeIndex ---
    if not isinstance(series.index, pd.DatetimeIndex):
        errors.append("Index is not DatetimeIndex "
                      f"(got {type(series.index).__name__}).")
        return ValidationResult(series_name=name, passed=False,
                                warnings=warnings, errors=errors)

    # --- Duplicate dates ---
    dup_count = series.index.duplicated().sum()
    if dup_count > 0:
        errors.append(f"Found {dup_count} duplicate date(s) in the index.")

    # --- Non-monotonic dates ---
    if not series.index.is_monotonic_increasing:
        errors.append("Date index is not monotonically increasing.")

    # --- NaN ratio ---
    total = len(series)
    nan_count = int(series.isna().sum())
    if nan_count == total:
        errors.append("Series is entirely NaN.")
    elif total > 0:
        nan_ratio = nan_count / total
        if nan_ratio > 0.10:
            warnings.append(
                f"High NaN ratio: {nan_count}/{total} "
                f"({nan_ratio:.1%}) values are NaN.")

    # --- Outlier detection (rolling z-score, window=60) ---
    numeric = series.dropna()
    if len(numeric) > 60:
        rolling_mean = numeric.rolling(window=60, min_periods=30).mean()
        rolling_std = numeric.rolling(window=60, min_periods=30).std()
        z_scores = ((numeric - rolling_mean) / rolling_std).dropna()
        outlier_mask = z_scores.abs() > 5.0
        outlier_count = int(outlier_mask.sum())
        if outlier_count > 0:
            warnings.append(
                f"Detected {outlier_count} outlier(s) "
                f"(>5 sigma from 60-period rolling mean).")

    passed = len(errors) == 0
    return ValidationResult(series_name=name, passed=passed,
                            warnings=warnings, errors=errors)


def validate_frequency(series, expected_freq, name="unnamed"):
    """Verify data is at the expected frequency ('D', 'M', 'Q', 'A').
    Check for unexpected gaps.

    Parameters
    ----------
    series        : pd.Series or single-column pd.DataFrame
    expected_freq : str, one of 'D', 'M', 'Q', 'A'
    name          : str

    Returns
    -------
    ValidationResult
    """
    series = _extract_series(series)
    warnings = []
    errors = []

    if len(series) < 3:
        warnings.append("Too few observations to reliably infer frequency.")
        return ValidationResult(series_name=name, passed=True,
                                warnings=warnings, errors=errors)

    if not isinstance(series.index, pd.DatetimeIndex):
        errors.append("Index is not DatetimeIndex; cannot check frequency.")
        return ValidationResult(series_name=name, passed=False,
                                warnings=warnings, errors=errors)

    inferred = pd.infer_freq(series.index)

    # Map common inferred codes to canonical letters
    _freq_map = {
        'D': 'D', 'B': 'D',  # business-day ~ daily
        'MS': 'M', 'ME': 'M', 'M': 'M', 'BMS': 'M', 'BME': 'M',
        'BM': 'M',
        'QS': 'Q', 'QE': 'Q', 'Q': 'Q', 'QS-OCT': 'Q', 'QE-OCT': 'Q',
        'BQS': 'Q', 'BQE': 'Q', 'BQ': 'Q',
        'AS': 'A', 'AE': 'A', 'A': 'A', 'YS': 'A', 'YE': 'A',
        'BAS': 'A', 'BAE': 'A', 'BA': 'A',
    }

    canonical = None
    if inferred:
        # Strip anchored suffixes like '-JAN' for matching
        base = inferred.split('-')[0] if '-' in inferred else inferred
        canonical = _freq_map.get(base, None)
        if canonical is None:
            canonical = _freq_map.get(inferred, None)

    if inferred is None:
        warnings.append("Could not infer frequency; data may have gaps.")
    elif canonical != expected_freq.upper():
        warnings.append(
            f"Expected frequency '{expected_freq}' but inferred "
            f"'{inferred}' (canonical: '{canonical}').")

    # Detect gaps using expected timedelta ranges
    _expected_deltas = {
        'D': (pd.Timedelta(days=1), pd.Timedelta(days=3)),    # weekends ok
        'M': (pd.Timedelta(days=27), pd.Timedelta(days=32)),
        'Q': (pd.Timedelta(days=89), pd.Timedelta(days=93)),
        'A': (pd.Timedelta(days=364), pd.Timedelta(days=367)),
    }
    freq_key = expected_freq.upper()
    if freq_key in _expected_deltas and len(series) > 1:
        lo, hi = _expected_deltas[freq_key]
        deltas = pd.Series(series.index[1:] - series.index[:-1])
        gaps = deltas[deltas > hi]
        if len(gaps) > 0:
            warnings.append(
                f"Found {len(gaps)} gap(s) exceeding expected "
                f"'{expected_freq}' spacing.")

    passed = len(errors) == 0
    return ValidationResult(series_name=name, passed=passed,
                            warnings=warnings, errors=errors)


def check_staleness(series, max_age_days=90, name="unnamed"):
    """Alert if the most recent observation is older than expected.

    Parameters
    ----------
    series       : pd.Series or single-column pd.DataFrame
    max_age_days : int
    name         : str

    Returns
    -------
    ValidationResult
    """
    series = _extract_series(series)
    warnings = []
    errors = []

    if len(series) == 0:
        errors.append("Series is empty; cannot check staleness.")
        return ValidationResult(series_name=name, passed=False,
                                warnings=warnings, errors=errors)

    if not isinstance(series.index, pd.DatetimeIndex):
        errors.append("Index is not DatetimeIndex; cannot check staleness.")
        return ValidationResult(series_name=name, passed=False,
                                warnings=warnings, errors=errors)

    last_date = series.index[-1]
    today = pd.Timestamp.now()
    age_days = (today - last_date).days

    if age_days > max_age_days:
        warnings.append(
            f"Data is stale: last observation is {age_days} days old "
            f"(threshold: {max_age_days} days). "
            f"Last date: {last_date.strftime('%Y-%m-%d')}.")

    passed = len(errors) == 0
    return ValidationResult(series_name=name, passed=passed,
                            warnings=warnings, errors=errors)


def detect_revisions(series_old, series_new, name="unnamed"):
    """Compare two vintages of the same series and highlight revisions.

    Parameters
    ----------
    series_old : pd.Series or single-column pd.DataFrame
    series_new : pd.Series or single-column pd.DataFrame
    name       : str

    Returns
    -------
    ValidationResult
    """
    series_old = _extract_series(series_old)
    series_new = _extract_series(series_new)
    warnings = []
    errors = []

    # Align on common dates
    common_idx = series_old.index.intersection(series_new.index)

    if len(common_idx) == 0:
        warnings.append("No overlapping dates between old and new series.")
        return ValidationResult(series_name=name, passed=True,
                                warnings=warnings, errors=errors)

    old_aligned = series_old.loc[common_idx]
    new_aligned = series_new.loc[common_idx]

    # Find values that differ (ignoring NaN == NaN)
    diff_mask = ~(
        (old_aligned == new_aligned) |
        (old_aligned.isna() & new_aligned.isna())
    )
    revision_count = int(diff_mask.sum())

    if revision_count > 0:
        revisions = (new_aligned[diff_mask] - old_aligned[diff_mask]).dropna()
        abs_revisions = revisions.abs()
        mean_rev = float(abs_revisions.mean())
        max_rev = float(abs_revisions.max())
        warnings.append(
            f"Detected {revision_count} revision(s) across "
            f"{len(common_idx)} overlapping dates. "
            f"Mean absolute revision: {mean_rev:.4f}, "
            f"largest: {max_rev:.4f}.")

    passed = len(errors) == 0
    return ValidationResult(series_name=name, passed=passed,
                            warnings=warnings, errors=errors)


def validate_all(series, name="unnamed", expected_freq=None,
                 max_age_days=90):
    """Run all validation checks and return a combined result.

    Parameters
    ----------
    series        : pd.Series or single-column pd.DataFrame
    name          : str
    expected_freq : str or None, one of 'D', 'M', 'Q', 'A'
    max_age_days  : int

    Returns
    -------
    ValidationResult
    """
    all_warnings = []
    all_errors = []

    # 1) Series validation
    r1 = validate_series(series, name=name)
    all_warnings.extend(r1.warnings)
    all_errors.extend(r1.errors)

    # 2) Frequency validation (if requested)
    if expected_freq is not None:
        r2 = validate_frequency(series, expected_freq, name=name)
        all_warnings.extend(r2.warnings)
        all_errors.extend(r2.errors)

    # 3) Staleness check
    r3 = check_staleness(series, max_age_days=max_age_days, name=name)
    all_warnings.extend(r3.warnings)
    all_errors.extend(r3.errors)

    passed = len(all_errors) == 0
    return ValidationResult(series_name=name, passed=passed,
                            warnings=all_warnings, errors=all_errors)
