#  Python Module for import                           Date : 2026-03-24
#  vim: set fileencoding=utf-8 ff=unix tw=78 ai syn=python : per Python PEP 0263
'''
_______________|  test_data_validation : Test fecon235 data_validation module.

Testing: As of fecon235 v4, we favor pytest over nosetests, so e.g.
    $ py.test tests/test_data_validation.py -v

CHANGE LOG  For latest version, see https://github.com/rsvp/fecon235
2026-03-24  First version.
'''

from __future__ import absolute_import, print_function

import pytest
import pandas as pd
import numpy as np
from fecon235.data_validation import (
    validate_series,
    validate_frequency,
    check_staleness,
    detect_revisions,
    validate_all,
    ValidationResult,
)


# ---------------------------------------------------------------------------
#  Helper to create a clean monthly series
# ---------------------------------------------------------------------------

def _make_monthly_series(start="2020-01-01", periods=120, seed=42):
    """Return a clean monthly pd.Series with no issues."""
    rng = np.random.RandomState(seed)
    idx = pd.date_range(start, periods=periods, freq="MS")
    values = 100 + rng.randn(periods).cumsum()
    return pd.Series(values, index=idx, name="test")


def _make_quarterly_series(start="2010-01-01", periods=40, seed=42):
    """Return a clean quarterly pd.Series."""
    rng = np.random.RandomState(seed)
    idx = pd.date_range(start, periods=periods, freq="QS")
    values = 100 + rng.randn(periods).cumsum()
    return pd.Series(values, index=idx, name="test")


# ===================================================================
#  TestValidateSeries
# ===================================================================

class TestValidateSeries:

    def test_clean_series_passes(self):
        """Monthly series with no issues should pass."""
        s = _make_monthly_series()
        result = validate_series(s, name="clean")
        assert result.passed is True
        assert result.errors == []
        assert result.series_name == "clean"

    def test_duplicate_dates_detected(self):
        """Series with duplicate index entries should report an error."""
        idx = pd.to_datetime(["2020-01-01", "2020-01-01", "2020-02-01",
                              "2020-03-01"])
        s = pd.Series([1.0, 2.0, 3.0, 4.0], index=idx)
        result = validate_series(s, name="dup")
        assert result.passed is False
        assert any("duplicate" in e.lower() for e in result.errors)

    def test_non_monotonic_dates_detected(self):
        """Series with out-of-order dates should report an error."""
        idx = pd.to_datetime(["2020-03-01", "2020-01-01", "2020-02-01"])
        s = pd.Series([1.0, 2.0, 3.0], index=idx)
        result = validate_series(s, name="nonmono")
        assert result.passed is False
        assert any("monotonic" in e.lower() for e in result.errors)

    def test_outlier_detected(self):
        """Series with a 10-sigma spike should flag an outlier."""
        s = _make_monthly_series(periods=120)
        # Inject a massive outlier at position 100
        std_val = s.std()
        s.iloc[100] = s.iloc[99] + 10 * std_val
        result = validate_series(s, name="outlier")
        assert any("outlier" in w.lower() for w in result.warnings)

    def test_high_nan_ratio_warned(self):
        """Series with >10% NaN values should produce a warning."""
        s = _make_monthly_series(periods=100)
        # Set 15 values to NaN (15%)
        s.iloc[10:25] = np.nan
        result = validate_series(s, name="nans")
        assert any("nan" in w.lower() for w in result.warnings)

    def test_empty_series(self):
        """Empty pd.Series should fail with an error."""
        s = pd.Series(dtype=float)
        result = validate_series(s, name="empty")
        assert result.passed is False
        assert any("empty" in e.lower() for e in result.errors)

    def test_single_observation(self):
        """Series with one data point should pass with a warning."""
        idx = pd.to_datetime(["2020-01-01"])
        s = pd.Series([42.0], index=idx)
        result = validate_series(s, name="single")
        assert result.passed is True
        assert any("1 observation" in w for w in result.warnings)

    def test_all_nan_series(self):
        """Series that is entirely NaN should report an error."""
        idx = pd.date_range("2020-01-01", periods=10, freq="MS")
        s = pd.Series([np.nan] * 10, index=idx)
        result = validate_series(s, name="allnan")
        assert result.passed is False
        assert any("entirely nan" in e.lower() for e in result.errors)

    def test_dataframe_input(self):
        """Single-column DataFrame should be handled transparently."""
        s = _make_monthly_series()
        df = s.to_frame(name="Y")
        result = validate_series(df, name="df_input")
        assert result.passed is True


# ===================================================================
#  TestValidateFrequency
# ===================================================================

class TestValidateFrequency:

    def test_monthly_detected(self):
        """Monthly series should match expected_freq='M'."""
        s = _make_monthly_series()
        result = validate_frequency(s, expected_freq="M", name="monthly")
        assert result.passed is True
        # Should not warn about mismatch
        assert not any("expected frequency" in w.lower()
                       for w in result.warnings)

    def test_quarterly_detected(self):
        """Quarterly series should match expected_freq='Q'."""
        s = _make_quarterly_series()
        result = validate_frequency(s, expected_freq="Q", name="quarterly")
        assert result.passed is True
        assert not any("expected frequency" in w.lower()
                       for w in result.warnings)

    def test_frequency_mismatch_warned(self):
        """Monthly data with expected_freq='Q' should warn."""
        s = _make_monthly_series()
        result = validate_frequency(s, expected_freq="Q", name="mismatch")
        assert any("expected frequency" in w.lower()
                   for w in result.warnings)

    def test_irregular_gaps_detected(self):
        """Series with a large gap should be flagged."""
        idx = pd.date_range("2020-01-01", periods=12, freq="MS")
        # Remove middle entries to create a gap
        idx_with_gap = idx.delete([5, 6, 7, 8])
        s = pd.Series(range(len(idx_with_gap)), index=idx_with_gap,
                      dtype=float)
        result = validate_frequency(s, expected_freq="M", name="gapped")
        has_gap_or_infer_warning = (
            any("gap" in w.lower() for w in result.warnings) or
            any("infer" in w.lower() for w in result.warnings)
        )
        assert has_gap_or_infer_warning

    def test_too_few_observations(self):
        """Fewer than 3 observations should produce a warning."""
        idx = pd.to_datetime(["2020-01-01", "2020-02-01"])
        s = pd.Series([1.0, 2.0], index=idx)
        result = validate_frequency(s, expected_freq="M", name="short")
        assert any("too few" in w.lower() for w in result.warnings)


# ===================================================================
#  TestCheckStaleness
# ===================================================================

class TestCheckStaleness:

    def test_fresh_data_passes(self):
        """Data ending recently should pass."""
        idx = pd.date_range(end=pd.Timestamp.now(), periods=30, freq="D")
        s = pd.Series(range(30), index=idx, dtype=float)
        result = check_staleness(s, max_age_days=90, name="fresh")
        assert result.passed is True
        assert not any("stale" in w.lower() for w in result.warnings)

    def test_stale_data_warned(self):
        """Data ending >90 days ago should warn about staleness."""
        end = pd.Timestamp.now() - pd.Timedelta(days=180)
        idx = pd.date_range(end=end, periods=30, freq="D")
        s = pd.Series(range(30), index=idx, dtype=float)
        result = check_staleness(s, max_age_days=90, name="stale")
        assert result.passed is True  # staleness is a warning, not error
        assert any("stale" in w.lower() for w in result.warnings)

    def test_empty_series_errors(self):
        """Empty series should error for staleness check."""
        s = pd.Series(dtype=float)
        result = check_staleness(s, name="empty")
        assert result.passed is False


# ===================================================================
#  TestDetectRevisions
# ===================================================================

class TestDetectRevisions:

    def test_no_revisions(self):
        """Identical series should show no revisions."""
        s = _make_monthly_series(periods=24)
        result = detect_revisions(s, s.copy(), name="same")
        assert result.passed is True
        assert not any("revision" in w.lower() for w in result.warnings)

    def test_revisions_detected(self):
        """Modified values should be detected as revisions."""
        s_old = _make_monthly_series(periods=24)
        s_new = s_old.copy()
        s_new.iloc[5] += 0.5
        s_new.iloc[10] -= 1.0
        result = detect_revisions(s_old, s_new, name="revised")
        assert any("2 revision" in w.lower() for w in result.warnings)

    def test_mismatched_dates_handled(self):
        """Series with partially overlapping dates should work."""
        idx_old = pd.date_range("2020-01-01", periods=12, freq="MS")
        idx_new = pd.date_range("2020-07-01", periods=12, freq="MS")
        s_old = pd.Series(range(12), index=idx_old, dtype=float)
        s_new = pd.Series(range(12), index=idx_new, dtype=float)
        result = detect_revisions(s_old, s_new, name="partial")
        # Should handle gracefully; overlapping dates have same values
        assert result.passed is True

    def test_no_overlap(self):
        """Series with no overlapping dates should warn."""
        idx_old = pd.date_range("2019-01-01", periods=6, freq="MS")
        idx_new = pd.date_range("2020-01-01", periods=6, freq="MS")
        s_old = pd.Series(range(6), index=idx_old, dtype=float)
        s_new = pd.Series(range(6), index=idx_new, dtype=float)
        result = detect_revisions(s_old, s_new, name="no_overlap")
        assert any("no overlapping" in w.lower() for w in result.warnings)


# ===================================================================
#  TestValidateAll
# ===================================================================

class TestValidateAll:

    def test_clean_series_all_pass(self):
        """Clean monthly series should pass all checks."""
        end = pd.Timestamp.now()
        idx = pd.date_range(end=end, periods=120, freq="MS")
        rng = np.random.RandomState(42)
        s = pd.Series(100 + rng.randn(len(idx)).cumsum(), index=idx)
        result = validate_all(s, name="all_clean", expected_freq="M",
                              max_age_days=90)
        assert result.passed is True
        assert result.errors == []

    def test_multiple_issues_combined(self):
        """validate_all should combine warnings and errors."""
        # Entirely NaN series => error from validate_series,
        # also staleness warning
        end = pd.Timestamp.now() - pd.Timedelta(days=200)
        idx = pd.date_range(end=end, periods=10, freq="MS")
        s = pd.Series([np.nan] * len(idx), index=idx)
        result = validate_all(s, name="multi", expected_freq="M",
                              max_age_days=90)
        assert result.passed is False
        assert len(result.errors) > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
