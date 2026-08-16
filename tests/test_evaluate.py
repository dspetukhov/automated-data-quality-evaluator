"""Baseline regression tests for evaluate.py — captures CURRENT behavior as-is."""

import polars as pl
import pytest

from evaluate import evaluate_data, evaluate_data_outliers


# ---------------------------------------------------------------------------
# evaluate_data_outliers
# ---------------------------------------------------------------------------


class TestEvaluateDataOutliers:
    def _stats(self, values):
        s = pl.Series(values)
        return s, s.mean(), s.std(), s.quantile(0.25), s.quantile(0.75)

    def test_iqr_and_zscore_counts_with_custom_config(self):
        data, mean, std, q1, q3 = self._stats([1, 2, 3, 4, 5, 6, 7, 8, 9, 100])
        # mean=14.5, std≈30.152, q1=3.0, q3=8.0
        config = {"criterion": "IQR", "multiplier_iqr": 1.5, "threshold_z_score": 2.0}
        outliers_iqr, outliers_zscore, bounds = evaluate_data_outliers(
            data, mean, std, q1, q3, config
        )
        assert outliers_iqr == 1  # only 100 is outside [-4.5, 15.5]
        assert outliers_zscore == 1  # only 100 has |z| > 2.0 (z≈2.836)
        assert bounds == pytest.approx((-4.5, 15.5))

    def test_defaults_used_when_config_missing_keys(self):
        data, mean, std, q1, q3 = self._stats([1, 2, 3, 4, 5, 100])
        # defaults: multiplier_iqr=1.5, threshold_z_score=3.0
        outliers_iqr, outliers_zscore, bounds = evaluate_data_outliers(
            data, mean, std, q1, q3, {}
        )
        assert outliers_iqr == 1  # bounds [-2.5, 9.5], 100 is outside
        assert outliers_zscore == 0  # z for 100 ≈ 2.04, below default threshold 3.0
        assert bounds == (None, None)  # no criterion specified

    def test_criterion_zscore_bounds(self):
        data, mean, std, q1, q3 = self._stats([1, 2, 3, 4, 5, 6, 7, 8, 9, 100])
        config = {"criterion": "Z-score", "threshold_z_score": 2.0}
        _, _, bounds = evaluate_data_outliers(data, mean, std, q1, q3, config)
        assert bounds == pytest.approx((mean - 2.0 * std, mean + 2.0 * std))

    def test_criterion_iqr_bounds(self):
        data, mean, std, q1, q3 = self._stats([1, 2, 3, 4, 5, 6, 7, 8, 9, 100])
        config = {"criterion": "IQR", "multiplier_iqr": 1.5}
        _, _, bounds = evaluate_data_outliers(data, mean, std, q1, q3, config)
        assert bounds == pytest.approx((3.0 - 1.5 * 5, 8.0 + 1.5 * 5))

    def test_unknown_criterion_gives_none_bounds(self):
        data, mean, std, q1, q3 = self._stats([1, 2, 3])
        config = {"criterion": "something-else"}
        _, _, bounds = evaluate_data_outliers(data, mean, std, q1, q3, config)
        assert bounds == (None, None)

    def test_zero_std_short_circuits_zscore_to_zero(self):
        data, mean, std, q1, q3 = self._stats([5, 5, 5, 5])
        assert std == 0
        outliers_iqr, outliers_zscore, bounds = evaluate_data_outliers(
            data, mean, std, q1, q3, {"criterion": "Z-score"}
        )
        assert outliers_zscore == 0
        assert outliers_iqr == 0
        # BUG-ish: bounds still computed from std=0 -> (mean, mean), collapsing
        # to a single point rather than signaling "no meaningful bounds".
        assert bounds == (mean, mean)

    def test_no_outliers_returns_zero_counts(self):
        data, mean, std, q1, q3 = self._stats([1, 2, 3, 4, 5])
        outliers_iqr, outliers_zscore, _ = evaluate_data_outliers(
            data, mean, std, q1, q3, {}
        )
        assert outliers_iqr == 0
        assert outliers_zscore == 0

    def test_none_std_treated_as_zero(self):
        # FIXED: with a single non-null value, Polars std() returns None.
        # evaluate_data_outliers now normalizes None -> 0 up front, so this
        # is handled the same intentional way as an explicit zero std,
        # rather than relying on Polars null-propagation to accidentally
        # land on 0 (see test_none_std_with_zscore_criterion_no_longer_crashes
        # for the case this used to crash on).
        data, mean, std, q1, q3 = self._stats([5])
        assert std is None
        outliers_iqr, outliers_zscore, bounds = evaluate_data_outliers(
            data, mean, std, q1, q3, {}
        )
        assert outliers_zscore == 0
        assert outliers_iqr == 0
        assert bounds == (None, None)

    def test_none_std_with_zscore_criterion_no_longer_crashes(self):
        # FIXED: previously `mean - threshold * std` with std=None raised
        # TypeError here, which (when called from evaluate_data) was
        # silently swallowed by @exception_handler(), returning the raw
        # input DataFrame instead of the documented tuple. Now std=None is
        # normalized to 0 before bounds are computed, collapsing to a
        # single point (mean, mean) -- consistent with the explicit
        # std == 0 case in test_zero_std_short_circuits_zscore_to_zero.
        data, mean, std, q1, q3 = self._stats([5])
        assert std is None
        outliers_iqr, outliers_zscore, bounds = evaluate_data_outliers(
            data, mean, std, q1, q3, {"criterion": "Z-score", "threshold_z_score": 3.0}
        )
        assert outliers_zscore == 0
        assert outliers_iqr == 0
        assert bounds == (mean, mean)

    def test_all_null_quartiles_return_none_bounds_without_crashing(self):
        # FIXED: an all-null column yields q1 = q3 = None. Previously
        # `q1 - multiplier * (q3 - q1)` raised TypeError; now this is
        # guarded and short-circuits to (0, 0, (None, None)).
        data = pl.Series([None, None, None], dtype=pl.Int64)
        outliers_iqr, outliers_zscore, bounds = evaluate_data_outliers(
            data, None, None, None, None, {"criterion": "IQR"}
        )
        assert outliers_iqr == 0
        assert outliers_zscore == 0
        assert bounds == (None, None)


# ---------------------------------------------------------------------------
# evaluate_data
# ---------------------------------------------------------------------------


class TestEvaluateData:
    def test_first_column_skipped_as_time_interval(self):
        df = pl.DataFrame({
            "__time_interval": [1, 2, 3],
            "__ a __Number of unique values": [1, 2, 3],
        })
        data_evals, outliers_bounds = evaluate_data(df, {})
        assert len(data_evals) == 1
        assert len(outliers_bounds) == 1

    def test_only_time_interval_column_returns_empty_results(self):
        df = pl.DataFrame({"__time_interval": [1, 2, 3]})
        data_evals, outliers_bounds = evaluate_data(df, {})
        assert data_evals == []
        assert outliers_bounds == []

    def test_title_extracted_from_last_segment_after_double_space_underscore(self):
        df = pl.DataFrame({
            "__time_interval": [1, 2, 3],
            "e__ my_col __Standard deviation": [1, 2, 3],
        })
        data_evals, _ = evaluate_data(df, {})
        assert data_evals[0]["title"] == "Standard deviation"

    def test_title_falls_back_to_full_name_without_marker(self):
        df = pl.DataFrame({
            "__time_interval": [1, 2, 3],
            "plain_column": [1, 2, 3],
        })
        data_evals, _ = evaluate_data(df, {})
        assert data_evals[0]["title"] == "plain_column"

    def test_full_column_stats_computed(self):
        df = pl.DataFrame({
            "__time_interval": [1, 2, 3, 4, 5, 6],
            "col": [1, 2, 3, 4, 5, 100],
        })
        data_evals, outliers_bounds = evaluate_data(df, {"multiplier_iqr": 1.5})
        entry = data_evals[0]
        s = pl.Series([1, 2, 3, 4, 5, 100])
        assert entry["μ±σ"] == pytest.approx((s.mean(), s.std()))
        assert entry["Range [Min]"] == 1
        assert entry["Range [Max]"] == 100
        assert entry["Range"] == 99
        assert entry["IQR [Q1]"] == s.quantile(0.25)
        assert entry["IQR [Q3]"] == s.quantile(0.75)
        assert entry["IQR"] == s.quantile(0.75) - s.quantile(0.25)
        # 1 outlier out of 6 rows -> 100/6 %
        assert entry["Outliers [IQR]"] == pytest.approx(100 * 1 / 6)
        assert len(outliers_bounds) == 1

    def test_outliers_percentage_uses_non_null_count_not_total_rows(self):
        # FIXED: percentages used to divide by data.shape[0] (total rows),
        # silently diluting the outlier percentage whenever the column had
        # nulls -- nulls were excluded from the outlier count (numerator)
        # but still counted in the denominator. Now the denominator is the
        # column's own non-null count.
        df = pl.DataFrame({
            "__time_interval": [1, 2, 3, 4, 5, 6, 7, 8],
            "col": [1, 2, 3, 4, 5, None, None, 100],
        })
        data_evals, _ = evaluate_data(df, {"multiplier_iqr": 1.5})
        entry = data_evals[0]
        # 1 outlier (100) out of 6 non-null values, not 8 total rows
        assert entry["Outliers [IQR]"] == pytest.approx(100 * 1 / 6)

    def test_multiple_columns_each_produce_one_eval_and_bound(self):
        df = pl.DataFrame({
            "__time_interval": [1, 2, 3],
            "a": [1, 2, 3],
            "b": [10, 20, 30],
        })
        data_evals, outliers_bounds = evaluate_data(df, {})
        assert len(data_evals) == 2
        assert len(outliers_bounds) == 2
        assert [e["title"] for e in data_evals] == ["a", "b"]

    def test_empty_dataframe_zero_rows_returns_none_stats_without_crashing(self):
        # FIXED: with 0 rows, mean/std/q1/q3/min/max are all None (same
        # degenerate case as an all-null column). This used to raise
        # ZeroDivisionError (percentage calc) or TypeError (Range/IQR
        # arithmetic on None), silently swallowed by @exception_handler(),
        # returning the raw input DataFrame instead of the documented
        # tuple[list, list]. Now it returns a proper eval entry with None
        # stats and 0% outliers instead of crashing.
        df = pl.DataFrame({
            "__time_interval": pl.Series([], dtype=pl.Int64),
            "a": pl.Series([], dtype=pl.Int64),
        })
        data_evals, outliers_bounds = evaluate_data(df, {})
        assert data_evals == [
            {
                "title": "a",
                "μ±σ": (None, None),
                "Range [Min]": None,
                "Range [Max]": None,
                "Range": None,
                "IQR [Q1]": None,
                "IQR [Q3]": None,
                "IQR": None,
                "Outliers [IQR]": 0.0,
                "Outliers [Z-score]": 0.0,
            }
        ]
        assert outliers_bounds == [(None, None)]

    def test_all_null_column_returns_none_stats_without_crashing(self):
        # FIXED: an all-null column yields mean/std/q1/q3/min/max = None.
        # The IQR bound computation `q1 - multiplier * (q3 - q1)` used to
        # raise TypeError on None arithmetic, silently swallowed by
        # @exception_handler(), returning the raw input DataFrame instead
        # of the documented (list, list) tuple. Now it returns a proper
        # eval entry with None stats and 0% outliers instead of crashing.
        df = pl.DataFrame({
            "__time_interval": [1, 2, 3],
            "a": pl.Series([None, None, None], dtype=pl.Int64),
        })
        data_evals, outliers_bounds = evaluate_data(df, {})
        assert data_evals == [
            {
                "title": "a",
                "μ±σ": (None, None),
                "Range [Min]": None,
                "Range [Max]": None,
                "Range": None,
                "IQR [Q1]": None,
                "IQR [Q3]": None,
                "IQR": None,
                "Outliers [IQR]": 0.0,
                "Outliers [Z-score]": 0.0,
            }
        ]
        assert outliers_bounds == [(None, None)]
