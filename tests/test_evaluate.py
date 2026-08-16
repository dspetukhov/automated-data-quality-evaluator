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

    def test_none_std_falls_through_to_zscore_branch(self):
        # NOTE (fragile-by-accident, not a crash): with a single non-null
        # value, Polars std() returns None. `std == 0` is False for None
        # (None != 0), so the code does NOT take the short-circuit branch and
        # instead evaluates `(data - mean) / std` with std=None. Polars
        # broadcasts the None as a null, producing a null Series; the
        # subsequent `> threshold` comparison and `.sum()` silently treat
        # that null as "not counted," landing on outliers_zscore == 0 anyway
        # -- but only because of null-propagation semantics, not because the
        # std==0 guard caught this degenerate single-value case.
        data, mean, std, q1, q3 = self._stats([5])
        assert std is None
        outliers_iqr, outliers_zscore, bounds = evaluate_data_outliers(
            data, mean, std, q1, q3, {}
        )
        assert outliers_zscore == 0
        assert outliers_iqr == 0
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

    def test_empty_dataframe_zero_rows_returns_original_data_due_to_exception_handler(self):
        # BUG: with 0 rows, `100 * outliers / data.shape[0]` divides by zero,
        # raising ZeroDivisionError inside the @exception_handler()-wrapped
        # evaluate_data. The decorator swallows the exception, logs it, and
        # (since exit_on_error defaults to False) returns `args[0]` — i.e. the
        # original input DataFrame — instead of the documented
        # tuple[list, list] return type. Callers relying on the documented
        # signature would break downstream.
        df = pl.DataFrame({
            "__time_interval": pl.Series([], dtype=pl.Int64),
            "a": pl.Series([], dtype=pl.Int64),
        })
        result = evaluate_data(df, {})
        assert isinstance(result, pl.DataFrame)
        assert result is df

    def test_all_null_column_returns_original_data_due_to_exception_handler(self):
        # BUG: an all-null column yields mean/std/q1/q3 = None. The IQR bound
        # computation `q1 - multiplier * (q3 - q1)` then does arithmetic on
        # None, raising TypeError, which is swallowed by @exception_handler()
        # the same way as above -> the raw input DataFrame is returned instead
        # of the documented (list, list) tuple.
        df = pl.DataFrame({
            "__time_interval": [1, 2, 3],
            "a": pl.Series([None, None, None], dtype=pl.Int64),
        })
        result = evaluate_data(df, {})
        assert isinstance(result, pl.DataFrame)
        assert result is df
