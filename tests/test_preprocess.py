"""Baseline regression tests for preprocess.py — captures CURRENT behavior as-is."""

from datetime import date, datetime

import polars as pl
import pytest

from preprocess import (
    apply_filter,
    apply_transformations,
    collect_aggregations,
    make_preprocessing,
    process_date_column,
)
from utility import PREFIX_COL, PREFIX_COL_E, TIME_INTERVAL_COL


# ---------------------------------------------------------------------------
# apply_filter
# ---------------------------------------------------------------------------


class TestApplyFilter:
    def test_none_returns_lazyframe_unchanged(self):
        lf = pl.LazyFrame({"a": [1, 2, 3]})
        result = apply_filter(lf, None)
        assert result is lf

    def test_non_string_non_none_returns_unchanged(self):
        # isinstance check only accepts str; anything else (e.g. dict, list) passes through untouched
        lf = pl.LazyFrame({"a": [1, 2, 3]})
        result = apply_filter(lf, {"not": "a string"})
        assert result is lf

    def test_sql_filter_applied(self):
        lf = pl.LazyFrame({"a": [1, 2, 3]})
        result = apply_filter(lf, "SELECT * FROM self WHERE a > 1")
        assert result.collect()["a"].to_list() == [2, 3]


# ---------------------------------------------------------------------------
# apply_transformations
# ---------------------------------------------------------------------------


class TestApplyTransformations:
    def test_none_returns_lazyframe_unchanged(self):
        lf = pl.LazyFrame({"a": [1, 2, 3]})
        result = apply_transformations(lf, None)
        assert result is lf

    def test_non_dict_returns_unchanged(self):
        lf = pl.LazyFrame({"a": [1, 2, 3]})
        result = apply_transformations(lf, ["not", "a", "dict"])
        assert result is lf

    def test_new_column_created(self):
        lf = pl.LazyFrame({"a": [1, 2, 3]})
        result = apply_transformations(lf, {"b": "a * 2"})
        df = result.collect()
        assert df["b"].to_list() == [2, 4, 6]
        assert "a" in df.columns

    def test_existing_column_replaced(self):
        lf = pl.LazyFrame({"a": [1, 2, 3]})
        result = apply_transformations(lf, {"a": "a * 10"})
        df = result.collect()
        assert df["a"].to_list() == [10, 20, 30]
        assert df.columns == ["a"]

    def test_multiple_transformations_applied_in_order(self):
        lf = pl.LazyFrame({"a": [1, 2, 3]})
        result = apply_transformations(lf, {"b": "a + 1", "c": "b + 1"})
        df = result.collect()
        assert df["b"].to_list() == [2, 3, 4]
        assert df["c"].to_list() == [3, 4, 5]

    def test_empty_dict_returns_unchanged_columns(self):
        lf = pl.LazyFrame({"a": [1, 2, 3]})
        result = apply_transformations(lf, {})
        assert result.collect().columns == ["a"]


# ---------------------------------------------------------------------------
# process_date_column
# ---------------------------------------------------------------------------


class TestProcessDateColumn:
    def test_missing_date_column_raises_system_exit(self):
        lf = pl.LazyFrame({"a": [1, 2, 3]})
        schema = lf.collect_schema()
        with pytest.raises(SystemExit) as exc:
            process_date_column(lf, schema, "missing_col", "1d")
        assert "no column 'missing_col'" in str(exc.value)

    def test_string_date_column_converted_and_renamed(self):
        lf = pl.LazyFrame({"d": ["2024-01-01", "2024-01-02"]})
        schema = lf.collect_schema()
        new_lf, new_schema = process_date_column(lf, schema, "d", "1d")
        assert TIME_INTERVAL_COL in new_schema.names()
        assert "d" not in new_schema.names()
        df = new_lf.collect()
        assert df[TIME_INTERVAL_COL].to_list() == [date(2024, 1, 1), date(2024, 1, 2)]

    def test_date_column_truncated_by_interval(self):
        lf = pl.LazyFrame({
            "d": [datetime(2024, 1, 1, 5), datetime(2024, 1, 1, 15), datetime(2024, 1, 2, 3)]
        })
        schema = lf.collect_schema()
        new_lf, _ = process_date_column(lf, schema, "d", "1d")
        df = new_lf.collect()
        assert df[TIME_INTERVAL_COL].to_list() == [
            datetime(2024, 1, 1), datetime(2024, 1, 1), datetime(2024, 1, 2)
        ]

    def test_datetime_column_truncated_by_hour(self):
        lf = pl.LazyFrame({
            "d": [datetime(2024, 1, 1, 5, 15), datetime(2024, 1, 1, 5, 45)]
        })
        schema = lf.collect_schema()
        new_lf, _ = process_date_column(lf, schema, "d", "1h")
        df = new_lf.collect()
        assert df[TIME_INTERVAL_COL].to_list() == [
            datetime(2024, 1, 1, 5), datetime(2024, 1, 1, 5)
        ]

    def test_non_string_non_date_column_raises_system_exit(self):
        # FIXED: date_column with an unsupported dtype (e.g. plain Int64) now
        # raises SystemExit immediately, instead of deferring to a confusing
        # generic Polars error at .collect() time.
        lf = pl.LazyFrame({"d": [1, 2, 3]})
        schema = lf.collect_schema()
        with pytest.raises(SystemExit) as exc:
            process_date_column(lf, schema, "d", "1d")
        assert "not supported" in str(exc.value)


# ---------------------------------------------------------------------------
# collect_aggregations
# ---------------------------------------------------------------------------


class TestCollectAggregations:
    def _agg(self, lf, aggs):
        return lf.group_by(TIME_INTERVAL_COL).agg(aggs).sort(TIME_INTERVAL_COL).collect()

    def test_basic_number_of_values_column(self):
        lf = pl.LazyFrame({TIME_INTERVAL_COL: [date(2024, 1, 1)] * 3, "a": [1, 2, 3]})
        schema = lf.collect_schema()
        aggs, metadata = collect_aggregations(schema, None, [], [])
        df = self._agg(lf, aggs)
        assert df[" __Number of values"].to_list() == [3]

    def test_target_column_adds_mean_expr(self):
        lf = pl.LazyFrame({
            TIME_INTERVAL_COL: [date(2024, 1, 1)] * 2,
            "a": [1, 2],
            "target": [0, 1],
        })
        schema = lf.collect_schema()
        aggs, metadata = collect_aggregations(schema, "target", [], [])
        df = self._agg(lf, aggs)
        assert df[" __Target average"].to_list() == [0.5]

    def test_excluded_column_not_in_metadata_or_aggs(self):
        lf = pl.LazyFrame({
            TIME_INTERVAL_COL: [date(2024, 1, 1)] * 2,
            "a": [1, 2],
            "skip_me": [10, 20],
        })
        schema = lf.collect_schema()
        aggs, metadata = collect_aggregations(schema, None, ["skip_me"], [])
        assert "skip_me" not in metadata
        assert "a" in metadata
        df = self._agg(lf, aggs)
        assert not any("skip_me" in c for c in df.columns)

    def test_time_interval_col_excluded_from_metadata(self):
        lf = pl.LazyFrame({TIME_INTERVAL_COL: [date(2024, 1, 1)] * 2, "a": [1, 2]})
        schema = lf.collect_schema()
        _, metadata = collect_aggregations(schema, None, [], [])
        assert TIME_INTERVAL_COL not in metadata

    def test_numeric_column_metadata_and_stats(self):
        lf = pl.LazyFrame({
            TIME_INTERVAL_COL: [date(2024, 1, 1)] * 3,
            "a": [1, 2, None],
        })
        schema = lf.collect_schema()
        aggs, metadata = collect_aggregations(schema, None, [], [])
        assert metadata["a"] == str(schema["a"])
        df = self._agg(lf, aggs)
        # FIXED: n_unique() is now computed on drop_nulls(), so null values
        # are excluded from the unique count (tracked separately via
        # "Proportion of missing values").
        assert df[f"{PREFIX_COL} a __Number of unique values"].to_list() == [2]
        assert df[f"{PREFIX_COL} a __Proportion of missing values"].to_list() == [pytest.approx(1 / 3)]
        assert df[f"{PREFIX_COL_E} a __Min"].to_list() == [1]
        assert df[f"{PREFIX_COL_E} a __Max"].to_list() == [2]
        assert df[f"{PREFIX_COL_E} a __Mean"].to_list() == [1.5]

    def test_non_numeric_column_uses_string_length(self):
        lf = pl.LazyFrame({
            TIME_INTERVAL_COL: [date(2024, 1, 1)] * 2,
            "s": ["ab", "abcd"],
        })
        schema = lf.collect_schema()
        aggs, metadata = collect_aggregations(schema, None, [], [])
        df = self._agg(lf, aggs)
        # Min/Max/Mean computed on string LENGTH, not lexical value
        assert df[f"{PREFIX_COL_E} s __Min"].to_list() == [2]
        assert df[f"{PREFIX_COL_E} s __Max"].to_list() == [4]
        assert df[f"{PREFIX_COL_E} s __Mean"].to_list() == [3]

    def test_columns_to_exclude_extra_statistics_sets_metadata_none(self):
        lf = pl.LazyFrame({
            TIME_INTERVAL_COL: [date(2024, 1, 1)] * 2,
            "a": [1, 2],
        })
        schema = lf.collect_schema()
        aggs, metadata = collect_aggregations(schema, None, [], ["a"])
        assert metadata["a"] is None
        df = self._agg(lf, aggs)
        # Extra stat columns must not exist for excluded column
        assert f"{PREFIX_COL_E} a __Min" not in df.columns
        # But common stats (n_unique, null rate) still present
        assert f"{PREFIX_COL} a __Number of unique values" in df.columns

    def test_no_target_column_no_target_average_expr(self):
        lf = pl.LazyFrame({TIME_INTERVAL_COL: [date(2024, 1, 1)] * 2, "a": [1, 2]})
        schema = lf.collect_schema()
        aggs, _ = collect_aggregations(schema, None, [], [])
        df = self._agg(lf, aggs)
        assert " __Target average" not in df.columns


# ---------------------------------------------------------------------------
# make_preprocessing (end-to-end, decorated with exception_handler(exit_on_error=True))
# ---------------------------------------------------------------------------


class TestMakePreprocessing:
    def test_full_pipeline_basic(self):
        lf = pl.LazyFrame({
            "d": [
                datetime(2024, 1, 1, 3), datetime(2024, 1, 1, 20),
                datetime(2024, 1, 2, 5),
            ],
            "value": [10, 20, 30],
        })
        config = {"date_column": "d", "time_interval": "1d"}
        df, metadata = make_preprocessing(lf, config)
        assert df[TIME_INTERVAL_COL].to_list() == [datetime(2024, 1, 1), datetime(2024, 1, 2)]
        assert df[" __Number of values"].to_list() == [2, 1]
        assert metadata["value"] == "Int64"

    def test_filter_and_transformation_applied(self):
        lf = pl.LazyFrame({
            "d": [datetime(2024, 1, 1), datetime(2024, 1, 2), datetime(2024, 1, 3)],
            "value": [1, 2, 3],
        })
        config = {
            "date_column": "d",
            "filter": "SELECT * FROM self WHERE value > 1",
            "transformations": {"doubled": "value * 2"},
        }
        df, metadata = make_preprocessing(lf, config)
        assert df[" __Number of values"].to_list() == [1, 1]
        assert "doubled" in metadata

    def test_missing_date_column_exits(self):
        # process_date_column raises SystemExit (BaseException, not Exception),
        # so exception_handler's `except Exception` does not intercept it —
        # it propagates straight out of make_preprocessing.
        lf = pl.LazyFrame({"value": [1, 2, 3]})
        config = {"date_column": "nonexistent"}
        with pytest.raises(SystemExit):
            make_preprocessing(lf, config)

    def test_target_column_not_in_schema_logs_warning_and_continues(self):
        lf = pl.LazyFrame({
            "d": [datetime(2024, 1, 1)],
            "value": [1],
        })
        config = {"date_column": "d", "target_column": "missing_target"}
        df, metadata = make_preprocessing(lf, config)
        assert " __Target average" not in df.columns

    def test_columns_to_exclude(self):
        lf = pl.LazyFrame({
            "d": [datetime(2024, 1, 1)],
            "value": [1],
            "secret": [42],
        })
        config = {"date_column": "d", "columns_to_exclude": ["secret"]}
        df, metadata = make_preprocessing(lf, config)
        assert "secret" not in metadata
        assert not any("secret" in c for c in df.columns)

    def test_default_time_interval_is_one_day(self):
        lf = pl.LazyFrame({
            "d": [datetime(2024, 1, 1, 1), datetime(2024, 1, 1, 23)],
            "value": [1, 2],
        })
        config = {"date_column": "d"}
        df, _ = make_preprocessing(lf, config)
        assert df[TIME_INTERVAL_COL].to_list() == [datetime(2024, 1, 1)]

    def test_invalid_sql_filter_triggers_exit_on_error(self):
        # apply_filter raises inside lf.sql() for malformed SQL; caught by
        # exception_handler(exit_on_error=True) → logs and sys.exit(1)
        lf = pl.LazyFrame({
            "d": [datetime(2024, 1, 1)],
            "value": [1],
        })
        config = {"date_column": "d", "filter": "NOT VALID SQL AT ALL !!"}
        with pytest.raises(SystemExit):
            make_preprocessing(lf, config)

    def test_default_date_column_key_missing_exits(self):
        # config.get("date_column", "date_column") falls back to literal string
        # "date_column" when key absent; if that column also doesn't exist, exits.
        lf = pl.LazyFrame({"value": [1, 2, 3]})
        config = {}
        with pytest.raises(SystemExit):
            make_preprocessing(lf, config)
