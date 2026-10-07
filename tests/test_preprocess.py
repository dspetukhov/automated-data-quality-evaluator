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
    def test_string_date_column_converted_and_renamed(self):
        lf = pl.LazyFrame({"d": ["2024-01-01", "2024-01-02"]})
        schema = lf.collect_schema()
        new_lf = process_date_column(lf, "d", schema["d"], "1d")
        new_schema = new_lf.collect_schema()
        assert TIME_INTERVAL_COL in new_schema.names()
        assert "d" not in new_schema.names()
        df = new_lf.collect()
        assert new_schema[TIME_INTERVAL_COL] == pl.Datetime
        assert df[TIME_INTERVAL_COL].to_list() == [
            datetime(2024, 1, 1),
            datetime(2024, 1, 2),
        ]

    def test_date_column_truncated_by_interval(self):
        lf = pl.LazyFrame(
            {
                "d": [
                    datetime(2024, 1, 1, 5),
                    datetime(2024, 1, 1, 15),
                    datetime(2024, 1, 2, 3),
                ]
            }
        )
        schema = lf.collect_schema()
        new_lf = process_date_column(lf, "d", schema["d"], "1d")
        df = new_lf.collect()
        assert df[TIME_INTERVAL_COL].to_list() == [
            datetime(2024, 1, 1),
            datetime(2024, 1, 1),
            datetime(2024, 1, 2),
        ]

    def test_datetime_column_truncated_by_hour(self):
        lf = pl.LazyFrame(
            {"d": [datetime(2024, 1, 1, 5, 15), datetime(2024, 1, 1, 5, 45)]}
        )
        schema = lf.collect_schema()
        new_lf = process_date_column(lf, "d", schema["d"], "1h")
        df = new_lf.collect()
        assert df[TIME_INTERVAL_COL].to_list() == [
            datetime(2024, 1, 1, 5),
            datetime(2024, 1, 1, 5),
        ]

    def test_returned_schema_matches_returned_lazyframe(self):
        lf = pl.LazyFrame({"d": [date(2024, 1, 1)], "v": [1]})
        new_lf = process_date_column(lf, "d", lf.collect_schema()["d"], "1d")
        new_schema = new_lf.collect_schema()
        assert new_schema.names() == [TIME_INTERVAL_COL, "v"]
        assert new_schema[TIME_INTERVAL_COL] == pl.Date

    def test_string_with_time_component_parsed_as_datetime(self):
        # String columns are parsed with `str.to_datetime(strict=True)`, so
        # datetime-formatted strings are supported.
        lf = pl.LazyFrame({"d": ["2024-01-01 10:30:00"]})
        new_lf = process_date_column(lf, "d", lf.collect_schema()["d"], "1h")
        assert new_lf.collect()[TIME_INTERVAL_COL].to_list() == [
            datetime(2024, 1, 1, 10)
        ]

    def test_unparseable_string_fails_lazily_at_collect(self):
        # Parse errors are deferred to .collect(), not raised here.
        lf = pl.LazyFrame({"d": ["not-a-date"]})
        new_lf = process_date_column(lf, "d", lf.collect_schema()["d"], "1d")
        with pytest.raises(pl.exceptions.ComputeError):
            new_lf.collect()

    def test_invalid_time_interval_fails_lazily_at_collect(self):
        # NOTE: time_interval is not validated up front; an invalid string only
        # fails at .collect() time with a Polars error (not a SystemExit).
        lf = pl.LazyFrame({"d": [date(2024, 1, 1)]})
        new_lf = process_date_column(lf, "d", lf.collect_schema()["d"], "xyz")
        with pytest.raises(pl.exceptions.InvalidOperationError):
            new_lf.collect()

    def test_sub_day_interval_on_date_column_is_silent_noop(self):
        # NOTE (suspected bug, not fixed): "1h" on a Date column silently leaves
        # dates unchanged rather than warning that the interval is too fine.
        lf = pl.LazyFrame({"d": [date(2024, 1, 1), date(2024, 1, 2)]})
        new_lf = process_date_column(lf, "d", lf.collect_schema()["d"], "1h")
        assert new_lf.collect()[TIME_INTERVAL_COL].to_list() == [
            date(2024, 1, 1),
            date(2024, 1, 2),
        ]

    def test_weekly_interval_truncates_to_monday(self):
        lf = pl.LazyFrame({"d": [date(2024, 1, 3), date(2024, 1, 7)]})  # Wed, Sun
        new_lf = process_date_column(lf, "d", lf.collect_schema()["d"], "1w")
        assert new_lf.collect()[TIME_INTERVAL_COL].to_list() == [date(2024, 1, 1)] * 2

    def test_other_columns_preserved(self):
        lf = pl.LazyFrame({"d": [date(2024, 1, 1)], "v": [5]})
        new_lf = process_date_column(lf, "d", lf.collect_schema()["d"], "1d")
        assert new_lf.collect()["v"].to_list() == [5]


# ---------------------------------------------------------------------------
# collect_aggregations
# ---------------------------------------------------------------------------


class TestCollectAggregations:
    @staticmethod
    def _run(lf, result):
        """Mimic make_preprocessing: aggregate `aggs`, and `aggs_extra` after `with_columns`."""
        aggs, aggs_extra, with_columns, _ = result
        df = lf.group_by(TIME_INTERVAL_COL).agg(aggs).sort(TIME_INTERVAL_COL).collect()
        if aggs_extra:
            extra = (
                lf.with_columns(with_columns)
                .group_by(TIME_INTERVAL_COL)
                .agg(aggs_extra)
                .sort(TIME_INTERVAL_COL)
                .collect()
            )
            df = df.join(extra, on=TIME_INTERVAL_COL, how="inner")
        return df

    def test_returns_four_tuple(self):
        lf = pl.LazyFrame({TIME_INTERVAL_COL: [date(2024, 1, 1)], "a": [1]})
        result = collect_aggregations(lf.collect_schema(), None, [], [])
        assert isinstance(result, tuple) and len(result) == 4
        aggs, aggs_extra, with_columns, metadata = result
        # " __Number of values" + (n_unique, null rate) for "a"
        assert len(aggs) == 3
        # min/max/mean/median/std for "a"
        assert len(aggs_extra) == 5
        assert with_columns == []
        assert metadata == {"a": "Int64"}

    def test_basic_number_of_values_column(self):
        lf = pl.LazyFrame({TIME_INTERVAL_COL: [date(2024, 1, 1)] * 3, "a": [1, 2, 3]})
        result = collect_aggregations(lf.collect_schema(), None, [], [])
        df = self._run(lf, result)
        assert df[" __Number of values"].to_list() == [3]

    def test_target_column_adds_mean_expr(self):
        lf = pl.LazyFrame(
            {
                TIME_INTERVAL_COL: [date(2024, 1, 1)] * 2,
                "a": [1, 2],
                "target": [0, 1],
            }
        )
        result = collect_aggregations(lf.collect_schema(), "target", [], [])
        df = self._run(lf, result)
        assert df[" __Target average"].to_list() == [0.5]

    def test_no_target_column_no_target_average_expr(self):
        lf = pl.LazyFrame({TIME_INTERVAL_COL: [date(2024, 1, 1)] * 2, "a": [1, 2]})
        df = self._run(lf, collect_aggregations(lf.collect_schema(), None, [], []))
        assert " __Target average" not in df.columns

    def test_excluded_column_not_in_metadata_or_aggs(self):
        lf = pl.LazyFrame(
            {
                TIME_INTERVAL_COL: [date(2024, 1, 1)] * 2,
                "a": [1, 2],
                "skip_me": [10, 20],
            }
        )
        result = collect_aggregations(lf.collect_schema(), None, ["skip_me"], [])
        assert "skip_me" not in result[3]
        assert "a" in result[3]
        df = self._run(lf, result)
        assert not any("skip_me" in c for c in df.columns)

    def test_excluded_target_column_still_gets_target_average(self):
        # NOTE (suspected inconsistency, not fixed): a target column listed in
        # `columns_to_exclude` is skipped in the per-column loop, but its
        # " __Target average" is still computed because that expression is added
        # before the exclusion check.
        lf = pl.LazyFrame(
            {
                TIME_INTERVAL_COL: [date(2024, 1, 1)] * 2,
                "a": [1, 2],
                "target": [0, 1],
            }
        )
        result = collect_aggregations(lf.collect_schema(), "target", ["target"], [])
        assert "target" not in result[3]
        df = self._run(lf, result)
        assert df[" __Target average"].to_list() == [0.5]
        assert not any("target __" in c for c in df.columns)

    def test_time_interval_col_excluded_from_metadata(self):
        lf = pl.LazyFrame({TIME_INTERVAL_COL: [date(2024, 1, 1)] * 2, "a": [1, 2]})
        _, _, _, metadata = collect_aggregations(lf.collect_schema(), None, [], [])
        assert TIME_INTERVAL_COL not in metadata

    def test_numeric_column_metadata_and_stats(self):
        lf = pl.LazyFrame(
            {
                TIME_INTERVAL_COL: [date(2024, 1, 1)] * 3,
                "a": [1, 2, None],
            }
        )
        schema = lf.collect_schema()
        result = collect_aggregations(schema, None, [], [])
        assert result[3]["a"] == str(schema["a"])
        df = self._run(lf, result)
        # n_unique is computed on drop_nulls(): nulls are not a distinct value
        assert df[f"{PREFIX_COL} a __Number of unique values"].to_list() == [2]
        assert df[f"{PREFIX_COL} a __Proportion of missing values"].to_list() == [
            pytest.approx(1 / 3)
        ]
        assert df[f"{PREFIX_COL_E} a __Min"].to_list() == [1]
        assert df[f"{PREFIX_COL_E} a __Max"].to_list() == [2]
        assert df[f"{PREFIX_COL_E} a __Mean"].to_list() == [1.5]
        assert df[f"{PREFIX_COL_E} a __Median"].to_list() == [1.5]
        assert df[f"{PREFIX_COL_E} a __Standard deviation"].to_list() == [
            pytest.approx(0.7071067811865476)
        ]

    def test_string_column_uses_char_length_via_with_columns(self):
        lf = pl.LazyFrame(
            {
                TIME_INTERVAL_COL: [date(2024, 1, 1)] * 2,
                "s": ["ab", "abcd"],
            }
        )
        result = collect_aggregations(lf.collect_schema(), None, [], [])
        assert len(result[2]) == 1  # one len_chars expression for "s"
        assert result[3]["s"] == "String"
        df = self._run(lf, result)
        # Min/Max/Mean computed on string LENGTH, not lexical value
        assert df[f"{PREFIX_COL_E} s __Min"].to_list() == [2]
        assert df[f"{PREFIX_COL_E} s __Max"].to_list() == [4]
        assert df[f"{PREFIX_COL_E} s __Mean"].to_list() == [3]
        # Common stats are computed on the original (non-length) values
        assert df[f"{PREFIX_COL} s __Number of unique values"].to_list() == [2]

    def test_categorical_column_uses_char_length(self):
        lf = pl.LazyFrame(
            {
                TIME_INTERVAL_COL: [date(2024, 1, 1)] * 2,
                "c": pl.Series(["x", "yyy"], dtype=pl.Categorical),
            }
        )
        result = collect_aggregations(lf.collect_schema(), None, [], [])
        assert len(result[2]) == 1
        assert result[3]["c"] == "Categorical"
        df = self._run(lf, result)
        assert df[f"{PREFIX_COL_E} c __Min"].to_list() == [1]
        assert df[f"{PREFIX_COL_E} c __Max"].to_list() == [3]

    def test_boolean_column_stats_not_length_based(self):
        lf = pl.LazyFrame(
            {TIME_INTERVAL_COL: [date(2024, 1, 1)] * 2, "b": [True, False]}
        )
        result = collect_aggregations(lf.collect_schema(), None, [], [])
        assert result[2] == []
        df = self._run(lf, result)
        assert df[f"{PREFIX_COL_E} b __Mean"].to_list() == [0.5]

    def test_columns_to_exclude_extra_statistics_sets_metadata_none(self):
        lf = pl.LazyFrame({TIME_INTERVAL_COL: [date(2024, 1, 1)] * 2, "a": [1, 2]})
        result = collect_aggregations(lf.collect_schema(), None, [], ["a"])
        aggs, aggs_extra, with_columns, metadata = result
        assert metadata["a"] is None
        assert aggs_extra == []
        df = self._run(lf, result)
        assert f"{PREFIX_COL_E} a __Min" not in df.columns
        # But common stats (n_unique, null rate) still present
        assert f"{PREFIX_COL} a __Number of unique values" in df.columns
        assert f"{PREFIX_COL} a __Proportion of missing values" in df.columns

    def test_extra_statistics_excluded_string_has_no_len_expression(self):
        lf = pl.LazyFrame({TIME_INTERVAL_COL: [date(2024, 1, 1)], "s": ["ab"]})
        _, aggs_extra, with_columns, metadata = collect_aggregations(
            lf.collect_schema(), None, [], ["s"]
        )
        assert with_columns == []
        assert aggs_extra == []
        assert metadata == {"s": None}

    def test_only_time_interval_column(self):
        lf = pl.LazyFrame({TIME_INTERVAL_COL: [date(2024, 1, 1)] * 2})
        result = collect_aggregations(lf.collect_schema(), None, [], [])
        assert len(result[0]) == 1
        assert result[1] == [] and result[2] == [] and result[3] == {}
        assert self._run(lf, result).columns == [
            TIME_INTERVAL_COL,
            " __Number of values",
        ]

    def test_metadata_preserves_schema_order(self):
        lf = pl.LazyFrame({TIME_INTERVAL_COL: [date(2024, 1, 1)], "z": [1], "a": ["x"]})
        _, _, _, metadata = collect_aggregations(lf.collect_schema(), None, [], [])
        assert list(metadata) == ["z", "a"]

    def test_string_target_column_yields_null_average(self):
        # collect_aggregations itself does not validate the target dtype;
        # make_preprocessing drops non-numeric targets before calling it.
        lf = pl.LazyFrame({TIME_INTERVAL_COL: [date(2024, 1, 1)] * 2, "t": ["a", "b"]})
        result = collect_aggregations(lf.collect_schema(), "t", [], [])
        df = self._run(lf, result)
        assert df[" __Target average"].to_list() == [None]


# ---------------------------------------------------------------------------
# make_preprocessing (end-to-end, decorated with exception_handler(exit_on_error=True))
# ---------------------------------------------------------------------------


class TestMakePreprocessing:
    def test_full_pipeline_basic(self):
        lf = pl.LazyFrame(
            {
                "d": [
                    datetime(2024, 1, 1, 3),
                    datetime(2024, 1, 1, 20),
                    datetime(2024, 1, 2, 5),
                ],
                "value": [10, 20, 30],
            }
        )
        config = {"date_column": "d", "time_interval": "1d"}
        df, metadata = make_preprocessing(lf, config)
        assert df[TIME_INTERVAL_COL].to_list() == [
            datetime(2024, 1, 1),
            datetime(2024, 1, 2),
        ]
        assert df[" __Number of values"].to_list() == [2, 1]
        assert metadata["value"] == "Int64"

    def test_filter_and_transformation_applied(self):
        lf = pl.LazyFrame(
            {
                "d": [datetime(2024, 1, 1), datetime(2024, 1, 2), datetime(2024, 1, 3)],
                "value": [1, 2, 3],
            }
        )
        config = {
            "date_column": "d",
            "filter": "SELECT * FROM self WHERE value > 1",
            "transformations": {"doubled": "value * 2"},
        }
        df, metadata = make_preprocessing(lf, config)
        assert df[" __Number of values"].to_list() == [1, 1]
        assert "doubled" in metadata

    def test_missing_date_column_exits(self):
        # make_preprocessing raises SystemExit (BaseException, not Exception),
        # so exception_handler's `except Exception` does not intercept it —
        # it propagates straight out of make_preprocessing.
        lf = pl.LazyFrame({"value": [1, 2, 3]})
        config = {"date_column": "nonexistent"}
        with pytest.raises(SystemExit):
            make_preprocessing(lf, config)

    def test_target_column_not_in_schema_logs_warning_and_continues(self):
        lf = pl.LazyFrame(
            {
                "d": [datetime(2024, 1, 1)],
                "value": [1],
            }
        )
        config = {"date_column": "d", "target_column": "missing_target"}
        df, metadata = make_preprocessing(lf, config)
        assert " __Target average" not in df.columns

    def test_columns_to_exclude(self):
        lf = pl.LazyFrame(
            {
                "d": [datetime(2024, 1, 1)],
                "value": [1],
                "secret": [42],
            }
        )
        config = {"date_column": "d", "columns_to_exclude": ["secret"]}
        df, metadata = make_preprocessing(lf, config)
        assert "secret" not in metadata
        assert not any("secret" in c for c in df.columns)

    def test_default_time_interval_is_one_day(self):
        lf = pl.LazyFrame(
            {
                "d": [datetime(2024, 1, 1, 1), datetime(2024, 1, 1, 23)],
                "value": [1, 2],
            }
        )
        config = {"date_column": "d"}
        df, _ = make_preprocessing(lf, config)
        assert df[TIME_INTERVAL_COL].to_list() == [datetime(2024, 1, 1)]

    def test_invalid_sql_filter_triggers_exit_on_error(self):
        # apply_filter raises inside lf.sql() for malformed SQL; caught by
        # exception_handler(exit_on_error=True) → logs and sys.exit(1)
        lf = pl.LazyFrame(
            {
                "d": [datetime(2024, 1, 1)],
                "value": [1],
            }
        )
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

    def test_output_sorted_by_time_interval(self):
        lf = pl.LazyFrame(
            {
                "d": [datetime(2024, 1, 3), datetime(2024, 1, 1), datetime(2024, 1, 2)],
                "value": [1, 2, 3],
            }
        )
        df, _ = make_preprocessing(lf, {"date_column": "d"})
        assert df[TIME_INTERVAL_COL].to_list() == sorted(
            df[TIME_INTERVAL_COL].to_list()
        )

    def test_string_date_column_pipeline(self):
        lf = pl.LazyFrame(
            {"d": ["2024-01-01", "2024-01-01", "2024-01-02"], "value": [1, 2, 3]}
        )
        df, _ = make_preprocessing(lf, {"date_column": "d"})
        assert df[TIME_INTERVAL_COL].to_list() == [
            datetime(2024, 1, 1),
            datetime(2024, 1, 2),
        ]
        assert df[" __Number of values"].to_list() == [2, 1]

    def test_column_layout_common_then_extra(self):
        lf = pl.LazyFrame({"d": [datetime(2024, 1, 1)], "value": [1]})
        df, _ = make_preprocessing(lf, {"date_column": "d"})
        assert df.columns == [
            TIME_INTERVAL_COL,
            " __Number of values",
            f"{PREFIX_COL} value __Number of unique values",
            f"{PREFIX_COL} value __Proportion of missing values",
            f"{PREFIX_COL_E} value __Min",
            f"{PREFIX_COL_E} value __Max",
            f"{PREFIX_COL_E} value __Mean",
            f"{PREFIX_COL_E} value __Median",
            f"{PREFIX_COL_E} value __Standard deviation",
        ]

    def test_target_column_average(self):
        lf = pl.LazyFrame(
            {
                "d": [datetime(2024, 1, 1), datetime(2024, 1, 1), datetime(2024, 1, 2)],
                "t": [0, 1, 1],
            }
        )
        df, metadata = make_preprocessing(
            lf, {"date_column": "d", "target_column": "t"}
        )
        assert df[" __Target average"].to_list() == [0.5, 1.0]
        # target column also gets regular per-column stats
        assert metadata["t"] == "Int64"

    def test_string_and_categorical_stats_use_lengths(self):
        lf = pl.LazyFrame(
            {
                "d": [datetime(2024, 1, 1)] * 2,
                "s": ["ab", "abcd"],
                "c": pl.Series(["x", "yyy"], dtype=pl.Categorical),
            }
        )
        df, metadata = make_preprocessing(lf, {"date_column": "d"})
        assert metadata == {"s": "String", "c": "Categorical"}
        assert df[f"{PREFIX_COL_E} s __Mean"].to_list() == [3]
        assert df[f"{PREFIX_COL_E} c __Max"].to_list() == [3]

    def test_columns_to_exclude_extra_statistics(self):
        lf = pl.LazyFrame({"d": [datetime(2024, 1, 1)], "a": [1], "b": [2]})
        config = {"date_column": "d", "columns_to_exclude_extra_statistics": ["a"]}
        df, metadata = make_preprocessing(lf, config)
        assert metadata == {"a": None, "b": "Int64"}
        assert f"{PREFIX_COL} a __Number of unique values" in df.columns
        assert f"{PREFIX_COL_E} a __Min" not in df.columns
        assert f"{PREFIX_COL_E} b __Min" in df.columns

    def test_no_extra_statistics_at_all_skips_join(self):
        lf = pl.LazyFrame({"d": [datetime(2024, 1, 1)], "a": [1]})
        config = {"date_column": "d", "columns_to_exclude_extra_statistics": ["a"]}
        df, _ = make_preprocessing(lf, config)
        assert not any(c.startswith(PREFIX_COL_E) for c in df.columns)

    def test_only_date_column_yields_only_counts(self):
        lf = pl.LazyFrame({"d": [datetime(2024, 1, 1)] * 3})
        df, metadata = make_preprocessing(lf, {"date_column": "d"})
        assert df.columns == [TIME_INTERVAL_COL, " __Number of values"]
        assert df[" __Number of values"].to_list() == [3]
        assert metadata == {}

    def test_filter_removing_all_rows_returns_empty_frame(self):
        lf = pl.LazyFrame({"d": [datetime(2024, 1, 1)], "value": [1]})
        config = {"date_column": "d", "filter": "SELECT * FROM self WHERE value > 100"}
        df, metadata = make_preprocessing(lf, config)
        assert df.height == 0
        assert metadata == {"value": "Int64"}

    def test_transformation_can_replace_date_column(self):
        lf = pl.LazyFrame({"d": [datetime(2024, 1, 1, 5)], "value": [1]})
        config = {"date_column": "d", "transformations": {"d": "d + INTERVAL '1 day'"}}
        df, _ = make_preprocessing(lf, config)
        assert df[TIME_INTERVAL_COL].to_list() == [datetime(2024, 1, 2)]

    def test_metadata_reflects_transformed_dtype(self):
        lf = pl.LazyFrame({"d": [datetime(2024, 1, 1)], "value": [1]})
        config = {
            "date_column": "d",
            "transformations": {"value": "CAST(value AS DOUBLE)"},
        }
        _, metadata = make_preprocessing(lf, config)
        assert metadata["value"] == "Float64"

    def test_streaming_engine_and_chunk_size_accepted(self):
        lf = pl.LazyFrame({"d": [datetime(2024, 1, 1)], "value": [1]})
        config = {"date_column": "d", "engine": "streaming", "streaming_chunk_size": 10}
        df, _ = make_preprocessing(lf, config)
        assert df[" __Number of values"].to_list() == [1]

    def test_non_int_streaming_chunk_size_ignored(self):
        lf = pl.LazyFrame({"d": [datetime(2024, 1, 1)], "value": [1]})
        config = {"date_column": "d", "streaming_chunk_size": "10"}
        df, _ = make_preprocessing(lf, config)
        assert df.height == 1

    def test_null_date_rows_kept_when_extra_stats_present(self):
        # The join of common and extra aggregations matches null keys, so the
        # null-date group survives (same as without extra stats).
        lf = pl.LazyFrame({"d": [datetime(2024, 1, 1), None], "value": [1, 2]})
        df, _ = make_preprocessing(lf, {"date_column": "d"})
        assert df[TIME_INTERVAL_COL].to_list() == [None, datetime(2024, 1, 1)]
        assert df[" __Number of values"].to_list() == [1, 1]
        assert df["e__ value __Max"].to_list() == [2, 1]

    def test_null_date_rows_kept_when_no_extra_stats(self):
        lf = pl.LazyFrame({"d": [datetime(2024, 1, 1), None]})
        df, _ = make_preprocessing(lf, {"date_column": "d"})
        assert df.height == 2
        assert df[TIME_INTERVAL_COL].to_list() == [None, datetime(2024, 1, 1)]

    def test_column_named_target_column_is_target_by_default(self):
        # "target_column" absent from config: defaults to a column of that name.
        lf = pl.LazyFrame({"d": [datetime(2024, 1, 1)] * 2, "target_column": [0, 1]})
        df, _ = make_preprocessing(lf, {"date_column": "d"})
        assert df[" __Target average"].to_list() == [0.5]

    def test_string_target_column_skipped(self):
        lf = pl.LazyFrame({"d": [datetime(2024, 1, 1)] * 2, "t": ["a", "b"]})
        df, metadata = make_preprocessing(
            lf, {"date_column": "d", "target_column": "t"}
        )
        assert " __Target average" not in df.columns
        assert "t" in metadata

    def test_boolean_target_column_averaged(self):
        lf = pl.LazyFrame({"d": [datetime(2024, 1, 1)] * 2, "t": [True, False]})
        df, _ = make_preprocessing(lf, {"date_column": "d", "target_column": "t"})
        assert df[" __Target average"].to_list() == [0.5]

    def test_bool_streaming_chunk_size_ignored(self):
        lf = pl.LazyFrame({"d": [datetime(2024, 1, 1)], "value": [1]})
        before = pl.Config.state().get("POLARS_STREAMING_CHUNK_SIZE")
        make_preprocessing(lf, {"date_column": "d", "streaming_chunk_size": True})
        assert pl.Config.state().get("POLARS_STREAMING_CHUNK_SIZE") == before

    def test_excluded_target_column_still_averaged(self):
        # Intended: a configured target always gets its average, even if excluded
        # from per-column statistics.
        lf = pl.LazyFrame({"d": [datetime(2024, 1, 1)] * 2, "t": [0, 1], "v": [1, 2]})
        config = {"date_column": "d", "target_column": "t", "columns_to_exclude": ["t"]}
        df, metadata = make_preprocessing(lf, config)
        assert "t" not in metadata
        assert df[" __Target average"].to_list() == [0.5]

    def test_boolean_date_dtype_exits(self):
        lf = pl.LazyFrame({"d": [True, False]})
        with pytest.raises(SystemExit):
            make_preprocessing(lf, {"date_column": "d"})

    def test_unsupported_date_dtype_exits(self):
        lf = pl.LazyFrame({"d": [1, 2], "value": [1, 2]})
        with pytest.raises(SystemExit):
            make_preprocessing(lf, {"date_column": "d"})

    def test_unparseable_string_date_exits_via_exception_handler(self):
        # Unlike SystemExit raised directly, a Polars ComputeError raised at
        # collect time IS caught by exception_handler(exit_on_error=True).
        lf = pl.LazyFrame({"d": ["not-a-date"], "value": [1]})
        with pytest.raises(SystemExit):
            make_preprocessing(lf, {"date_column": "d"})
