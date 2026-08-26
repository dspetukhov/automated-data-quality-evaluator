"""Baseline regression tests for report.py — captures CURRENT behavior as-is."""

from datetime import date
from pathlib import Path

import polars as pl
import pytest

import report
from report import (
    collect_md_content,
    format_number,
    get_report_variables,
    make_md_table,
    make_report,
    write_md_file,
)
from utility import OVERVIEW_COL, TIME_INTERVAL_COL

# ---------------------------------------------------------------------------
# get_report_variables
# ---------------------------------------------------------------------------


class TestGetReportVariables:
    def test_output_dir_defaults_to_file_stem(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        config = {"source": {"file_path": "data.subset.csv"}}
        output, *_ = get_report_variables(config)
        # split(".")[0] on the file name only keeps the part before the FIRST
        # dot, so "data.subset.csv" collapses to "data", not "data.subset".
        assert output == "data"
        assert Path(output).is_dir()

    def test_output_dir_defaults_to_postgresql_when_no_file_path(
        self, tmp_path, monkeypatch
    ):
        monkeypatch.chdir(tmp_path)
        config = {"source": {"query": "SELECT 1"}}
        output, *_ = get_report_variables(config)
        assert output == "postgresql"
        assert Path(output).is_dir()

    def test_output_dir_from_config_is_used_verbatim(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        config = {"output": "my_report", "source": {"file_path": "data.csv"}}
        output, *_ = get_report_variables(config)
        assert output == "my_report"
        assert Path(output).is_dir()

    def test_file_path_source_is_bolded_and_asterisks_escaped(
        self, tmp_path, monkeypatch
    ):
        monkeypatch.chdir(tmp_path)
        config = {"source": {"file_path": "data/*.csv"}}
        _, source, *_ = get_report_variables(config)
        assert source == "**data/\\*.csv**"

    def test_query_source_is_wrapped_in_sql_code_block(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        config = {"source": {"query": "SELECT * FROM t"}}
        _, source, *_ = get_report_variables(config)
        assert source == "\n```sql\nSELECT * FROM t\n```\n"

    def test_css_style_added_when_file_exists(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        css_file = Path("style.css")
        css_file.write_text("body {}")
        config = {
            "output": "out",
            "source": {"file_path": "data.csv"},
            "markdown": {"css_style": str(css_file)},
        }
        _, _, content, *_ = get_report_variables(config)
        assert len(content) == 1
        assert "<link rel='stylesheet' href='../style.css'>" in content[0]

    def test_css_style_with_absolute_path_and_relative_output_dir(
        self, tmp_path, monkeypatch
    ):
        # Fixed bug: css_style_file_path.relative_to(output_dir, walk_up=True)
        # used to raise ValueError ("different anchors") whenever one path was
        # absolute and the other relative. Both sides are now resolve()d first.
        monkeypatch.chdir(tmp_path)
        css_file = tmp_path / "style.css"
        css_file.write_text("body {}")
        config = {
            "output": "out",
            "source": {"file_path": "data.csv"},
            "markdown": {"css_style": str(css_file)},
        }
        _, _, content, *_ = get_report_variables(config)
        assert len(content) == 1
        assert "<link rel='stylesheet' href='../style.css'>" in content[0]

    def test_css_style_skipped_when_file_missing(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        config = {
            "output": "out",
            "source": {"file_path": "data.csv"},
            "markdown": {"css_style": "does_not_exist.css"},
        }
        _, _, content, *_ = get_report_variables(config)
        assert content == []

    def test_precision_defaults_to_four_when_not_configured(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        config = {"source": {"file_path": "data.csv"}}
        *_, precision, outliers, plotly = get_report_variables(config)
        assert precision == 4
        assert outliers == {}
        assert plotly == {}

    def test_precision_and_outliers_and_plotly_passed_through(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        config = {
            "source": {"file_path": "data.csv"},
            "markdown": {"float_precision": 3},
            "outliers": {"criterion": "IQR"},
            "plotly": {"format": "svg"},
        }
        *_, precision, outliers, plotly = get_report_variables(config)
        assert precision == 3
        assert outliers == {"criterion": "IQR"}
        assert plotly == {"format": "svg"}


# ---------------------------------------------------------------------------
# format_number
# ---------------------------------------------------------------------------


class TestFormatNumber:
    def test_plain_float_formatted_with_precision(self):
        assert format_number(3.14159, 2) == "3.14"

    def test_float_without_dot_in_str_uses_scientific_notation(self):
        # BUG-ish: the branch choice is based on `"." in str(value)`, not on
        # magnitude. A float that Python renders without a decimal point in
        # its str() (e.g. very large/small values using exponent notation)
        # falls into the scientific-notation branch instead of the plain one.
        assert format_number(1e20, 2) == "1.00e+20"

    def test_float_with_dot_uses_fixed_notation_even_if_large(self):
        assert format_number(123456.789, 2) == "123,456.79"

    def test_none_precision_raises_for_float_value(self):
        # format_number's float branch does f"{value:,.{precision}f}", which
        # raises ValueError when precision is None. get_report_variables no
        # longer passes None here (it defaults float_precision to 4), so this
        # is no longer reachable through the normal config flow, but it's
        # still format_number's behavior if called directly with precision=None.
        with pytest.raises(ValueError):
            format_number(3.14, None)

    def test_none_precision_is_fine_for_non_float_values(self):
        assert format_number(1000, None) == "1,000"
        assert format_number("hello", None) == "hello"
        assert format_number(None, None) == "None"

    def test_tuple_formatted_as_mean_plus_minus_std(self):
        assert format_number((1.5, 2.25), 2) == "1.50 ± 2.25"

    def test_int_formatted_with_thousands_separator(self):
        assert format_number(1000, 2) == "1,000"

    def test_bool_is_formatted_as_int(self):
        # BUG-ish: bool is a subclass of int in Python, so True/False hit the
        # `isinstance(value, int)` branch and print as "1"/"0" instead of
        # being passed through as-is.
        assert format_number(True, 2) == "1"

    def test_string_passed_through_unchanged(self):
        assert format_number("hello", 2) == "hello"

    def test_none_passed_through_as_string(self):
        assert format_number(None, 2) == "None"

    def test_nan_float_formatted_with_precision(self):
        assert format_number(float("nan"), 2) == "nan"


# ---------------------------------------------------------------------------
# make_md_table
# ---------------------------------------------------------------------------


class TestMakeMdTable:
    def test_basic_table_rendering(self):
        data = [{"title": "col_a", "Range": 1.0}, {"title": "col_b", "Range": 2.0}]
        table = make_md_table(data, 2)
        assert "col_a" in table
        assert "col_b" in table
        assert "**Range**" in table
        assert table.endswith("\n")

    def test_title_key_gets_blank_row_label(self):
        # When "title" is the only key, it becomes the tabulate header row
        # (headers="firstrow" consumes it), so the rendered table has a
        # header + separator but no data rows at all.
        data = [{"title": "col_a"}, {"title": "col_b"}]
        table = make_md_table(data, 2)
        lines = table.splitlines()
        assert len(lines) == 2
        assert lines[0].split("|")[1].strip() == ""
        assert "col_a" in lines[0]
        assert "col_b" in lines[0]

    def test_single_entry_padded_to_two_columns(self):
        # `while len(data) < 2: data.append({})` pads short lists so tabulate
        # always has at least 2 data columns; the padded column renders as
        # "None" for every key present only in the first entry.
        data = [{"title": "col_a", "Range": 1.0}]
        table = make_md_table(data, 2)
        assert "None" in table

    def test_empty_list_returns_blank_line(self):
        # Fixed bug: an empty `data` list is padded to two empty dicts, so
        # `data[0].keys()` is empty and `rows` stays `[]`. `make_md_table` now
        # guards against this and returns early instead of indexing into the
        # empty `rows` list (which used to raise IndexError).
        assert make_md_table([], 2) == "\n"

    def test_none_precision_raises_when_float_values_present(self):
        # format_number(value, precision=None) still raises for direct calls;
        # no longer reachable through the normal config flow (see
        # TestGetReportVariables.test_precision_defaults_to_four_when_not_configured).
        with pytest.raises(ValueError):
            make_md_table([{"title": "a", "x": 1.5}], None)


# ---------------------------------------------------------------------------
# collect_md_content
# ---------------------------------------------------------------------------


class TestCollectMdContent:
    def _evals(self):
        return [{"title": "stat", "Range": 1.0}, {"title": "stat2", "Range": 2.0}]

    def test_overview_section_uses_fixed_alias_and_anchor(self, monkeypatch):
        monkeypatch.setattr(report.time, "strftime", lambda *a, **k: "TIMESTAMP")
        data = {OVERVIEW_COL: {"evals": self._evals()}}
        result = collect_md_content(data, [], "out", "**src**", 2)
        full = "".join(result)
        assert "[Overview](#overview)" in full
        assert "## Overview" in full
        assert "![__overview](__overview)" in full

    def test_column_section_uses_backtick_alias_and_lowercased_anchor(self, monkeypatch):
        monkeypatch.setattr(report.time, "strftime", lambda *a, **k: "TIMESTAMP")
        data = {"My Col": {"evals": self._evals()}}
        result = collect_md_content(data, [], "out", "**src**", 2)
        full = "".join(result)
        assert "[`My Col`](#my-col)" in full
        assert "## `My Col`" in full
        assert "![My-Col](My-Col)" in full

    def test_extra_section_added_only_when_dtype_present(self, monkeypatch):
        monkeypatch.setattr(report.time, "strftime", lambda *a, **k: "TIMESTAMP")
        data = {
            "col_a": {
                "evals": self._evals(),
                "evals_numeric": self._evals(),
                "dtype": "Int64",
            },
            "col_b": {"evals": self._evals()},
        }
        result = collect_md_content(data, [], "out", "**src**", 2)
        full = "".join(result)
        assert "### `Int64`" in full
        assert "![col_a](col_a__extra)" in full
        assert "col_b__extra" not in full

    def test_backlink_appears_once_per_section(self, monkeypatch):
        monkeypatch.setattr(report.time, "strftime", lambda *a, **k: "TIMESTAMP")
        data = {
            "col_a": {
                "evals": self._evals(),
                "evals_numeric": self._evals(),
                "dtype": "Int64",
            },
        }
        result = collect_md_content(data, [], "out", "**src**", 2)
        full = "".join(result)
        # Backlink is only appended once per top-level column, even though
        # the extra numeric subsection is a distinct "### " block above it.
        assert full.count("[Back to table of contents](#table-of-contents)") == 1

    def test_header_includes_output_name_timestamp_and_source(self, monkeypatch):
        monkeypatch.setattr(report.time, "strftime", lambda *a, **k: "TIMESTAMP")
        data = {OVERVIEW_COL: {"evals": self._evals()}}
        result = collect_md_content(data, [], "my_output", "**src**", 2)
        assert result[0] == "# my_output | TIMESTAMP\n\n"
        assert result[1] == "Data source: **src**\n\n"

    def test_existing_content_prefix_is_preserved(self, monkeypatch):
        monkeypatch.setattr(report.time, "strftime", lambda *a, **k: "TIMESTAMP")
        data = {OVERVIEW_COL: {"evals": self._evals()}}
        prefix = ["<link rel='stylesheet' href='style.css'>\n"]
        result = collect_md_content(data, list(prefix), "out", "**src**", 2)
        assert prefix[0] in result[-1]


# ---------------------------------------------------------------------------
# write_md_file
# ---------------------------------------------------------------------------


class TestWriteMdFile:
    def test_default_filename_is_readme(self, tmp_path):
        write_md_file(["# hello\n"], str(tmp_path), None)
        assert (tmp_path / "README.md").read_text() == "# hello\n"

    def test_custom_filename_gets_md_suffix_appended(self, tmp_path):
        write_md_file(["# hello\n"], str(tmp_path), "report")
        assert (tmp_path / "report.md").read_text() == "# hello\n"

    def test_custom_filename_with_md_suffix_is_untouched(self, tmp_path):
        write_md_file(["# hello\n"], str(tmp_path), "report.md")
        assert (tmp_path / "report.md").read_text() == "# hello\n"

    def test_content_list_is_written_via_writelines(self, tmp_path):
        write_md_file(["a", "b", "c"], str(tmp_path), None)
        assert (tmp_path / "README.md").read_text() == "abc"

    def test_missing_output_dir_exits_via_exception_handler(self, tmp_path):
        # write_md_file is decorated with @exception_handler(exit_on_error=True),
        # so a failure to open the file for writing (e.g. non-existent output
        # directory) logs the error and calls sys.exit(1) rather than raising
        # the underlying FileNotFoundError.
        missing_dir = tmp_path / "does_not_exist"
        with pytest.raises(SystemExit) as exc_info:
            write_md_file(["# hello\n"], str(missing_dir), None)
        assert exc_info.value.code == 1


# ---------------------------------------------------------------------------
# make_report (integration, with evaluate_data/make_charts stubbed out)
# ---------------------------------------------------------------------------


class TestMakeReport:
    def _df(self):
        return pl.DataFrame(
            {
                TIME_INTERVAL_COL: [date(2024, 1, 1), date(2024, 1, 2)],
                " __Number of values": [10, 20],
                "__ col_a __Number of unique values": [1, 2],
                "__ col_a __Proportion of missing values": [0.0, 0.1],
                "e__ col_a __Min": [1.0, 2.0],
                "e__ col_a __Max": [5.0, 6.0],
                "e__ col_a __Mean": [3.0, 4.0],
                "e__ col_a __Median": [3.0, 4.0],
                "e__ col_a __Standard deviation": [0.5, 0.6],
                "__ col b __Number of unique values": [3, 4],
                "__ col b __Proportion of missing values": [0.0, 0.0],
            }
        )

    def _metadata(self):
        # col_a gets extra numeric stats; "col b" (with a space, and no dtype)
        # does not, exercising both branches and the space-to-hyphen renaming.
        return {"col_a": "Int64", "col b": None}

    def _stub_evaluate_and_charts(self, monkeypatch):
        eval_calls = []
        chart_calls = []

        def fake_evaluate_data(data, outliers_config):
            eval_calls.append(list(data.columns))
            return ([{"title": "stat", "Range": 1.0}], [(None, None)])

        def fake_make_charts(data, bounds, config, file_path):
            chart_calls.append(str(file_path))

        monkeypatch.setattr(report, "evaluate_data", fake_evaluate_data)
        monkeypatch.setattr(report, "make_charts", fake_make_charts)
        return eval_calls, chart_calls

    def test_selects_expected_columns_per_section(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        eval_calls, chart_calls = self._stub_evaluate_and_charts(monkeypatch)
        config = {
            "output": "out",
            "source": {"file_path": "data.csv"},
            "markdown": {"float_precision": 2},
        }
        make_report(self._df(), self._metadata(), config)

        assert eval_calls[0] == [TIME_INTERVAL_COL, " __Number of values"]
        assert eval_calls[1] == [
            TIME_INTERVAL_COL,
            "__ col_a __Number of unique values",
            "__ col_a __Proportion of missing values",
        ]
        assert eval_calls[2] == [
            TIME_INTERVAL_COL,
            "e__ col_a __Min",
            "e__ col_a __Max",
            "e__ col_a __Mean",
            "e__ col_a __Median",
            "e__ col_a __Standard deviation",
        ]
        assert eval_calls[3] == [
            TIME_INTERVAL_COL,
            "__ col b __Number of unique values",
            "__ col b __Proportion of missing values",
        ]
        # 4 chart calls: overview, col_a (common), col_a (extra), col_b
        # (common). "col b" has no extra chart since metadata["col b"] is
        # None/falsy.
        assert len(chart_calls) == 4
        assert chart_calls[0].endswith("__overview")
        assert chart_calls[1].endswith("col_a")
        assert chart_calls[2].endswith("col_a__extra")
        assert chart_calls[3].endswith("col-b")

    def test_space_in_column_name_replaced_with_hyphen_in_chart_path(
        self, tmp_path, monkeypatch
    ):
        monkeypatch.chdir(tmp_path)
        _, chart_calls = self._stub_evaluate_and_charts(monkeypatch)
        config = {
            "output": "out",
            "source": {"file_path": "data.csv"},
            "markdown": {"float_precision": 2},
        }
        make_report(self._df(), self._metadata(), config)
        assert chart_calls[3].endswith("col-b")

    def test_writes_readme_with_expected_sections(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        self._stub_evaluate_and_charts(monkeypatch)
        config = {
            "output": "out",
            "source": {"file_path": "data.csv"},
            "markdown": {"float_precision": 2},
        }
        make_report(self._df(), self._metadata(), config)

        readme = Path("out", "README.md").read_text()
        assert "## Overview" in readme
        assert "## `col_a`" in readme
        assert "### `Int64`" in readme
        assert "## `col b`" in readme
        assert "### `None`" not in readme  # col b has no dtype -> no extra section

    def test_default_float_precision_writes_readme(self, tmp_path, monkeypatch):
        # Fixed bug: when markdown.float_precision is not configured,
        # get_report_variables used to return precision=None, and
        # make_md_table/format_number raised ValueError as soon as any float
        # value was rendered (silently swallowed by make_report's
        # @exception_handler(), so no README.md was ever written). precision
        # now defaults to 4, so the default/unconfigured path succeeds.
        monkeypatch.chdir(tmp_path)
        self._stub_evaluate_and_charts(monkeypatch)
        config = {"output": "out", "source": {"file_path": "data.csv"}}
        df = self._df()
        result = make_report(df, self._metadata(), config)
        assert result is None
        assert Path("out", "README.md").exists()
