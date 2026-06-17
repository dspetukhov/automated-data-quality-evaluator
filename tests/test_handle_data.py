"""Baseline regression tests for utility/handle_data.py — captures CURRENT behavior as-is."""

import os
from unittest.mock import MagicMock, patch

import polars as pl
import pytest

from utility.handle_data import (
    _read_source,
    handle_environment_variables,
    handle_schema_overrides,
    read_source,
)


# ---------------------------------------------------------------------------
# handle_environment_variables
# ---------------------------------------------------------------------------


class TestHandleEnvironmentVariables:
    def test_str_no_dollar_returns_as_is(self):
        assert handle_environment_variables("plain_value") == "plain_value"

    def test_str_with_dollar_env_exists(self, monkeypatch):
        monkeypatch.setenv("MY_VAR", "secret")
        assert handle_environment_variables("$MY_VAR") == "secret"

    def test_str_with_dollar_env_missing_returns_name_without_dollar(self, monkeypatch):
        monkeypatch.delenv("MISSING_VAR", raising=False)
        # Current behavior: strips "$" and returns the bare name
        assert handle_environment_variables("$MISSING_VAR") == "MISSING_VAR"

    def test_dict_resolves_dollar_values(self, monkeypatch):
        monkeypatch.setenv("KEY1", "val1")
        result = handle_environment_variables({"a": "$KEY1", "b": "literal"})
        assert result == {"a": "val1", "b": "literal"}

    def test_dict_missing_env_returns_name_without_dollar(self, monkeypatch):
        monkeypatch.delenv("ABSENT", raising=False)
        result = handle_environment_variables({"x": "$ABSENT"})
        assert result == {"x": "ABSENT"}

    def test_dict_non_string_value_passes_through(self):
        result = handle_environment_variables({"n": 42, "flag": True})
        assert result == {"n": 42, "flag": True}

    def test_dict_empty_returns_empty(self):
        assert handle_environment_variables({}) == {}

    def test_unsupported_type_returns_value_unchanged(self):
        # Non-str, non-dict input: logs warning, returns the value as-is
        result = handle_environment_variables(123)
        assert result == 123

    def test_unsupported_list_returns_value_unchanged(self):
        lst = ["$SOMETHING"]
        result = handle_environment_variables(lst)
        assert result is lst


# ---------------------------------------------------------------------------
# handle_schema_overrides
# ---------------------------------------------------------------------------


class TestHandleSchemaOverrides:
    def test_none_returns_none(self):
        assert handle_schema_overrides(None) is None

    def test_valid_string_types_mapped(self):
        result = handle_schema_overrides({
            "col_a": "String",
            "col_b": "Date",
            "col_c": "Datetime",
            "col_d": "Categorical",
        })
        assert result == {
            "col_a": pl.String,
            "col_b": pl.Date,
            "col_c": pl.Datetime,
            "col_d": pl.Categorical,
        }

    def test_unknown_type_string_is_skipped(self):
        result = handle_schema_overrides({"col": "Float64", "valid": "String"})
        assert "col" not in result
        assert result["valid"] == pl.String

    def test_empty_dict_returns_empty_dict(self):
        assert handle_schema_overrides({}) == {}

    def test_non_dict_non_none_returns_none_implicitly(self):
        # No explicit return in the else branch → None
        assert handle_schema_overrides("String") is None

    def test_list_input_returns_none_implicitly(self):
        assert handle_schema_overrides(["String"]) is None

    def test_all_unknown_types_returns_empty_dict(self):
        result = handle_schema_overrides({"a": "Int32", "b": "Float64"})
        assert result == {}


# ---------------------------------------------------------------------------
# _read_source (private — tested directly for full branch coverage)
# ---------------------------------------------------------------------------


class TestReadSourceInternal:
    def _mock_lazy(self):
        lf = MagicMock(spec=pl.LazyFrame)
        return lf

    def test_csv_by_extension_calls_scan_csv(self):
        mock_lf = self._mock_lazy()
        with patch("polars.scan_csv", return_value=mock_lf) as mock_scan:
            result = _read_source("data.csv", None, {"opt": "val"}, {"col": pl.String})
        mock_scan.assert_called_once_with(
            "data.csv",
            schema_overrides={"col": pl.String},
            storage_options={"opt": "val"},
        )
        assert result is mock_lf

    def test_parquet_by_extension_calls_scan_parquet(self):
        mock_lf = self._mock_lazy()
        with patch("polars.scan_parquet", return_value=mock_lf) as mock_scan:
            result = _read_source("data.parquet", None, {}, None)
        mock_scan.assert_called_once_with("data.parquet", storage_options={})
        assert result is mock_lf

    def test_iceberg_by_extension_calls_scan_iceberg(self):
        mock_lf = self._mock_lazy()
        with patch("polars.scan_iceberg", return_value=mock_lf) as mock_scan:
            result = _read_source("table.iceberg", None, {}, None)
        mock_scan.assert_called_once_with("table.iceberg", storage_options={})
        assert result is mock_lf

    def test_xlsx_by_extension_calls_read_excel_then_lazy(self):
        mock_df = MagicMock()
        mock_lf = self._mock_lazy()
        mock_df.lazy.return_value = mock_lf
        with patch("polars.read_excel", return_value=mock_df) as mock_read:
            result = _read_source("data.xlsx", None, {}, {"col": pl.Date})
        mock_read.assert_called_once_with("data.xlsx", schema_overrides={"col": pl.Date})
        mock_df.lazy.assert_called_once()
        assert result is mock_lf

    def test_explicit_file_format_overrides_extension(self):
        mock_lf = self._mock_lazy()
        with patch("polars.scan_csv", return_value=mock_lf) as mock_scan:
            # Source ends in .parquet but format is forced to csv
            result = _read_source("data.parquet", "csv", {}, None)
        mock_scan.assert_called_once_with(
            "data.parquet", schema_overrides=None, storage_options={}
        )
        assert result is mock_lf

    def test_explicit_file_format_uppercase_causes_key_error(self):
        # .lower() is used for the membership check but the ORIGINAL key is used for lookup
        # → "CSV".lower() == "csv" passes the check, but read_source_func["CSV"] raises KeyError
        with pytest.raises(KeyError):
            _read_source("data.txt", "CSV", {}, None)

    def test_unknown_extension_raises_system_exit(self):
        with pytest.raises(SystemExit) as exc:
            _read_source("data.json", None, {}, None)
        assert "Unable to determine file format" in str(exc.value)
        assert "data.json" in str(exc.value)

    def test_unknown_explicit_format_falls_to_else_and_raises_type_error(self):
        # file_format is a known string branch path but not in read_source_func
        # → read_func stays None → None(...) in else branch → TypeError
        with pytest.raises(TypeError):
            _read_source("data.txt", "jsonl", {}, None)

    def test_csv_schema_overrides_none_passed_through(self):
        mock_lf = self._mock_lazy()
        with patch("polars.scan_csv", return_value=mock_lf) as mock_scan:
            _read_source("data.csv", None, None, None)
        mock_scan.assert_called_once_with(
            "data.csv", schema_overrides=None, storage_options=None
        )

    def test_source_uppercase_extension_is_matched(self):
        # source.lower() is used in the comparison so ".CSV" DOES match ".csv"
        mock_lf = self._mock_lazy()
        with patch("polars.scan_csv", return_value=mock_lf) as mock_scan:
            result = _read_source("data.CSV", None, {}, None)
        mock_scan.assert_called_once_with(
            "data.CSV", schema_overrides=None, storage_options={}
        )
        assert result is mock_lf


# ---------------------------------------------------------------------------
# read_source (public, decorated with exception_handler)
# ---------------------------------------------------------------------------


class TestReadSource:
    def test_non_dict_source_raises_system_exit(self):
        with pytest.raises(SystemExit) as exc:
            read_source("not_a_dict")
        assert "Source specification must be a dictionary" in str(exc.value)

    def test_list_source_raises_system_exit(self):
        with pytest.raises(SystemExit) as exc:
            read_source(["file_path", "data.csv"])
        assert "Source specification must be a dictionary" in str(exc.value)

    def test_missing_both_keys_raises_system_exit(self):
        with pytest.raises(SystemExit) as exc:
            read_source({"some_key": "value"})
        assert "cannot be read" in str(exc.value)

    def test_query_without_uri_falls_through_to_system_exit(self):
        # source.get("query") is truthy but source.get("uri") is falsy → skips db branch
        with pytest.raises(SystemExit) as exc:
            read_source({"query": "SELECT 1"})
        assert "cannot be read" in str(exc.value)

    def test_uri_without_query_falls_through_to_system_exit(self):
        with pytest.raises(SystemExit) as exc:
            read_source({"uri": "postgresql://localhost/db"})
        assert "cannot be read" in str(exc.value)

    def test_file_path_calls_read_source_internal(self):
        mock_lf = MagicMock(spec=pl.LazyFrame)
        with patch("utility.handle_data._read_source", return_value=mock_lf) as mock_rs:
            result = read_source({"file_path": "data.csv"})
        mock_rs.assert_called_once_with("data.csv", None, {}, None)
        assert result is mock_lf

    def test_file_path_with_file_format_passed_through(self):
        mock_lf = MagicMock(spec=pl.LazyFrame)
        with patch("utility.handle_data._read_source", return_value=mock_lf) as mock_rs:
            read_source({"file_path": "data.txt", "file_format": "csv"})
        mock_rs.assert_called_once_with("data.txt", "csv", {}, None)

    def test_file_path_with_storage_options_resolved(self, monkeypatch):
        monkeypatch.setenv("MY_KEY", "resolved_key")
        mock_lf = MagicMock(spec=pl.LazyFrame)
        with patch("utility.handle_data._read_source", return_value=mock_lf) as mock_rs:
            read_source({
                "file_path": "s3://bucket/data.parquet",
                "storage_options": {"key": "$MY_KEY"},
            })
        mock_rs.assert_called_once_with(
            "s3://bucket/data.parquet", None, {"key": "resolved_key"}, None
        )

    def test_file_path_with_schema_overrides_resolved(self):
        mock_lf = MagicMock(spec=pl.LazyFrame)
        with patch("utility.handle_data._read_source", return_value=mock_lf) as mock_rs:
            read_source({
                "file_path": "data.csv",
                "schema_overrides": {"col": "Date"},
            })
        mock_rs.assert_called_once_with("data.csv", None, {}, {"col": pl.Date})

    def test_db_path_calls_read_database_uri(self):
        mock_df = pl.DataFrame({"a": [1, 2]})
        with patch("polars.read_database_uri", return_value=mock_df) as mock_db:
            result = read_source({
                "query": "SELECT 1",
                "uri": "postgresql://localhost/db",
            })
        mock_db.assert_called_once_with(
            query="SELECT 1",
            uri="postgresql://localhost/db",
        )
        # Result should be a LazyFrame (called .lazy() on the DataFrame)
        assert isinstance(result, pl.LazyFrame)

    def test_db_uri_env_variable_resolved(self, monkeypatch):
        monkeypatch.setenv("DB_URI", "postgresql://host/db")
        mock_df = pl.DataFrame({"a": [1]})
        with patch("polars.read_database_uri", return_value=mock_df) as mock_db:
            read_source({"query": "SELECT 1", "uri": "$DB_URI"})
        mock_db.assert_called_once_with(
            query="SELECT 1",
            uri="postgresql://host/db",
        )

    def test_empty_dict_raises_system_exit(self):
        with pytest.raises(SystemExit) as exc:
            read_source({})
        assert "cannot be read" in str(exc.value)

    def test_empty_query_string_skips_db_branch(self):
        # get("query") returns "" which is falsy → skips db branch
        with pytest.raises(SystemExit) as exc:
            read_source({"query": "", "uri": "postgresql://localhost/db"})
        assert "cannot be read" in str(exc.value)
