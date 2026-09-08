"""Data source reading utilities for the automated data quality evaluator."""

import os

import polars as pl

from .handle_exceptions import exception_handler
from .setup_logging import logging

# Supported file formats and their corresponding Polars reader functions.
_READERS = {
    "xlsx": pl.read_excel,
    "csv": pl.scan_csv,
    "parquet": pl.scan_parquet,
    "iceberg": pl.scan_iceberg,
}

_SUPPORTED_FORMATS = ", ".join(_READERS)


@exception_handler(exit_on_error=True)
def read_source(source: dict[str, str]) -> pl.LazyFrame:
    """Read a data source into a Polars LazyFrame.

    Supports CSV, Parquet, Iceberg, XLSX (file-based), and PostgreSQL
    (URI-based). Cloud-storage credentials in `storage_options` and `uri`
    are resolved from environment variables when prefixed with `$`.

    Args:
        source: Data source specification dict. Accepted shapes:

            - File-based: must contain `"file_path"` (str). Optional keys:
              `"file_format"` (str), `"storage_options"` (dict),
              `"schema_overrides"` (dict[str, str]).
            - Database: must contain both `"query"` (str) and `"uri"`
              (str, PostgreSQL connection URI).

    Returns:
        pl.LazyFrame containing the loaded data, ready for processing.

    Raises:
        SystemExit: If `source` is not a dict, if neither `"file_path"`
            nor the `"query"`/`"uri"` pair is present, or if the
            underlying read operation raises an exception.
    """
    if not isinstance(source, dict):
        raise SystemExit(
            f"Source specification must be a dictionary, got: {type(source).__name__}"
        )

    lf = None

    # Read from PostgreSQL database
    if source.get("query") and source.get("uri"):
        logging.info(f"Data to read: {source['query']}")
        lf = pl.read_database_uri(
            query=source["query"], uri=handle_environment_variables(source["uri"])
        ).lazy()

    # Read from file/directory
    elif source.get("file_path"):
        logging.info(f"Data to read: {source['file_path']}")

        storage_options = handle_environment_variables(
            source.get("storage_options", {})
        )
        schema_overrides = handle_schema_overrides(source.get("schema_overrides"))
        file_path = source["file_path"]
        file_format = _resolve_file_format(file_path, source.get("file_format"))
        read_func = _READERS[file_format]

        if file_format == "xlsx":
            lf = read_func(file_path, schema_overrides=schema_overrides).lazy()
        elif file_format == "csv":
            lf = read_func(
                file_path,
                schema_overrides=schema_overrides,
                storage_options=storage_options,
            )
        else:
            lf = read_func(file_path, storage_options=storage_options)

    if lf is None:
        raise SystemExit(
            f"Specified source cannot be read: {source}, "
            "expected 'file_path' or 'query' / 'uri' keys."
        )

    return lf


def _resolve_file_format(source: str, file_format: str | None) -> str:
    """Resolve and validate the file format for a given source.

    When `file_format` is a string it is normalised to lowercase and
    validated against the supported set. When `None`, the format is
    inferred from the file extension of `source` (case-insensitive).

    Args:
        source: Path or URL to the file; used for extension-based detection
            when `file_format` is `None`.
        file_format: Explicit format string or `None` for auto-detection.

    Returns:
        Lowercase format string, guaranteed to be a key in `_READERS`.

    Raises:
        SystemExit: If `file_format` is an unrecognised string, or if
            `file_format` is `None` and the extension of `source`
            does not match any supported format.
    """
    if isinstance(file_format, str):
        ff_lower = file_format.lower()
        if ff_lower not in _READERS:
            raise SystemExit(
                f"Unsupported file format '{file_format}', "
                f"supported formats: {_SUPPORTED_FORMATS}"
            )
        return ff_lower

    for ff_lower in _READERS:
        if source.lower().endswith(f".{ff_lower}"):
            logging.info(f"Identified file format: {ff_lower}")
            return ff_lower

    raise SystemExit(
        f"Unable to determine file format for: {source}, "
        f"supported formats: {_SUPPORTED_FORMATS}"
    )


def handle_schema_overrides(data: dict[str, str]) -> dict[str, pl.DataType]:
    """Map string type names to Polars DataType instances.

    Unknown type strings are skipped with a warning. Non-dict, non-`None`
    input logs a warning and returns `None`.

    Supported type names: `"String"`, `"Date"`, `"Datetime"`,
    `"Categorical"`.

    Args:
        data: Mapping of column name to type-name string, or `None` to
            opt out of schema overrides. Any other non-dict type is treated
            the same as `None`.

    Returns:
        A `dict[str, pl.DataType]` mapping column names to Polars types,
        or `None` if `data` is `None` or not a `dict`.
    """
    dtypes = {
        "String": pl.String,
        "Date": pl.Date,
        "Datetime": pl.Datetime,
        "Categorical": pl.Categorical,
    }

    if isinstance(data, dict):
        output = {}
        for key, value in data.items():
            if value in dtypes:
                output[key] = dtypes[value]
            else:
                logging.warning(f"Unsupported data type '{value}' for column '{key}'")
        return output
    elif data is None:
        return None
    else:
        logging.warning(f"'schema_overrides' expected dict, got {type(data).__name__}")
        return None


def handle_environment_variables(params: str | dict[str, str]) -> str | dict[str, str]:
    """Resolve `$VAR` placeholders in a string or dict of strings.

    Each string value starting with `$` has the leading `$` stripped and
    the remainder looked up in `os.environ`. If the variable is set the
    value is replaced; if not, `None` is returned for that placeholder and
    a warning is logged. Strings not starting with `$` are returned
    unchanged. For unsupported input types the original value is returned
    as-is with a warning.

    Args:
        params: A `str`, a `dict[str, Any]`, or any other type.
            Dict values that are not strings are passed through unchanged.

    Returns:
        For `str` input: the resolved `str`, or `None` if the
        referenced environment variable is absent.
        For `dict` input: a new `dict` with the same keys and each
        string value resolved (`str | None`); non-string values are
        unchanged.
        For any other type: the original value unchanged.
    """

    def _resolve_environment_variable(value: str) -> str | None:
        """Resolve a `$VAR` placeholder to its environment variable value, or return `value` unchanged."""
        if not value.startswith("$"):
            return value
        name = value[1:]
        if name in os.environ:
            logging.info(f"Environment variable for '{name}' found")
            return os.getenv(name)
        logging.warning(f"Environment variable for '{name}' not found")
        return None

    if isinstance(params, str):
        return _resolve_environment_variable(params)
    elif isinstance(params, dict):
        return {
            key: _resolve_environment_variable(value)
            if isinstance(value, str)
            else value
            for key, value in params.items()
        }
    else:
        logging.warning(
            f"Unsupported input type: expected dict or str, got {type(params).__name__}"
        )
        return params
