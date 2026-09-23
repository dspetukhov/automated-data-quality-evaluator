"""Read CSV, XLSX, Parquet, or Iceberg file formats and PostgreSQL databases."""

import os
from typing import Any

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
def read_source(source: dict[str, Any]) -> pl.LazyFrame:
    """Read a data source into a Polars LazyFrame.

    Supports CSV, Parquet, Iceberg, XLSX (file-based), and PostgreSQL
    (URI-based). Cloud-storage credentials in `storage_options` and `uri`
    are resolved from environment variables when prefixed with `$`.

    Args:
        source (dict[str, Any]): Data source specification dict defined by
            `"source"` key in the configuration. Can read:

            - Files: must contain `"file_path"` (str). Optional keys:
              `"file_format"` (str, one of `"csv"`, `"parquet"`,
              `"iceberg"`, `"xlsx"`, case-insensitive; inferred from the
              `"file_path"` extension when omitted), `"storage_options"`
              (dict[str, str], applied for `"csv"`, `"parquet"`, and
              `"iceberg"` only, ignored for `"xlsx"`),
              `"schema_overrides"` (dict[str, str], applied for `"csv"`
              and `"xlsx"` only, ignored for `"parquet"` and `"iceberg"`).
            - Databases: must contain both `"query"` (str, SQL query) and
              `"uri"` (str, PostgreSQL connection URI).

    Returns:
        pl.LazyFrame containing the loaded data, ready for processing.

    Raises:
        SystemExit:
            - If `source` is not a `dict`.
            - If neither `"file_path"` nor the `"query"`/`"uri"` pair is present.
            - If `"file_format"` (or the extension inferred from `"file_path"`)
                does not match a supported format.
            - If the underlying read operation raises any exception (converted to
                `SystemExit` by the `exception_handler` decorator).
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
        source (str): Path or URL to the file from `"source.file_path"` key
            in the configuration; used for extension-based detection
            when `file_format` is `None`.
        file_format (str | None): Explicit format string from
            `"source.file_format"` key in the configuration or `None` for
            auto-detection.

    Returns:
        Lowercase format string, guaranteed to be a key in `_READERS`.

    Raises:
        SystemExit:
            - If `file_format` is an unrecognised string.
            - If `file_format` is `None` and `source` extension
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


def handle_schema_overrides(
    data: dict[str, str] | None,
) -> dict[str, pl.DataType] | None:
    """Map string type names to `pl.DataType` instances.

    Unknown type strings are skipped with a warning. Non-`dict`, non-`None`
    input logs a warning and returns `None`. Supported type names
    (case-sensitive, matched exactly): `"String"`, `"Date"`, `"Datetime"`,
    `"Categorical"`.

    Args:
        data (dict[str, str] | None): Mapping of column name to type-name
            string from `"source.schema_overrides"` in the configuration.

    Returns:
        A `dict[str, pl.DataType]` mapping column names to Polars types,
        or `None` if `data` is not a `dict`.
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
    else:
        logging.warning(f"'schema_overrides' expected dict, got {type(data).__name__}")
        return None


def handle_environment_variables(
    params: str | dict[str, str],
) -> str | dict[str, str | None]:
    """Resolve `$VAR` placeholders in a string or dict of strings.

    Each string value starting with `$` has the leading `$` stripped and
    the remainder looked up in `os.environ`. If the variable is set the
    value is replaced; if not, `None` is returned for that placeholder and
    a warning is logged. Strings not starting with `$` are returned
    unchanged. For unsupported input types the original value is returned
    as-is with a warning.

    Args:
        params (str | dict[str, str]): A `str` or `dict[str, Any]`
            (typically `"source.uri"` or `"source.storage_options"` value
            from the configuration) or any other type.

    Returns:
        - For `str` input: the resolved `str`, or `None` if the
        referenced environment variable is absent.
        - For `dict` input: a new `dict` with the same keys and each
        string value resolved (`str | None`); non-string values are
        unchanged.
        - For any other type: the original value unchanged.
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
