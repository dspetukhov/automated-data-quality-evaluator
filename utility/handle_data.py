"""Data source reading utilities for the automated data quality evaluator."""

import os

import polars as pl

from .handle_exceptions import exception_handler
from .setup_logging import logging


@exception_handler(exit_on_error=True)
def read_source(source: dict[str, str]) -> pl.LazyFrame:
    """Read a data source into a Polars LazyFrame.

    Supports CSV, Parquet, Iceberg, XLSX (file-based), and PostgreSQL
    (URI-based). Cloud-storage credentials in ``storage_options`` and ``uri``
    are resolved from environment variables when prefixed with ``$``.

    Args:
        source: Data source specification dict. Accepted shapes:

            - File-based: must contain ``"file_path"`` (str). Optional keys:
              ``"file_format"`` (str), ``"storage_options"`` (dict),
              ``"schema_overrides"`` (dict[str, str]).
            - Database: must contain both ``"query"`` (str) and ``"uri"``
              (str, PostgreSQL connection URI).

    Returns:
        pl.LazyFrame containing the loaded data, ready for processing.

    Raises:
        SystemExit: If ``source`` is not a dict, if neither ``"file_path"``
            nor the ``"query"``/``"uri"`` pair is present, or if the
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

    elif source.get("file_path"):
        logging.info(f"Data to read: {source['file_path']}")

        # Get storage_options to read from cloud providers
        storage_options = handle_environment_variables(
            source.get("storage_options", {})
        )
        # Get schema_overrides to alter schema dtypes for csv / xlsx
        schema_overrides = handle_schema_overrides(source.get("schema_overrides"))

        lf = _read_source(
            source["file_path"],
            source.get("file_format"),
            storage_options,
            schema_overrides,
        )

    if lf is None:
        raise SystemExit(
            f"Specified source cannot be read: {source}, "
            "expected 'file_path' or 'query' / 'uri' keys."
        )

    return lf


def _read_source(
    source: str,
    file_format: str | None,
    storage_options: dict[str, str] | None,
    schema_overrides: dict[str, str] | None,
) -> pl.LazyFrame:
    """Select and call the appropriate Polars reader for a file source.

    When ``file_format`` is provided it is normalised to lowercase and
    looked up directly. When ``None``, the file extension of ``source``
    is matched case-insensitively against the supported formats.

    Args:
        source: Path or URL to the file.
        file_format: Explicit format override (``"csv"``, ``"xlsx"``,
            ``"parquet"``, or ``"iceberg"``; case-insensitive). Pass
            ``None`` to auto-detect from the file extension.
        storage_options: Credentials or reader options forwarded to the
            underlying Polars reader for cloud-storage sources. Ignored
            for XLSX.
        schema_overrides: Column-name-to-Polars-type mapping forwarded to
            the reader. Applied to CSV and XLSX only.

    Returns:
        pl.LazyFrame containing the data read from ``source``.

    Raises:
        SystemExit: If ``file_format`` is a string not in the supported
            set, or if ``file_format`` is ``None`` and the extension of
            ``source`` does not match a supported format (csv, xlsx,
            parquet, iceberg).
    """
    # {file format: read function} mapping
    read_source_func = {
        "xlsx": pl.read_excel,
        "csv": pl.scan_csv,
        "parquet": pl.scan_parquet,
        "iceberg": pl.scan_iceberg,
    }
    lf, read_func = None, None

    if isinstance(file_format, str):
        file_format_lower = file_format.lower()
        if file_format_lower in read_source_func:
            read_func = read_source_func[file_format_lower]
            file_format = file_format_lower
        else:
            raise SystemExit(
                f"Unsupported file format '{file_format}', "
                f"supported formats: csv, xlsx, parquet, iceberg"
            )
    else:
        # Try to match source ending with supported file formats
        for ff, rf in read_source_func.items():
            if source.lower().endswith(f".{ff}"):
                logging.info(f"Identified file format: {ff}")
                file_format, read_func = ff, rf

        if read_func is None:
            raise SystemExit(
                f"Unable to determine file format for: {source}, "
                f"supported formats: csv, xlsx, parquet, iceberg"
            )

    if file_format == "xlsx":
        lf = read_func(source, schema_overrides=schema_overrides).lazy()
    elif file_format == "csv":
        lf = read_func(
            source, schema_overrides=schema_overrides, storage_options=storage_options
        )
    else:
        lf = read_func(source, storage_options=storage_options)

    return lf


def handle_schema_overrides(data: dict[str, str]) -> dict[str, pl.DataType]:
    """Map string type names to Polars DataType instances.

    Unknown type strings are skipped with a warning. Non-dict, non-``None``
    input logs a warning and returns ``None``.

    Supported type names: ``"String"``, ``"Date"``, ``"Datetime"``,
    ``"Categorical"``.

    Args:
        data: Mapping of column name to type-name string, or ``None`` to
            opt out of schema overrides. Any other non-dict type is treated
            the same as ``None``.

    Returns:
        A ``dict[str, pl.DataType]`` mapping column names to Polars types,
        or ``None`` if ``data`` is ``None`` or not a ``dict``.
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
    """Resolve ``$VAR`` placeholders in a string or dict of strings.

    Each string value starting with ``$`` has the leading ``$`` stripped and
    the remainder looked up in ``os.environ``. If the variable is set the
    value is replaced; if not, ``None`` is returned for that placeholder and
    a warning is logged. Strings not starting with ``$`` are returned
    unchanged. For unsupported input types the original value is returned
    as-is with a warning.

    Args:
        params: A ``str``, a ``dict[str, Any]``, or any other type.
            Dict values that are not strings are passed through unchanged.

    Returns:
        For ``str`` input: the resolved ``str``, or ``None`` if the
        referenced environment variable is absent.
        For ``dict`` input: a new ``dict`` with the same keys and each
        string value resolved (``str | None``); non-string values are
        unchanged.
        For any other type: the original value unchanged.
    """

    def get_environment_variable(value: str) -> str | None:
        if value.startswith("$"):
            value = value[1:]
            if value in os.environ:
                logging.info(f"Environment variable for '{value}' found")
                return os.getenv(value)
            else:
                logging.warning(f"Environment variable for '{value}' not found")
                return None
        else:
            return value

    if isinstance(params, str):
        return get_environment_variable(params)
    elif isinstance(params, dict):
        output = {}
        for key, value in params.items():
            if isinstance(value, str):
                output[key] = get_environment_variable(value)
            else:
                output[key] = value
        return output
    else:
        logging.warning(
            "Unsupported input type for 'storage_options' or 'uri': "
            f"expected dict or str, got {type(params).__name__}"
        )
        return params
