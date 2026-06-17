import os

import polars as pl

from .handle_exceptions import exception_handler
from .setup_logging import logging


@exception_handler(exit_on_error=True)
def read_source(source: dict[str, str]) -> pl.LazyFrame:
    """Read specified source of data as Polars LazyFrame.

    Supports CSV, Parquet, Iceberg, XLSX, and database URIs.
    Returns LazyFrame for preprocessing. If loading fails,
    it terminates the main program.
    In case of reading from cloud providers `storage_options` definition
    is expected, otherwise Polars will try to infer credentials implicitly
    from environment variables, e.g. AWS_REGION in case of S3.
    `uri` and `storage_options` values can be read from environment variables
    if they are specified with a $ sign at the beginning.

    Args:
        source (dict[str, str]): Data source specification.
            Can be a dictionary with `query` and `uri` keys
            to read from a PostgreSQL database,
            or a dictionary with `file_path` key
            to read from file, Google Drive, S3
            (`storage_options` and `file_format` are optional).

    Returns:
        pl.LazyFrame: Polars lazy data frame.

    Raises:
        SystemExit: If data cannot be loaded or source specification is invalid.
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
    """Read source based on file format specified or based on source name ending.

    This function selects suitable read function
    based on file format specified. If it wasn't specified,
    it selects read function by matching source name ending
    with supported file formats. If read function is found,
    it reads source as Lazy data frame, otherwise SystemExit raised.

    Args:
        source (str): Path to the file to read.
        file_format (str | None): File format to read.
        storage_options (dict[str, str] | None): Credentials to read from cloud providers.
        schema_overrides (dict[str, str] | None): Mapping to change types of certain columns.

    Returns:
        pl.LazyFrame: Polars lazy data frame.

    Raises:
        SystemExit: If there is no read function for the file format provided.
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
    """Replace string data type representation into Polars data type.

    Args:
        data (dict[str, str]): Dict of types representation
            to be mapped with Polars data types.

    Returns:
        dict[str, pl.DataType]: Mapping of string data type representation
            to Polars data type.
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
    """Replace environment variable placeholders with actual values.

    Any value in `params` starting with "$" sign is considered as
    environment variable placeholders that will be replaced with
    the environment variable value.

    Args:
        params (str | dict[str, str]): Input parameters potentially
            containing environment variable placeholders.

    Returns:
        str | dict[str, str]: Updated parameters
            with environment variable placeholders replaced by their actual values.
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
