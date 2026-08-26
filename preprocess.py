"""Filters, transforms, and aggregates input data into per-time-interval descriptive statistics."""

from typing import Any

import polars as pl
import polars.selectors as cs

from utility import (
    PREFIX_COL,
    PREFIX_COL_E,
    TIME_INTERVAL_COL,
    exception_handler,
    logging,
)


@exception_handler(exit_on_error=True)
def make_preprocessing(
    lf: pl.LazyFrame, config: dict[str, Any]
) -> tuple[pl.DataFrame, dict[str, str | None]]:
    """Preprocess data for evaluation through aggregation by dates.

    Applies the configured filter and transformations, validates and divides
    date_column into time intervals, collects per-column aggregation
    expressions, and eagerly aggregates the LazyFrame by time interval.

    Args:
        lf (pl.LazyFrame): Input data.
        config (dict[str, Any]): Configuration dictionary. Reads the keys
            "filter", "transformations", "date_column" (default "date_column"),
            "time_interval" (default "1d"), "target_column" (default "target_column"),
            "columns_to_exclude", "columns_to_exclude_extra_statistics",
            "streaming_chunk_size" (int), and "engine" (default "auto").

    Returns:
        tuple[pl.DataFrame, dict[str, str | None]]:
            - Aggregated data with descriptive statistics per time interval.
            - Metadata dict mapping each column to its dtype as a string,
              or None for columns in columns_to_exclude_extra_statistics.

    Raises:
        SystemExit: Via the exception_handler decorator, if any exception is
            raised while preprocessing (e.g. an invalid or missing
            date_column, or a malformed filter/transformation expression).
    """
    # Apply filter for rows and columns
    lf = apply_filter(lf, config.get("filter"))
    # Apply transformations for columns
    lf = apply_transformations(lf, config.get("transformations"))

    # Get and print LazyFrame schema
    schema = lf.collect_schema()
    schema_str = "\n".join(f"{col}: {dtype}" for col, dtype in schema.items())
    logging.info(f"Data schema:\n{schema_str}")

    # Prepare date_column for data aggregation
    date_column = config.get("date_column", "date_column")
    lf, schema = process_date_column(
        lf, schema, date_column, config.get("time_interval", "1d")
    )

    # Get target_column
    target_column = config.get("target_column", "target_column")
    if schema.get(target_column):
        logging.info(f"Target column: {target_column}")
    else:
        target_column = None
        logging.warning("Target column not found")

    # Collect aggregation expressions for each column except excluded ones
    aggs, metadata = collect_aggregations(
        schema,
        target_column,
        config.get("columns_to_exclude", []),
        config.get("columns_to_exclude_extra_statistics", []),
    )

    # Set chunk size used in streaming engine
    if isinstance(config.get("streaming_chunk_size"), int):
        pl.Config.set_streaming_chunk_size(config["streaming_chunk_size"])

    # Aggregate data by time intervals
    lf_agg = lf.group_by(TIME_INTERVAL_COL).agg(aggs).sort(TIME_INTERVAL_COL)
    # lf_agg.explain()  # uncomment to get the query plan or turn off/on optimizations
    lf_agg = lf_agg.collect(engine=config.get("engine", "auto"))
    return lf_agg, metadata


def apply_filter(lf: pl.LazyFrame, filter_str: str | None) -> pl.LazyFrame:
    """Apply a SQL filter expression to a Polars LazyFrame.

    If filter_str is not a string (e.g. None, absent from config),
    LazyFrame is returned unchanged.

    Args:
        lf (pl.LazyFrame): Input data.
        filter_str (str | None): SQL expression to filter LazyFrame data.

    Returns:
        pl.LazyFrame: Filtered LazyFrame, or the original one
        if filter_str is not a string.
    """
    if isinstance(filter_str, str):
        lf = lf.sql(filter_str)
        logging.info(f"Filter applied: {filter_str}")
    return lf


def apply_transformations(
    lf: pl.LazyFrame, transformations: dict[str, str] | None
) -> pl.LazyFrame:
    """Apply SQL-expression transformations to a Polars LazyFrame.

    Each transformation is added via with_columns(pl.sql_expr(expr).alias(
    alias)); if alias matches an existing column name it replaces that
    column, otherwise it creates a new one. If transformations is not a
    dict (e.g. None, absent from config), LazyFrame is returned unchanged.

    Args:
        lf (pl.LazyFrame): Input data.
        transformations (dict[str, str] | None): Mapping of column name
            (created or replaced) to a string of SQL code evaluated via
            pl.sql_expr().

    Returns:
        pl.LazyFrame: LazyFrame with transformed columns, or the original one
        if transformations is not a dict.
    """
    if isinstance(transformations, dict):
        # Iterate over transformations specified in configuration file
        for alias, expr in transformations.items():
            lf = lf.with_columns(pl.sql_expr(expr).alias(alias))
            logging.info(f"Transformation applied: {expr}")

    return lf


def process_date_column(
    lf: pl.LazyFrame, schema: pl.Schema, date_column: str, time_interval: str
) -> tuple[pl.LazyFrame, pl.Schema]:
    """Validate date_column, divide it into time intervals, and rename it.

    Checks that date_column is present in schema and is of a supported type
    (String, Datetime, or Date), converts it to a Date if it is a String,
    divides it into time intervals via pl.Expr.dt.truncate(), then renames
    it to TIME_INTERVAL_COL for consistency within the tool.

    Args:
        lf (pl.LazyFrame): Input data.
        schema (pl.Schema): Schema of input data.
        date_column (str): Name of the date/datetime column to process.
        time_interval (str): Interval size to divide date/datetime column,
            e.g. "1d" for one day or "1h" for one hour.

    Returns:
        tuple[pl.LazyFrame, pl.Schema]:
            - LazyFrame with date_column divided into time intervals and
                renamed to TIME_INTERVAL_COL.
            - Schema of the returned LazyFrame.

    Raises:
        SystemExit: If date_column is absent from schema,
            or its dtype is not one of String, Datetime, or Date.
    """
    date_dtype = schema.get(date_column)

    if date_dtype is None:
        raise SystemExit(f"Exit: no column '{date_column}' in data for preprocessing")

    if date_dtype in (pl.String, pl.Datetime, pl.Date):
        if date_dtype == pl.String:
            # Convert date_column of string type into Polars date type
            lf = lf.with_columns(pl.col(date_column).str.to_date(strict=True))

        # Divide date or datetime range into time intervals
        lf = lf.with_columns(pl.col(date_column).dt.truncate(time_interval))

        # Rename date_column as TIME_INTERVAL_COL for consistency
        lf = lf.rename({date_column: TIME_INTERVAL_COL})
        logging.info(f"Date column: {date_column}")

        return lf, lf.collect_schema()
    else:
        raise SystemExit(f"Exit: 'date_column' type '{date_dtype}' is not supported")


def collect_aggregations(
    schema: pl.Schema,
    target_column: str | None,
    columns_to_exclude: list[str],
    columns_to_exclude_extra_statistics: list[str],
) -> tuple[list[pl.Expr], dict[str, str | None]]:
    """Collect per-time-interval aggregation expressions for each column.

    Always includes an expression for the row count per interval, and, if
    target_column is set, its mean. For every remaining column (excluding
    TIME_INTERVAL_COL and columns_to_exclude), adds "Number of unique
    values" (n_unique() of non-null values, computed via drop_nulls() so
    nulls are not counted as a distinct value) and "Proportion of missing values"
    (mean of is_null()). Unless the column is in
    columns_to_exclude_extra_statistics, also adds min/max/mean/median/std;
    for non-numeric columns these are computed on string length
    (cast to String then str.len_chars()) rather than on the values
    themselves.

    Args:
        schema (pl.Schema): Schema of LazyFrame to be aggregated.
        target_column (str | None): Column to calculate target average per time interval,
            or None to skip it.
        columns_to_exclude (list[str]): Columns excluded from processing entirely.
        columns_to_exclude_extra_statistics (list[str]): Columns for which
            min/max/mean/median/std statistics will not be calculated.

    Returns:
        tuple[list[pl.Expr], dict[str, str | None]]:
            - aggs: Aggregation expressions for LazyFrame.agg().
            - metadata: Maps each processed column to its dtype as a string,
              or None if the column is in columns_to_exclude_extra_statistics.
    """
    # Start with common aggregation expression for the number of values
    aggs = [pl.len().alias(" __Number of values")]

    # If target column is found in schema,
    # calculate its mean (i.e. class balance in binary classification problems)
    if target_column:
        aggs.append(pl.col(target_column).mean().alias(" __Target average"))

    metadata = {}

    for col in schema.names():
        if col == TIME_INTERVAL_COL or col in columns_to_exclude:
            continue
        # Add common statistics for the column
        aggs.extend(
            [
                pl.col(col)
                .drop_nulls()
                .n_unique()
                .alias(f"{PREFIX_COL} {col} __Number of unique values"),
                pl.col(col)
                .is_null()
                .mean()
                .alias(f"{PREFIX_COL} {col} __Proportion of missing values"),
            ]
        )

        if col in columns_to_exclude_extra_statistics:
            metadata[col] = None
        else:
            col_expr = pl.col(col)
            # Add extra statistics
            if col not in cs.expand_selector(schema, cs.numeric()):
                col_expr = col_expr.cast(pl.String).str.len_chars()
            aggs.extend(
                [
                    col_expr.min().alias(f"{PREFIX_COL_E} {col} __Min"),
                    col_expr.max().alias(f"{PREFIX_COL_E} {col} __Max"),
                    col_expr.mean().alias(f"{PREFIX_COL_E} {col} __Mean"),
                    col_expr.median().alias(f"{PREFIX_COL_E} {col} __Median"),
                    col_expr.std().alias(f"{PREFIX_COL_E} {col} __Standard deviation"),
                ]
            )
            metadata[col] = str(schema[col])

    return aggs, metadata
