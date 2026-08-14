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
) -> tuple[pl.DataFrame, dict[str, str]]:
    """Preprocess data for evaluation through aggregation by dates.

    This function processes input data frame by applying filter
    and transformations as specified by SQL expressions in the configuration.
    Then it gets and validates date_column to ensure data aggregation by dates,
    collects aggregation expressions and metadata for each column,
    and performs aggregation.

    Args:
        lf (pl.LazyFrame): Input data.
        config (dict[str, Any]): Configuration dictionary specifying
            extra variables, filter, and transformations.

    Returns:
        tuple[pl.DataFrame, dict[str, str]]:
            - Aggregated data with descriptive statistics for each column.
            - Dictionary of columns indicating types of numeric columns.
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

    # Aggregate data by time intervals / buckets
    lf_agg = lf.group_by(TIME_INTERVAL_COL).agg(aggs).sort(TIME_INTERVAL_COL)
    # lf_agg.explain()  # uncomment to get the query plan or turn off/on optimizations
    lf_agg = lf_agg.collect(engine=config.get("engine", "auto"))
    return lf_agg, metadata


def apply_filter(lf: pl.LazyFrame, filter_str: str | None) -> pl.LazyFrame:
    """Apply filter to Polars LazyFrame.

    This function applies SQL expression to make a slice of data
    as specified in the configuration.

    Args:
        lf (pl.LazyFrame): Input data.
        filter_str (str | None): SQL expression to filter data.

    Returns:
        pl.LazyFrame: Filtered LazyFrame.
    """
    if isinstance(filter_str, str):
        lf = lf.sql(filter_str)
        logging.info(f"Filter applied: {filter_str}")
    return lf


def apply_transformations(
    lf: pl.LazyFrame, transformations: dict[str, str] | None
) -> pl.LazyFrame:
    """Apply transformations to Polars LazyFrame.

    This function applies transformations to alter LazyFrame columns
    as specified by SQL in the configuration.
    If key in a transformation matches existing column name,
    it replaces that column with transformed values,
    otherwise creates a new column.

    Args:
        lf (pl.LazyFrame): Input data.
        transformations (dict[str, str] | None): Dictionary where:
            each key is column name to be created or replaced,
            each value is SQL expression to transform LazyFrame.

    Returns:
        pl.LazyFrame: LazyFrame with transformed columns.
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
    """Check date_column presence in schema and validate its type.

    This function checks date_column type in schema, makes conversion
    to datetime type if necessary, and renames it to TIME_INTERVAL_COL
    ('__time_interval') to ensure consistency in data evaluation process.
    Division by time intervals implemented with polars.Expr.dt.truncate.
    Raises SystemExit if date_column is not found in schema.

    Args:
        lf (pl.LazyFrame): Input data.
        schema (pl.Schema): Data schema provided by Polars.
        date_column (str): Name of date column.
        time_interval (str): Time interval to truncate date_column,
            "1d" for one day by default, or "1h" for one hour, etc.

    Returns:
        tuple[pl.LazyFrame, pl.Schema]:
            - LazyFrame with date column processed.
            - Updated data schema.

    Raises:
        SystemExit: If no date column in data.
    """
    if not schema.get(date_column):
        raise SystemExit(f"Exit: no column '{date_column}' in data for preprocessing")

    if schema.get(date_column) in (pl.String, pl.Datetime, pl.Date):
        if schema.get(date_column) == pl.String:
            # Convert date_column of string type into Polars date type
            lf = lf.with_columns(pl.col(date_column).str.to_date(strict=True))

        # Divide date or datetime range into time intervals / buckets
        lf = lf.with_columns(pl.col(date_column).dt.truncate(time_interval))

        # Rename date_column as TIME_INTERVAL_COL for consistency across tool
        lf = lf.rename({date_column: TIME_INTERVAL_COL})
        logging.info(f"Date column: {date_column}")

        return lf, lf.collect_schema()
    else:
        raise SystemExit(
            f"Exit: 'date_column' type '{schema.get(date_column)}' is not supported"
        )


def collect_aggregations(
    schema: pl.Schema,
    target_column: str | None,
    columns_to_exclude: list[str],
    columns_to_exclude_extra_statistics: list[str],
) -> tuple[list[pl.Expr], dict[str, str | None]]:
    """Collect aggregation expressions.

    This function collects expressions to perform data aggregation by dates.
    The first expression calculates the number of values per date,
    the rest are column-wise statistics.
    Numeric columns marked in metadata dictionary by their types.

    Args:
        schema (pl.Schema): Data schema provided by Polars.
        target_column (str | None): Target column to compute target average.
        columns_to_exclude (list[str]): List of columns
            to be excluded from aggregation.

    Returns:
        tuple[list[pl.Expr], dict[str, str]]:
            - aggs (list[pl.Expr]): Aggregation expressions,
            - metadata (dict[str, str | None]): Dict of aggregated columns
                indicating types for numeric columns.
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
                col_expr = col_expr.cast(pl.String).str.len_chars().alias(col)
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
