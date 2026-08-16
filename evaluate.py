"""Calculates per-column descriptive statistics and detects outliers (IQR/Z-score)."""

from typing import Any

from polars import DataFrame, Series

from utility import exception_handler


@exception_handler()
def evaluate_data(
    data: DataFrame, config: dict[str, str | float]
) -> tuple[list[dict[str, Any]], list[tuple[float | None, float | None]]]:
    """Calculates descriptive statistics and outlier counts for each column in data.

    Skips the first column of `data` (assumed to be the time-interval index).
    For every other column, computes mean, standard deviation, min, max,
    quartiles, range, IQR, and outlier percentages (delegating outlier
    counting and chart bounds to `evaluate_data_outliers`). Outlier
    percentages are relative to each column's non-null count, and degenerate
    cases (all-null column, zero rows) resolve to `None`/`0.0` values instead
    of raising.

    Args:
        data (DataFrame): Aggregated data, e.g. as produced by
            `preprocess.make_preprocessing`. The first column is treated as
            the time-interval index and is not evaluated; remaining columns
            must support `.mean()`, `.std()`, `.quantile()`, `.min()`,
            `.max()` (numeric dtypes, or string length for text columns).
        config (dict[str, str | float]): Parameters for detecting outliers,
            forwarded to `evaluate_data_outliers`:
            - 'criterion' (str): IQR or Z-score,
            - 'multiplier_iqr' (float): multiplier for IQR criterion (defaults to 1.5).
            - 'threshold_z_score' (float): threshold for Z-score criterion (defaults to 3.0).

    Returns:
        tuple[list[dict[str, Any]], list[tuple[float | None, float | None]]]:
            - List of dictionaries with description of each column in data:
                - name of the column,
                - mean and standard deviation,
                - range of values (`None` if the column has no non-null values),
                - Q1, Q3, and IQR (`None` if the column has no non-null values),
                - outliers percentage (of non-null values) according to IQR
                  and Z-score criteria (`0.0` if the column has no non-null values).
            - List of boundaries for outliers to be highlighted on a chart,
              aligned by index with the dictionaries above.

    Note:
        Decorated with `@exception_handler()`: on any exception, logs the
        error and returns `data` unchanged instead of the documented tuple.
    """
    data_evals, outliers_bounds = [], []

    # Evaluate each column in data, skip first time interval column
    for col in data.columns[1:]:
        # Calculate mean and standard deviation, first and third quartile
        mean, std = data[col].mean(), data[col].std()
        min_val, max_val = data[col].min(), data[col].max()
        q1, q3 = data[col].quantile(0.25), data[col].quantile(0.75)

        # Handle TypeError in case of all-null column
        range_val = None if min_val is None or max_val is None else max_val - min_val
        iqr_val = None if q1 is None or q3 is None else q3 - q1

        outliers_iqr, outliers_zscore, bounds = evaluate_data_outliers(
            data[col], mean, std, q1, q3, config
        )

        # Subtract nulls from total row count (denominator) to get correct percentages
        # as the number of outliers (numerator) is counted over only non-null values
        non_null_count = data[col].len() - data[col].null_count()

        # Handle ZeroDivisionError in case of all-null column
        if non_null_count:
            outliers_iqr_pct = 100 * outliers_iqr / non_null_count
            outliers_zscore_pct = 100 * outliers_zscore / non_null_count
        else:
            outliers_iqr_pct, outliers_zscore_pct = 0.0, 0.0

        data_evals.append(
            {
                "title": col.split(" __")[-1],
                "μ±σ": (mean, std),
                "Range [Min]": min_val,
                "Range [Max]": max_val,
                "Range": range_val,
                "IQR [Q1]": q1,
                "IQR [Q3]": q3,
                "IQR": iqr_val,
                "Outliers [IQR]": outliers_iqr_pct,
                "Outliers [Z-score]": outliers_zscore_pct,
            }
        )
        outliers_bounds.append(bounds)

    return data_evals, outliers_bounds


def evaluate_data_outliers(
    data: Series,
    mean: float,
    std: float,
    q1: float | None,
    q3: float | None,
    config: dict[str, str | float],
) -> tuple[int, int, tuple[float | None, float | None]]:
    """Counts IQR and Z-score outliers in a column and derives chart bounds.

    Returns immediately with `(0, 0, (None, None))` if `q1` or `q3` is
    `None` (all-null or zero-row column). Otherwise counts values outside
    `[q1 - multiplier_iqr*(q3-q1), q3 + multiplier_iqr*(q3-q1)]` as IQR
    outliers. `std` of `None` (all-null or single-value column) is treated
    as `0`, which short-circuits the Z-score outlier count to `0` instead of
    dividing by zero. Chart bounds are only populated when `criterion`
    matches; otherwise they are `(None, None)`.

    Args:
        data (Series): Column values to test for outliers; must be numeric
            (or castable to numeric, e.g. string length for text columns)
            to support the arithmetic comparisons below.
        mean (float): Mean of `data`, used for Z-score outlier counting and
            Z-score bounds.
        std (float | None): Standard deviation of `data`. `None` is
            normalized to `0` internally.
        q1 (float | None): First quartile (0.25) of `data`. `None` short-circuits
            to `(0, 0, (None, None))`.
        q3 (float | None): Third quartile (0.75) of `data`. `None` short-circuits
            to `(0, 0, (None, None))`.
        config (dict[str, str | float]): Parameters for detecting outliers:
            - 'criterion' (str): "IQR" or "Z-score"; selects which bounds are
              returned. Any other value (or key absent) yields `(None, None)`.
            - 'multiplier_iqr' (float): multiplier for IQR criterion (defaults to 1.5).
            - 'threshold_z_score' (float): threshold for Z-score criterion (defaults to 3.0).

    Returns:
        tuple[int, int, tuple[float | None, float | None]]:
            - Number of IQR outliers.
            - Number of Z-score outliers (`0` if `std` is `0` or `None`).
            - `(lower, upper)` bounds for chart highlighting: IQR bounds if
              `criterion` is "IQR", `mean ± threshold_z_score*std` if
              `criterion` is "Z-score", else `(None, None)`.
    """
    bounds = (None, None)

    # No data to compute IQR in case of all-null column
    if q1 is None or q3 is None:
        return 0, 0, (None, None)

    # Determine boundaries for outliers based on IQR
    lower_bound = q1 - config.get("multiplier_iqr", 1.5) * (q3 - q1)
    upper_bound = q3 + config.get("multiplier_iqr", 1.5) * (q3 - q1)
    # Count the number of outliers
    outliers_iqr = ((data < lower_bound) | (data > upper_bound)).sum()

    # Standard deviation value is None in case of all-null or single-value column;
    # Equate it to zero
    if std is None:
        std = 0
    # Count the number of outliers based on Z-score
    if std == 0:
        outliers_zscore = 0
    else:
        outliers_zscore = (
            ((data - mean) / std).abs() > config.get("threshold_z_score", 3.0)
        ).sum()

    # Get boundaries to highlight outliers on a chart
    # if criterion was specified in configuration
    if config.get("criterion") == "IQR":
        bounds = (lower_bound, upper_bound)
    if config.get("criterion") == "Z-score":
        bounds = (
            mean - config.get("threshold_z_score", 3.0) * std,
            mean + config.get("threshold_z_score", 3.0) * std,
        )

    return outliers_iqr, outliers_zscore, bounds
