"""Calculates per-column descriptive statistics and detects outliers (IQR/Z-score)."""

from typing import Any

from polars import DataFrame, Series

from utility import PREFIX_COL, exception_handler


@exception_handler()
def evaluate_data(
    data: DataFrame, config: dict[str, str | float]
) -> tuple[list[dict[str, Any]], list[tuple[float | None, float | None]]]:
    """Calculates descriptive statistics and outlier counts for each column in data.

    Skips the first column of data (assumed to be the time-interval column).
    For every other column, computes mean, standard deviation, min, max,
    quartiles, range, IQR, and outlier percentages with evaluate_data_outliers.
    Outlier percentages are relative to each column's non-null count, and
    edge cases (all-null or empty column) resolve to None/0.0 values.

    Args:
        data (DataFrame): Input data as produced by preprocess.make_preprocessing.
        config (dict[str, str | float]): Parameters for detecting outliers,
            passed to evaluate_data_outliers:
            - "criterion" (str): IQR or Z-score,
            - "multiplier_iqr" (float): multiplier for IQR criterion (defaults to 1.5).
            - "threshold_z_score" (float): threshold for Z-score criterion (defaults to 3.0).

    Returns:
        tuple[list[dict[str, Any]], list[tuple[float | None, float | None]]]:
            - List of dictionaries with description of each column in data:
                - name of the column,
                - mean and standard deviation,
                - range of values (None if the column is all-null or empty),
                - Q1, Q3, and IQR (None if the column is all-null or empty),
                - outliers percentage of non-null values according to IQR
                  and Z-score criteria (0.0 if the column is all-null or empty).
            - List of boundaries for outliers to be highlighted on a chart,
              aligned by index with the dictionaries above.
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
                "title": col.split(f" {PREFIX_COL}")[-1],
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
    """Counts IQR and Z-score outliers in a column and derives chart boundaries.

    Returns (0, 0, (None, None)) if q1 or q3 is None (all-null or empty column).
    std of None (all-null or single-value column) is treated as 0, which equates
    Z-score outlier count to 0. Chart bounds are only populated when criterion matches;
    otherwise they are (None, None).

    Args:
        data (Series): polars.Series of numeric values.
        mean (float): Mean of data to count Z-score outliers and boundaries.
        std (float | None): Standard deviation of data.
        q1 (float | None): First quartile (0.25) of data.
        q3 (float | None): Third quartile (0.75) of data.
        config (dict[str, str | float]): Parameters for detecting outliers:
            - "criterion" (str): "IQR" or "Z-score"; selects which bounds are
              returned. Any other value yields (None, None).
            - "multiplier_iqr" (float): multiplier for IQR criterion (defaults to 1.5).
            - "threshold_z_score" (float): threshold for Z-score criterion (defaults to 3.0).

    Returns:
        tuple[int, int, tuple[float | None, float | None]]:
            - Number of IQR outliers.
            - Number of Z-score outliers (0 if std is 0 or None).
            - (lower, upper) bounds for chart highlighting: IQR bounds if
              criterion is "IQR", mean ± threshold_z_score*std if
              criterion is "Z-score", else (None, None).
    """
    bounds = (None, None)
    criterion = config.get("criterion")
    multiplier_iqr = config.get("multiplier_iqr", 1.5)
    threshold_z_score = config.get("threshold_z_score", 3.0)

    # No data to compute IQR in case of all-null column
    if q1 is None or q3 is None:
        return 0, 0, (None, None)

    # Determine boundaries for outliers based on IQR
    lower_bound = q1 - multiplier_iqr * (q3 - q1)
    upper_bound = q3 + multiplier_iqr * (q3 - q1)
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
        outliers_zscore = ((data - mean) / std).abs() > threshold_z_score
        outliers_zscore = outliers_zscore.sum()

    # Get boundaries to highlight outliers on a chart
    # if criterion was specified in configuration
    if criterion == "IQR":
        bounds = (lower_bound, upper_bound)
    if criterion == "Z-score":
        bounds = (
            mean - threshold_z_score * std,
            mean + threshold_z_score * std,
        )

    return outliers_iqr, outliers_zscore, bounds
