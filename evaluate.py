from typing import Any

from polars import DataFrame, Series

from utility import exception_handler


@exception_handler()
def evaluate_data(
    data: DataFrame, config: dict[str, str | float]
) -> tuple[list[dict[str, Any]], list[tuple[float | None, float | None]]]:
    """Evaluates descriptive statistic and detects outliers in data.

    This function calculates descriptive statistics and
    detects outliers based on IQR and Z-score criteria for each column in data.

    Args:
        data (DataFrame): Input data.
        config (dict[str, str | float]): Parameters for detecting outliers:
            - 'criterion' (str): IQR or Z-score,
            - 'multiplier_iqr' (float): multiplier for IQR criterion (defaults to 1.5).
            - 'threshold_z_score' (float): threshold for Z-score criterion (defaults to 3.0).

    Returns:
        tuple[list[dict[str, Any]], list[tuple[float | None, float | None]]]:
            - List of dictionaries with description of each column in data:
                - name of the column,
                - mean and standard deviation,
                - range of values,
                - Q1, Q3, and IQR,
                - outliers percentage according to IQR and Z-score criteria.
            - List of boundaries for outliers to be highlighted on a chart.
    """
    data_evals, outliers_bounds = [], []

    # Evaluate each column in data, skip first time interval column
    for col in data.columns[1:]:
        # Calculate mean and standard deviation, first and third quartile
        mean, std = data[col].mean(), data[col].std()
        q1, q3 = data[col].quantile(0.25), data[col].quantile(0.75)

        outliers_iqr, outliers_zscore, bounds = evaluate_data_outliers(
            data[col], mean, std, q1, q3, config
        )
        # Use non-null count so percentages aren't diluted by nulls, and guard
        # against division by zero when there is no non-null data at all
        non_null_count = data[col].len() - data[col].null_count()
        outliers_iqr_pct = 100 * outliers_iqr / non_null_count if non_null_count else 0.0
        outliers_zscore_pct = 100 * outliers_zscore / non_null_count if non_null_count else 0.0

        # No non-null data to derive a range or IQR from (e.g. all-null column)
        min_val, max_val = data[col].min(), data[col].max()
        range_val = max_val - min_val if min_val is not None and max_val is not None else None
        iqr_val = q3 - q1 if q1 is not None and q3 is not None else None

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
    q1: float,
    q3: float,
    config: dict[str, str | float],
) -> tuple[int, int, tuple[float | None, float | None]]:
    """Evaluates outliers in data.

    This function calculates the number of outliers
    according to IQR and Z-score criteria, determines boundaries
    to highlight outliers on a chart if criterion was specified in configuration.

    Args:
        data (Series): Input data.
        mean (float): Average of data.
        std (float): Standard deviation.
        q1 (float): First quartile.
        q3 (float): Third quartile.
        config (dict[str, str | float]): Parameters for detecting outliers:
            - 'criterion' (str): IQR or Z-score,
            - 'multiplier_iqr' (float): multiplier for IQR criterion (defaults to 1.5).
            - 'threshold_z_score' (float): threshold for Z-score criterion (defaults to 3.0).

    Returns:
        tuple[int, int, tuple[float | None, float | None]]:
            - number of outliers based on IQR and Z-score,
            - boundaries to highlight outliers on a chart.
    """
    # A missing standard deviation (e.g. all-null or single-value column) has
    # no meaningful spread, so treat it the same as zero variance
    if std is None:
        std = 0

    # Count the number of outliers based on Z-score
    if std == 0:
        outliers_zscore = 0
    else:
        outliers_zscore = (
            ((data - mean) / std).abs() > config.get("threshold_z_score", 3.0)
        ).sum()

    # No data to compute meaningful quartile-based bounds (e.g. all-null column)
    if q1 is None or q3 is None:
        return 0, 0, (None, None)

    # Determine boundaries for outliers based on IQR
    lower_bound = q1 - config.get("multiplier_iqr", 1.5) * (q3 - q1)
    upper_bound = q3 + config.get("multiplier_iqr", 1.5) * (q3 - q1)
    # Count the number of outliers
    outliers_iqr = ((data < lower_bound) | (data > upper_bound)).sum()

    # Get boundaries to highlight outliers on a chart
    # if criterion was specified in configuration
    if config.get("criterion") == "Z-score":
        bounds = (
            mean - config.get("threshold_z_score", 3.0) * std,
            mean + config.get("threshold_z_score", 3.0) * std,
        )
    elif config.get("criterion") == "IQR":
        bounds = (lower_bound, upper_bound)
    else:
        bounds = (None, None)

    return outliers_iqr, outliers_zscore, bounds
