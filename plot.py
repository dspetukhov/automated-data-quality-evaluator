"""Builds Plotly charts with outlier regions highlighted, and saves them to disk."""

from typing import Any

from plotly.graph_objs import Scatter
from plotly.graph_objs._figure import Figure
from plotly.subplots import make_subplots
from polars import DataFrame, Series

from utility import PREFIX_COL, TIME_INTERVAL_COL, exception_handler


@exception_handler()
def make_charts(
    data: DataFrame,
    bounds: list[tuple[float | None, float | None]],
    config: dict[str, Any],
    file_path: str,
) -> None:
    """Make charts from input data using Plotly.

    The first column in data is the time-interval column that is used
    as the x-axis for every subplot. Every other column is plotted
    in its own subplot as a line trace, with outlier regions highlighted.
    The chart is written to `file_path`.

    Args:
        data (DataFrame): Data to plot.
        bounds (list[tuple[float | None, float | None]]): Boundaries
            to highlight outliers as produced by `evaluate.evaluate_data`
            (aligned 1:1 by index with `data.columns[1:]`).
        config (dict[str, Any]): Plotly styling settings:
            - `"subplots"` (dict): Subplot kwargs passed to `make_subplots`.
            - `"plot"` (dict): Trace kwargs passed to `Scatter`.
            - `"outliers"` (dict): Style for outlier regions passed to `highlight_outliers`.
            - `"layout"`, `"grid"`, `"annotations"`, `"tickformat"`: Extra settings passed to `adjust_chart`.
            - `"format"` (str): Image format for `write_image` (default `"png"`).
            - `"scale_factor"` (float): Image scale for `write_image` (default `1`).
        file_path (str): Full path (including filename) where the image will be saved.

    Returns:
        `None`: Charts are saved to disk at `file_path`.
    """
    # Determine the number of subplots
    # with at least 2 subplots due to possible absence of "Target average"
    n_subplots = max(2, data.shape[1] - 1)
    # Create chart with subplots
    chart, n_cols, n_rows = create_chart(
        n_subplots, config.get("subplots", {}), titles=data.columns[1:]
    )
    # Plot each data column as a trace, skip the time interval column
    for i, col in enumerate(data.columns[1:]):
        chart.add_trace(
            Scatter(
                x=data[TIME_INTERVAL_COL].to_list(),
                y=data[col].to_list(),
                **config.get("plot", {}),
            ),
            row=(i // n_cols) + 1,
            col=(i % n_cols) + 1,
        )
        # Highlight outlier regions using Plotly shapes
        chart = highlight_outliers(
            chart,
            i,
            data[TIME_INTERVAL_COL],
            data[col],
            bounds[i],
            n_cols,
            config.get("outliers", {}).get("style", {}),
        )

    # Adjust chart style
    chart = adjust_chart(chart, n_cols, n_rows, config)

    # Save chart to disk
    chart.write_image(
        file_path,
        format=config.get("format", "png"),
        scale=config.get("scale_factor", 1),
    )


def create_chart(
    n_subplots: int, config: dict[str, Any], titles: list[str]
) -> tuple[Figure, int, int]:
    """Create Plotly chart with required number of subplots.

    This function creates a chart with `plotly.subplots.make_subplots`.
    The grid always has 2 columns; the number of rows is derived
    based on `n_subplots`. Each subplot title is derived from
    the matching entry in `titles` by keeping only the part
    after the last `f" {PREFIX_COL}"` delimiter.

    Args:
        n_subplots (int): Total number of subplots in the subplot grid.
        config (dict[str, Any]): Plotly styling settings for subplots:
            `"horizontal_spacing"` (float, default `0.1`) and
            `"vertical_spacing"` (float, default `0.1`), passed to `make_subplots`.
        titles (list[str]): Raw titles for each subplot.

    Returns:
        tuple[Figure, int, int]:
            - `plotly.graph_objects.Figure` instance.
            - Number of columns in the subplot grid (always `2`).
            - Number of rows in the subplot grid.
    """
    n_cols = 2  # number of columns in the subplot grid is always equal to 2
    # Determine the number of rows required for the given number of subplots
    n_rows = (n_subplots + n_cols - 1) // n_cols
    # Create a chart with subplots
    chart = make_subplots(
        rows=n_rows,
        cols=n_cols,
        horizontal_spacing=config.get("horizontal_spacing", 0.1),
        vertical_spacing=config.get("vertical_spacing", 0.1),
        subplot_titles=[item.split(f" {PREFIX_COL}")[-1] for item in titles],
    )
    return chart, n_cols, n_rows


def highlight_outliers(
    chart: Figure,
    s: int,
    x: Series,
    data: Series,
    bounds: tuple[float | None, float | None],
    n_cols: int,
    config: dict[str, Any],
) -> Figure:
    """Highlight outliers using Plotly shapes.

    If both `bounds` are not `None`, adds two rectangle shapes to the subplot
    at index `s`: one spanning from `data.min()` to the lower bound, and one
    from the upper bound to `data.max()`, each spanning the full x-range
    (`x.min()` to `x.max()`). Nulls in `x` or `data` are skipped via
    `Series.min()`/`Series.max()`. If either bound is `None`, the chart is
    returned unchanged.

    Args:
        chart (Figure): `plotly.graph_objects.Figure` instance.
        s (int): Subplot index.
        x (Series): x-axis data for the subplot.
        data (Series): y-axis data to size the highlighted regions in the subplot.
        bounds (tuple[float | None, float | None]): Lower and upper boundaries for y-axis.
        n_cols (int): Number of columns in the subplot grid, used to map
            `s` to a `(row, col)` position.
        config (dict[str, Any]): Plotly styling kwargs for `Figure.add_shape`.

    Returns:
        `Figure`: The same `Figure` instance, with shapes added in place
            if `bounds` were both set.
    """
    # If lower and upper boundaries are not None
    if None not in bounds:
        lower_bound, upper_bound = bounds
        shape = ((data.min(), lower_bound), (upper_bound, data.max()))
        for i in range(len(shape)):
            chart.add_shape(
                x0=x.min(),
                x1=x.max(),
                y0=shape[i][0],
                y1=shape[i][1],
                **config,
                row=(s // n_cols) + 1,
                col=(s % n_cols) + 1,
            )
    return chart


def adjust_chart(
    chart: Figure, n_cols: int, n_rows: int, config: dict[str, Any]
) -> Figure:
    """Adjust Plotly chart parameters.

    This function sets template, chart width/height based on subplot pixel size
    from config, formats x-axis tick labels, applies grid styling to both axes,
    and changes x/y position for each subplot title.

    Args:
        chart (Figure): Plotly figure object.
        n_cols (int): Number of columns in the subplot grid.
        n_rows (int): Number of rows in the subplot grid.
        config (dict[str, Any]): Plotly Figure configuration parameters:
            - `"layout"` (dict): May set `"template"` (default `"plotly_white"`),
                `"width"` (px per column, default `700`), `"height"` (px per row, default `240`).
            - `"grid"` (dict): Grid styling for x/y axes.
            - `"tickformat"` (str): X-axis tick format (default `"%Y-%m-%d"`).
            - `"annotations"` (dict): Parameters used to adjust subplot title position.

    Returns:
        `Figure`: The same `Figure` instance, updated in place.
    """
    layout = config.get("layout", {}).copy()
    grid_config = config.get("grid", {})
    annotations_config = config.get("annotations", {}).copy()

    # Adjust layout
    layout.update(
        {
            "template": layout.get("template", "plotly_white"),
            "width": layout.get("width", 700) * n_cols,
            "height": layout.get("height", 240) * n_rows,
        }
    )
    chart.update_layout(layout)

    # Adjust x-axis tick format, add grid
    chart.update_xaxes(tickformat=config.get("tickformat", "%Y-%m-%d"), **grid_config)
    chart.update_yaxes(**grid_config)

    # Adjust subplot title position
    x_offset = annotations_config.pop("x_offset", 0)
    y_offset = annotations_config.pop("y_offset", 0)
    for annotation in chart.layout.annotations:
        annotation.update(
            x=annotation.x + (1 / n_cols) / (n_cols + x_offset),
            y=annotation.y + y_offset,
            **annotations_config,
        )
    return chart
