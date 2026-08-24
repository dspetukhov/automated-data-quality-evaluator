"""Builds per-column Plotly subplot charts with outlier regions highlighted, and saves them to disk."""

from typing import Any
from polars import DataFrame, Series
from plotly.graph_objs import Scatter
from plotly.graph_objs._figure import Figure
from plotly.subplots import make_subplots
from utility import exception_handler, TIME_INTERVAL_COL


@exception_handler()
def make_charts(
    data: DataFrame,
    bounds: list[tuple[float | None, float | None]],
    config: dict[str, Any],
    file_path: str
) -> None:
    """Make charts from input data using Plotly.

    Every column of data except the first (assumed to be the time-interval
    column, used as the x-axis for every subplot) is plotted in its own
    subplot as a line trace, with outlier regions highlighted per
    highlight_outliers. The finished figure is written to file_path.

    Args:
        data (DataFrame): Data to plot. First column supplies x-axis values
            (expected to be TIME_INTERVAL_COL); remaining columns are
            plotted one per subplot, in order.
        bounds (list[tuple[float | None, float | None]]): Outlier
            highlighting boundaries, aligned 1:1 by index with
            data.columns[1:] (as returned by evaluate.evaluate_data).
        config (dict[str, Any]): Plotly styling settings, including:
            - "subplots" (dict): passed to create_figure.
            - "plot" (dict): Scatter trace style kwargs.
            - "outliers" (dict): must contain "style" (dict) with
              add_shape kwargs for outlier regions, passed to
              highlight_outliers.
            - "format" (str): image format for write_image (default "png").
            - "scale_factor" (float): image scale for write_image (default 1).
            - "layout", "grid", "annotations", "tickformat": consumed by
              adjust_figure.
        file_path (str): Full path (including filename, without requiring
            a matching extension) where the image will be saved.

    Returns:
        None: Figure is saved to disk at file_path; nothing is returned.
    """
    # Determine the number of subplots
    # with at least 2 subplots due to possible absence of "Target average"
    n_subplots = max(2, data.shape[1] - 1)
    # Create a figure with subplots
    fig, n_cols, n_rows = create_figure(
        n_subplots,
        config.get("subplots", {}),
        titles=data.columns[1:]
    )
    # Make a chart for each column in data, skip first time interval column
    for i, col in enumerate(data.columns[1:]):
        # Add data series as a trace to the subplot
        fig.add_trace(
            Scatter(
                x=data[TIME_INTERVAL_COL].to_list(),
                y=data[col].to_list(),
                **config.get("plot", {})
            ),
            row=(i // n_cols) + 1, col=(i % n_cols) + 1
        )
        # Highlight outliers regions using Plotly shapes
        fig = highlight_outliers(
            fig, i, data[TIME_INTERVAL_COL], data[col], bounds[i], n_cols,
            config.get("outliers", {}).get("style", {})
        )

    # Adjust figure parameters
    fig = adjust_figure(fig, n_cols, n_rows, config)

    # Save figure
    fig.write_image(
        file_path,
        format=config.get("format", "png"),
        scale=config.get("scale_factor", 1)
    )


def create_figure(
    n_subplots: int,
    config: dict[str, Any],
    titles: list[str]
) -> tuple[Figure, int, int]:
    """Creates Plotly figure with required number of subplots.

    This function creates figure using `plotly.subplots.make_subplots`.
    The grid always has 2 columns; the number of rows is derived from
    n_subplots via ceiling division. Each subplot title is derived from
    the matching entry in titles by keeping only the part after the last
    " __" delimiter.

    Args:
        n_subplots (int): Total number of subplots in the subplot grid.
        config (dict[str, Any]): Plotly styling settings for subplots:
            "horizontal_spacing" (float, default 0.1) and
            "vertical_spacing" (float, default 0.1), passed to make_subplots.
        titles (list[str]): Raw titles for each subplot, one per column
            (e.g. data.columns[1:] in make_charts).

    Returns:
        tuple[Figure, int, int]: Plotly figure object,
            number of columns in subplot grid (always 2),
            number of rows in subplot grid.
    """
    n_cols = 2  # number of columns in the subplot grid is always equal to 2
    # Determine the number of rows required for the given number of subplots
    n_rows = (n_subplots + n_cols - 1) // n_cols
    # Create a figure with subplots
    fig = make_subplots(
        rows=n_rows, cols=n_cols,
        horizontal_spacing=config.get("horizontal_spacing", 0.1),
        vertical_spacing=config.get("vertical_spacing", 0.1),
        subplot_titles=[item.split(" __")[-1] for item in titles]
    )
    return fig, n_cols, n_rows


def highlight_outliers(
    fig: Figure,
    s: int,
    x: Series,
    data: Series,
    bounds: tuple[float | None, float | None],
    n_cols: int,
    config: dict[str, Any]
) -> Figure:
    """Highlight outliers using Plotly shapes.

    If both bounds are set (i.e. an outlier criterion was configured in
    evaluate.evaluate_data_outliers), adds two rectangle shapes to the
    subplot at index s: one spanning from data.min() up to the lower
    bound, and one from the upper bound up to data.max(), each spanning
    the full x-range (x.min() to x.max()). Nulls in x or data (e.g. a
    null-keyed time-interval group) are skipped via Series.min()/max().
    If either bound is None, fig is returned unchanged.

    Args:
        fig (Figure): Plotly figure object.
        s (int): Subplot index (0-based, in row-major order over the grid).
        x (Series): x-axis data (time-interval values) for the subplot;
            may contain nulls.
        data (Series): y-axis data for the subplot; only its min/max are
            used, to size the highlighted regions.
        bounds (tuple[float | None, float | None]): Lower and upper
            outlier boundaries; no shapes are added if either is None.
        n_cols (int): Number of columns in the subplot grid, used to map
            s to a (row, col) position.
        config (dict[str, Any]): Plotly styling kwargs forwarded to
            `Figure.add_shape` for each region (e.g. fillcolor, opacity,
            line, layer).

    Returns:
        Figure: The same Figure instance, with shapes added if bounds
            were both set (fig is also mutated in place).
    """
    # If lower and upper boundaries are not None
    if None not in bounds:
        lower_bound, upper_bound = bounds
        shape = (
            (data.min(), lower_bound),
            (upper_bound, data.max())
        )
        for i in range(len(shape)):
            fig.add_shape(
                x0=x.min(), x1=x.max(), y0=shape[i][0], y1=shape[i][1],
                **config,
                row=(s // n_cols) + 1, col=(s % n_cols) + 1)
    return fig


def adjust_figure(
    fig: Figure,
    n_cols: int,
    n_rows: int,
    config: dict[str, Any]
) -> Figure:
    """Adjust Plotly figure parameters.

    This function sets figure width/height (per-subplot pixel size from
    config, multiplied by n_cols/n_rows) and template, formats x-axis
    tick labels, applies grid styling to both axes, and shifts each
    subplot title annotation's x/y position by a configurable offset.

    Args:
        fig (Figure): Plotly figure object.
        n_cols (int): Number of columns in the subplot grid.
        n_rows (int): Number of rows in the subplot grid.
        config (dict[str, Any]): Plotly figure configuration parameters:
            - "layout" (dict): forwarded to `Figure.update_layout`; may
              set "template" (default "plotly_white"), "width" (px per
              column, default 700), "height" (px per row, default 240).
            - "grid" (dict): forwarded to both `update_xaxes` and
              `update_yaxes`.
            - "tickformat" (str): x-axis tick format (default "%Y-%m-%d").
            - "annotations" (dict): "x_offset" and "y_offset" (both
              default 0) are popped and used to shift each subplot title
              annotation's position; remaining keys are forwarded to
              `annotation.update` for every subplot title.

    Returns:
        Figure: The same Figure instance, updated in place.
    """
    layout = config.get("layout", {}).copy()
    grid_config = config.get("grid", {})
    annotations_config = config.get("annotations", {})

    # Adjust figure layout
    layout.update({
        "template": layout.get("template", "plotly_white"),
        "width": layout.get("width", 700) * n_cols,
        "height": layout.get("height", 240) * n_rows
    })
    fig.update_layout(layout)

    # Alter x-axis tick format, add grid
    fig.update_xaxes(
        tickformat=config.get("tickformat", "%Y-%m-%d"),
        **grid_config
    )
    fig.update_yaxes(**grid_config)

    # Adjust subplot titles
    x_offset = annotations_config.pop("x_offset", 0)
    y_offset = annotations_config.pop("y_offset", 0)
    for annotation in fig.layout.annotations:
        annotation.update(
            x=annotation.x + (1 / n_cols) / (n_cols + x_offset),
            y=annotation.y + y_offset,
            **annotations_config
        )
    return fig
