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
    """
    Make charts from input data using Plotly.

    Each column from input data is plotted in a separate subplot.
    The function supports custom plot styles, subplot arrangement,
    and anomaly highlighting (using IQR or Z-score) as specified
    in the configuration.

    Args:
        data (DataFrame): Data to plot.
        bounds (list[tuple[float | None, float | None]]): List of boundaries
            to highlight outliers.
        config (dict[str, Any]): Plotly styling settings, including:
            - 'plot': dict of Scatter style settings.
            - 'outliers': dict of outliers highlighting settings.
            - 'layout', 'grid', and other Plotly configuration parameters.
        file_path (str): Full path (including filename) where the image will be saved.

    Returns:
        None: Figure is saved to disk.
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
                x=data[TIME_INTERVAL_COL],
                y=data[col],
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
    """
    Creates Plotly figure with required number of subplots.

    This function creates figure using `plotly.subplots.make_subplots`.
    The size of the subplots grid depends on the total number of subplots.
    Each subplot has a title specified in `titles`.

    Args:
        n_subplots (int): Total number of subplots in the subplot grid.
        config (dict[str, Any]): Plotly styling settings for subplots.
        titles (list[str]): Raw titles for each subplot.

    Returns:
        tuple[Figure, int, int]: Plotly figure object,
            number of columns in subplot grid, number of rows in subplot grid.
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
    """
    Highlight outliers using Plotly shapes.

    This function add shapes to highlight outliers on plots
    if boundaries were determined by `evaluate_data_outliers` function.

    Args:
        fig (Figure): Plotly figure object.
        s (int): Subplot index.
        x (Series): x-axis data.
        data (Series): y-axis data.
        bounds (tuple[float | None, float | None]): Tuple with lower and upper boundaries.
        n_cols (int): Number of columns in the subplot grid.
        config (dict[str, Any]): Plotly styling settings for outliers.

    Returns:
        Figure: Plotly figure with added shapes.
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
                x0=min(x), x1=max(x), y0=shape[i][0], y1=shape[i][1],
                **config,
                row=(s // n_cols) + 1, col=(s % n_cols) + 1)
    return fig


def adjust_figure(
    fig: Figure,
    n_cols: int,
    n_rows: int,
    config: dict[str, Any]
) -> Figure:
    """
    Adjust Plotly figure parameters.

    This function changes figure width and height based on
    the number of rows and cols and settings specified in the configuration.
    It formats tick labels for x-axis, add grid if specified in the configuration,
    align subplot titles to the left side of plots.

    Args:
        fig (Figure): Plotly figure object.
        n_cols (int): Number of columns in the subplot grid.
        n_rows (int): Number of rows in the subplot grid.
        config (dict[str, Any]): Plotly figure configation parameters.

    Returns:
        Figure: Adjusted Plotly figure.
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
