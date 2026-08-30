"""Assembles the markdown report with embedded charts and tables and writes it to disk."""

import time
from pathlib import Path
from typing import Any

from polars import DataFrame
from tabulate import tabulate

from evaluate import evaluate_data
from plot import make_charts
from utility import (
    OVERVIEW_COL,
    PREFIX_COL,
    PREFIX_COL_E,
    TIME_INTERVAL_COL,
    exception_handler,
)


@exception_handler()
def make_report(
    df: DataFrame, metadata: dict[str, str | None], config: dict[str, Any]
) -> None:
    """Create markdown report with charts and tables.

    This function produces a markdown report with charts and tables
    by processing input data frame and metadata according to the
    parameters specified in the configuration file.

    Args:
        df (DataFrame): Aggregated per-time-interval data, expected to
        contain a `TIME_INTERVAL_COL` column plus per-column stat columns
        prefixed with `PREFIX_COL`/`PREFIX_COL_E`.
        metadata (dict[str, str | None]): Dict of aggregated columns
            indicating types for numeric columns.
        config (dict[str, Any]): Configuration dictionary specifying
            data source name, markdown options, and plotting options.

    Returns:
        None: Writes the report to disk on success.
    """
    # Get key variables to make the report
    output, source, content, precision, outliers, plotly = get_report_variables(config)

    data_evals = {}
    # Get evaluations and create overview chart for columns
    # representing general aggregations of source data:
    # number of values and target average
    data = df.select(
        [TIME_INTERVAL_COL]
        + [item for item in df.columns if item.startswith(f" {PREFIX_COL}")]
    )
    col = OVERVIEW_COL
    # Evaluate data
    evals, bounds = evaluate_data(data, outliers)
    data_evals[col] = {"evals": evals}
    # Make chart
    make_charts(data, bounds=bounds, config=plotly, file_path=Path(output, col))

    # Get evaluations and create charts for columns
    # representing aggregations for a column in source data:
    # number of unique values and proportion of missing values
    for col in metadata:
        # Replace whitespaces in column name with a hyphen
        # to ensure proper reference to charts in Markdown
        col_ = col.replace(" ", "-")

        data = df.select(
            [TIME_INTERVAL_COL]
            + [
                item
                for item in df.columns
                if item.startswith(f"{PREFIX_COL} {col} {PREFIX_COL}")
            ]
        )
        evals, bounds = evaluate_data(data, outliers)
        data_evals[col] = {"evals": evals}
        make_charts(data, bounds=bounds, config=plotly, file_path=Path(output, col_))

        # Get evaluations and create charts for columns
        # representing extra aggregations for a numeric column in source data:
        # minimum, maximum, mean, median, and standard deviation
        if metadata.get(col):
            data = df.select(
                [TIME_INTERVAL_COL]
                + [
                    item
                    for item in df.columns
                    if item.startswith(f"{PREFIX_COL_E} {col} {PREFIX_COL}")
                ]
            )
            evals, bounds = evaluate_data(data, outliers)
            data_evals[col].update({"evals_numeric": evals, "dtype": metadata[col]})
            make_charts(
                data,
                bounds=bounds,
                config=plotly,
                file_path=Path(output, f"{col_}__extra"),
            )

    # Collect markdown content
    content = collect_md_content(data_evals, content, output, source, precision)

    # Write content to file
    write_md_file(content, output, config.get("markdown", {}).get("name"))


def get_report_variables(
    config: dict[str, Any],
) -> tuple[str, str, list[str], int | None, dict, dict]:
    """Get key variables to make the report using the configuration provided.

    This function pulls variables from the configuration for markdown report
    generation. They include: output directory to store report data, path to
    the data source, style for markdown tables, precision to format floats in
    markdown tables, outliers detection parameters, Plotly parameters for charts.

    Creates the output directory for storing markdown report file and charts
    if it does not exist.

    Args:
        config (dict[str, Any]): Configuration dictionary. Reads
            "output", "source.file_path"/"source.query",
            "markdown.css_style", "markdown.float_precision",
            "outliers", and "plotly".

    Returns:
        tuple[str, str, list[str], int, dict, dict]:
            - Directory name to store report file and charts.
            - Formatted path to the file to read or SQL query to get data.
            - Content of markdown report: a one-element list with a CSS
              `<link>` tag if "markdown.css_style" points to an existing
              file, resolved relative to the output directory; otherwise
              an empty list.
            - Precision to format floats in markdown tables; defaults to
              4 decimal places when "markdown.float_precision" is unset.
            - Outliers detection parameters.
            - Plotly configuration for charts.
    """
    # Determine the name of the output directory using `output` parameter
    # in configuration or based on the source specification
    output_dir = config.get(
        "output",
        Path(config["source"]["file_path"]).name.split(".")[0]
        if config["source"].get("file_path")
        else "postgresql",
    )
    # Create output directory
    Path(output_dir).mkdir(exist_ok=True)

    # Determine and format source of data for markdown report
    if config["source"].get("file_path"):
        # Replace "*" to ensure correct representation in Markdown
        source = "**{}**".format(config["source"]["file_path"].replace("*", r"\*"))
    else:
        source = "\n```sql\n{}\n```\n".format(config["source"]["query"])

    # Initialize markdown content list
    md_content = []

    # CSS style for markdown tables
    css_style = config.get("markdown", {}).get("css_style")
    if css_style:
        css_style_file_path = Path(css_style)
        if css_style_file_path.exists() and css_style_file_path.is_file():
            file_path = css_style_file_path.resolve().relative_to(
                Path(output_dir).resolve(), walk_up=True
            )
            md_content = [f"<link rel='stylesheet' href='{file_path}'>\n"]

    # Number of decimal places to format numbers in markdown tables
    precision = config.get("markdown", {}).get("float_precision", 4)

    # Outliers detection parameters
    outliers_config = config.get("outliers", {})

    # Plotly configuration
    plotly_config = config.get("plotly", {})

    return (output_dir, source, md_content, precision, outliers_config, plotly_config)


def collect_md_content(
    data: dict[str, Any],
    content: list[str],
    output: str,
    source: str,
    precision: int,
) -> list[str]:
    """Collects markdown content with table-of-contents from input data.

    Args:
        data (dict[str, Any]): Raw data to construct markdown content with
            sections, tables, and reference links.
        content (list[str]): Content initialized by `get_report_variables`.
        output (str): Directory name to store report file, used
            only in the report header.
        source (str): Path to the file to read or SQL query to fetch data.
        precision (int): Number of decimal places to format `float` values.

    Returns:
        list[str]: List of strings to be written in file.
    """
    toc = []
    for col in data:
        # Replace whitespaces in column name with a hyphen
        # to ensure proper reference links in Markdown
        col_ = col.replace(" ", "-")
        # Get section title (`alias`)
        alias = "Overview" if col == OVERVIEW_COL else f"`{col}`"

        # Add new section to the table-of-contents with anchor
        toc.append(
            f"- [{alias}](#{'overview' if col == OVERVIEW_COL else col_.lower()})"
        )

        # Add new entry to the content: section with anchor, chart, and table
        content.append(
            ("## {alias}\n\n![{col}]({col})\n\n{table}").format(
                col=col_,
                alias=alias,
                table=make_md_table(data[col]["evals"], precision),
            )
        )
        # Add extra section for numeric columns
        if data[col].get("dtype"):
            content.append(
                ("### `{alias}`\n\n![{col}]({col}__extra)\n\n{table}").format(
                    col=col_,
                    alias=data[col]["dtype"],
                    table=make_md_table(data[col]["evals_numeric"], precision),
                )
            )
        # Add backlink to the table-of-contents at the end of each section
        content.append("[Back to table of contents](#table-of-contents)\n")

    timestamp = time.strftime("%Y-%m-%d %H:%M", time.localtime())
    toc = "\n".join(toc)

    md_output = [
        f"# {output} | {timestamp}\n\n",
        f"Data source: {source}\n\n",
        f"## Table of contents\n\n{toc}\n\n",
        "\n".join(content),
    ]
    return md_output


def make_md_table(data: list[dict], precision: int) -> str:
    """Create a markdown table from input data.

    This function converts a list of dictionaries into a markdown table
    using the tabulate library. Dict values for a specific key form
    a table row. The table is returned as a string to be included
    as a part of the markdown report.

    Args:
        data (list[dict]): List of dictionaries with calculated statistics.
        precision (int): Number of decimal places to format `float`
            passed through to `format_number`.

    Returns:
        str: Markdown table, or a single newline character if `data` is empty.
    """
    # Ensure the minimum number of columns is 2
    data = list(data)
    while len(data) < 2:
        data.append({})

    # Compose formatted table content
    rows = []
    for key in data[0]:
        col_index = [" " if key == "title" else f"**{key}**"]
        col_values = [format_number(item.get(key), precision) for item in data]
        rows.append(col_index + col_values)

    if not rows:
        return "\n"

    # Create markdown table
    return (
        tabulate(
            rows,
            headers="firstrow",
            tablefmt="pipe",
            colalign=["left"] + ["center"] * (len(rows[0]) - 1),
        )
        + "\n"
    )


@exception_handler(exit_on_error=True)
def write_md_file(content: list[str], output: str, file_name: str | None) -> None:
    """Write markdown file to disk.

    This function writes a markdown file to disk, adding `.md` extension if absent.
    The name of the file is defined by file_name variable (default README.md).
    The file is written to the specified output directory.

    Args:
        content (list[str]): List of strings with content in Markdown format.
        output (str): Output directory where the file will be saved.
        file_name (str): The name of the file to be written.

    Returns:
        None: Writes the markdown file to disk.

    Raises:
        SystemExit: If writing the file fails (e.g. `output` directory
            does not exist); the exception is logged and the process
            exits with status 1.
    """
    # Add extension to file_name if it is necessary
    if file_name and not file_name.endswith(".md"):
        file_name += ".md"
    # Write final content string to file
    with open(Path(output, file_name or "README.md"), "w", encoding="utf-8") as f:
        f.writelines(content)


def format_number(value: Any, precision: int) -> str:
    """Format input value with specified precision.

    This function formats `float` with thousands separators at `precision`
    decimal places if its string representation has `.`; if `float` is in
    exponential form (like `1e+20`), it formats `float` in scientific
    notation. A `tuple` of floats is formatted as `mean ± std`-style pairs
    at `precision` decimal places. An `int` (including `bool`) is formatted
    with thousands separators, ignoring `precision`.

    Args:
        value (float | tuple | int): Number or a tuple of numbers to format.
        precision (int): Number of decimal places to format `float`/`tuple`
            values; ignored for `int` values.

    Returns:
        str: Formatted number(s) as a string.
    """
    if isinstance(value, float):
        if "." in str(value):
            value = f"{value:,.{precision}f}"
        else:
            value = f"{value:.{precision}e}"
    elif isinstance(value, tuple):
        value = " ± ".join([f"{v:,.{precision}f}" for v in value])
    elif isinstance(value, int):
        value = f"{value:,}"
    return str(value)
