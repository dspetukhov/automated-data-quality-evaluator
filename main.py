"""Entry point for the automated data quality evaluation pipeline."""

import json
import sys
from pathlib import Path

from preprocess import make_preprocessing
from report import make_report
from utility import exception_handler, logging, read_source


@exception_handler()
def main(config_file_path: Path) -> None:
    """Execute the data quality evaluation pipeline from a JSON config file.

    Loads the configuration, reads the data source, preprocesses the data,
    and generates a markdown report. Any exception raised by the pipeline
    steps is caught by `@exception_handler`, logged, and suppressed.

    Args:
        config_file_path (Path): Path to the JSON configuration file.

    Raises:
        SystemExit: If `config_file_path` is not a file, or
            if `"source"` key is absent or null in the configuration.
    """
    # Try to load the configuration file
    if config_file_path.is_file():
        with open(config_file_path, encoding="utf-8") as file:
            config = json.load(file)
            logging.info(f"Configuration loaded: {config_file_path}")
    else:
        raise SystemExit("Exit: configuration file wasn't found")

    # Proceed if configuration was loaded and contains `source`
    if config.get("source") is not None:
        source_data = read_source(config["source"])
        # Preprocess data
        df, metadata = make_preprocessing(source_data, config)
        # Generate a report if preprocessing was successful
        make_report(df, metadata, config)
    else:
        raise SystemExit("Exit: configuration is missing required 'source' section")


if __name__ == "__main__":
    if len(sys.argv) == 2:
        main(Path(sys.argv[1]))
    else:
        raise SystemExit("Usage: python main.py <config_file_path>")
