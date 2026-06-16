import json
import sys
from pathlib import Path

from preprocess import make_preprocessing
from report import make_report
from utility import exception_handler, logging, read_source


@exception_handler()
def main(config_file_path: Path) -> None:
    """
    Main function to execute the data quality evaluation pipeline.
    Requires correct configuration preset.

    This pipeline includes the following steps:
      1. Checks if the configuration exists
         and loads it as a configuration using json library.
      2. Reads the data source as specified in the configuration.
      3. Preprocess data: applies filters and transformations (optional),
         aggregates data over dates.
      4. Generates a report as a markdown file based on the preprocessed data.

    Raises:
        SystemExit: If the configuration file wasn't found
            or doesn't have 'source' section.
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
