# Automated Quality Evaluation of Temporal Data

An automated configurable Python tool for evaluating the quality of temporal data.

This tool evaluates data quality by calculating descriptive statistics (e.g., number of unique values) for each column in your dataset across configurable time intervals. It generates structured markdown reports with charts and tables that visualize changes in these statistics over time.

The final assessment of data consistency, validity, and overall quality is the responsibility of an individual reviewing the markdown report.

This tool is particularly useful for:

- Validating consistency and finding faults in temporal data,
- Identifying temporal anomalies and detecting data drift,
- Monitoring data quality in production pipelines.

A detailed explanation of why it was done and how it works is available on [medium.com/@dspetukhov](https://medium.com/@dspetukhov/automated-quality-evaluation-of-temporal-data-75f2a6f89627).

Powered by [Polars](https://docs.pola.rs/) and [Plotly](https://docs.plotly.com/).

## Table of contents

- [Quick start](#quick-start)
- [Project overview](#project-overview)
  - [Features](#features)
  - [Structure](#structure)
  - [Configuration](#configuration)
- [Troubleshooting](#troubleshooting)
- [Dataset reading examples](#dataset-reading-examples)
  - [Kaggle](#kaggle)
  - [Hugging Face](#hugging-face)

## Requirements

```txt
# Python 3.10 or higher

polars>=1.37.0
plotly==6.3.0
kaleido==0.2.1
tabulate==0.9.0
```

## Quick start

1. **Specify your source of data and date column name:**

    Edit [**config.json**](config.json):

    ```json
    {
        "source": {
            "file_path": "path/to/your/dataset.csv"
        },
        "date_column": "timestamp"
    }
    ```

    Supported file formats: CSV, XLSX, Parquet, and Iceberg. Cloud storage and PostgreSQL databases are also supported.
    Details about the `source` definition can be found in the [**source**](#source) section of the configuration description.

    The `date_column` parameter is expected to be a name of the date or datetime type column. Various examples of defining this parameter can be found in [**Dataset reading examples**](#dataset-reading-examples).

2. **Run evaluation process:**

    ```bash
    python main.py config.json
    ```

    A configuration file is expected as a command-line argument.

3. **Review generated markdown report**

    By default, the report will be:
    - named README.md, which can be changed by the parameter `name` in the [**markdown**](#markdown) section of the configuration description,
    - saved into a directory whose name matches the input filename. This can be altered according to the [**output**](#output) section of the configuration description.

[Back to table of contents](#table-of-contents)

## Project overview

### Features

- **Comprehensive data evaluation**: descriptive statistics to evaluate data changes over configurable time intervals,
- **Configurable time intervals**: (e.g., 1h, 13h, 1d, 6d, 1d1h) to analyze data changes over different time scales,
- **Various data sources**: CSV, XLSX, Parquet, and Iceberg file formats supported as well as reading from cloud providers or PostgreSQL databases,
- **Flexible & performant data preprocessing**: data filtering and transformation using SQL expressions powered by Polars with lazy evaluation,
- **Outliers detection**: evaluation and visual representation of anomalous changes based on IQR or Z-score criteria,
- **Professional markdown reports**: with formatted tables and customized charts embedded,
- **Configuration in one place**: various preprocessing and reporting parameters specified in a human-readable JSON file passed as a command line argument.

### Structure

| Module                         | Description                                                                                              |
|--------------------------------|----------------------------------------------------------------------------------------------------------|
| `main.py`                      | Entry point: loads configuration, reads, preprocesses, and evaluates data, generates report              |
| `preprocess.py`                | Preprocesses data by applying a filter and transformations, aggregates data by date                      |
| `evaluate.py`                  | Calculates descriptive statistics for aggregated data, detects outliers based on IQR or Z-score criteria |
| `plot.py`                      | Generates charts with outliers highlighted                                                               |
| `report.py`                    | Generates structured markdown reports with tables and charts embedded                                    |
| `style.css`                    | Report and table styling                                                                                 |
| `config.json`                  | Configuration                                                                                            |
| `utility/__init__.py`          | Default Plotly template, utility imports                                                                 |
| `utility/setup_logging.py`     | Logging configuration                                                                                    |
| `utility/handle_data.py`       | Reads data from file, cloud, or database into Polars LazyFrame                                           |
| `utility/handle_exceptions.py` | Decorator to handle exceptions                                                                           |

### Configuration

Data evaluation configuration is defined in a single JSON file ([config.json](config.json)) by the following sections:

| Section / Parameter    | Description                                                              | Expected parameters                                                                                     |
|------------------------|--------------------------------------------------------------------------|---------------------------------------------------------------------------------------------------------|
| `source`               | Configuration to read data (Required)                                    | `file_path`, `file_format`, `storage_options`, `schema_overrides`                                       |
| `engine`               | Polars engine used to process the data (Optional)                        |                                                                                                         |
| `streaming_chunk_size` | Chunk size for streaming engine to avoid Out of Memory errors (Optional) |                                                                                                         |
| `output`               | Directory to save report and charts (Optional)                           |                                                                                                         |
| `filter`               | SQL expression to filter data by rows and by columns (Optional)          |                                                                                                         |
| `transformations`      | Dictionary of SQL expressions to transform data by columns (Optional)    |                                                                                                         |
| `date_column`          | Column to aggregate data by time intervals (Required)                    |                                                                                                         |
| `time_interval`        | Time interval to aggregate data (Optional)                               |                                                                                                         |
| `target_column`        | Column to calculate target average (Optional)                            |                                                                                                         |
| `columns_to_exclude`   | List of columns to be excluded from evaluation (Optional)                |                                                                                                         |
| `outliers`             | Outlier detection settings (Optional)                                    | `criterion`, `multiplier_iqr`, `threshold_z_score`                                                      |
| `markdown`             | Markdown report settings (Optional)                                      | `name`, `css_style`, `float_precision`                                                                  |
| `plotly`               | Plotly styling settings (Optional)                                       | `plot`, `outliers`, `layout`, `annotations`, `grid`, `subplots`, `format`, `scale_factor`, `tickformat` |

Each of these sections is described below in detail:

#### `source`

This section specifies parameters to read the source of data using Polars:

- `file_path` is the mandatory parameter that defines the path to the file to read,
- `file_format` is required when the file being read is missing an extension at the end of its name or when reading from a directory with partitioned files,
- `storage_options` is required for reading from cloud providers. If not explicitly specified, Polars will try to get credentials from environment variables. Explicit specification is recommended:

```python
storage_options = {
    # Definitions with a `$` sign as the first symbol will be interpreted as environment variables
    # Definitions without a `$` sign will be interpreted as string literals
    "aws_access_key_id": "$S3_KEY_ID",
    ...
}
```

- `schema_overrides` may be required for CSV or XLSX files to alter column data types during schema inference:

  - when a date or datetime column does not match ISO 8601 standard,
  - when a categorical string column is inferred as a numerical one.

Supported types for `schema_overrides` include `String`, `Date`, and `Datetime`.

More details of how `schema_overrides` can be useful when reading data can be found in the [Troubleshooting](#troubleshooting).

When reading from a PostgreSQL database, use `uri` and `query` instead of the file-related parameters above:

```python
{
    # `uri` can be specified as an environment variable, e.g., "$PG_URI"
    "uri": "postgresql://username:password@server:port/database",
    "query": "select * from foo"
}
```

#### `engine`

This parameter specifies the engine used to process the data. Possible values include:

- `auto` (default),
- `gpu` for data processing on the GPU,
- `streaming` for processing datasets that do not fit entirely in memory.

If the data cannot be processed using the specified engine, Polars will use its in-memory engine.

#### `streaming_chunk_size`

This parameter specifies the chunk size used in the streaming engine to avoid Out of Memory errors. Applicable only when the `engine` parameter is set to `streaming`.

#### `output`

This parameter specifies the directory where the report and charts will be saved. By default, the output directory:

- will be created in the current directory,
- its name will match the input filename when reading from a file or be named `postgresql` when reading from a database.

It is recommended to define this parameter explicitly.

#### `filter`

This parameter specifies a SQL expression to filter data by rows and/or by columns.

#### `transformations`

This parameter specifies a dictionary with at least one key-value pair, where the key is a column name to be created or replaced and the value is a SQL expression to be applied to one or multiple columns in the data. If the key matches any existing column name in the data, it replaces that column with transformed values; otherwise, a new column is created.

#### `date_column`

This parameter specifies the name of a date or datetime type column to use for aggregating data over time intervals. If not specified, the tool will try to use a column literally named `date_column`, which can be created using `transformations`. If there is no such column, the tool will print the data schema and exit.

#### `time_interval`

This parameter is used to divide the date or datetime range in `date_column` into equal time intervals. The division is implemented with [polars.Expr.dt.truncate](https://docs.pola.rs/api/python/stable/reference/expressions/api/polars.Expr.dt.truncate.html). The default value is `1d` (one day), so any other value must be specified explicitly.

#### `target_column`

This optional parameter specifies a column in the data to calculate the target average, which is the class balance in machine learning binary classification problems. The calculation result will be shown in the `Overview` section of the markdown report.

If not specified, the tool will try to use a column literally named `target_column`, which can be created using `transformations`, and if there is no such column in the data, the target average will not be calculated.

#### `columns_to_exclude`

This optional parameter specifies a list of columns to be excluded from the evaluation process.

#### `outliers`

This section specifies parameters to evaluate outliers and highlight outlier regions on charts:

- `criterion` defines a method for outlier detection: `IQR` or `Z-score`. This parameter must be specified to display outlier regions on charts,
- `multiplier_iqr` defines the multiplier for the IQR range to determine outlier boundaries (defaults to 1.5),
- `threshold_z_score` defines the Z-score threshold for identifying outliers (defaults to 3.0).

#### `markdown`

This section specifies parameters related to the markdown report being produced:

- `name` defines the report filename (defaults to `README.md`),
- `css_style` defines the path to a CSS file for table styling, e.g., [style.css](style.css). Specifying table styles is not mandatory, but it improves readability and appearance,
- `float_precision` defines the number of decimal places to format floats in markdown tables (defaults to 4).

#### `plotly`

This section specifies Plotly configuration parameters and styles, which can be adjusted to match your preferences.

- `plot` defines style for [plotly.graph_objs.Scatter](https://plotly.com/python-api-reference/generated/plotly.graph_objects.Scatter.html#plotly.graph_objects.Scatter), which renders evaluated descriptive statistics over time intervals,
- `outliers` defines style for Plotly shapes to highlight outliers,
- `layout` defines parameters to adjust [layout](https://plotly.com/python/reference/layout/). The default chart height equals 512 pixels, default template is `plotly_white`,
- `annotations` defines parameters to adjust [annotations](https://plotly.com/python/reference/layout/annotations/), used to modify font in subplot titles.
- `grid` defines style for grid lines,
- `subplots` defines extra parameters to adjust spacing in the [subplot grid](https://plotly.com/python-api-reference/generated/plotly.subplots.make_subplots.html),
- `format` defines the file format for saving charts; supports PNG (default), JPEG, WebP, SVG, and PDF.
- `scale_factor` defines the scaling factor for charts (defaults to 1),
- `tickformat` defines date format for x-axis (defaults to `%Y-%m-%d`).

If none of the parameters are specified, Plotly will use its default parameters.

[Back to table of contents](#table-of-contents)

## Troubleshooting

This section describes potential issues when processing CSV or XLSX file formats that may cause the tool to fail.

The first issue is caused by a column with mixed alphanumeric values, which may be interpreted as an integer type during schema inference. The problem is that, by default, Polars infers schema from the first 100 rows of CSV and XLSX files, which is defined by the `infer_schema_length` parameter in `scan_csv` or `read_excel` functions.

The solution is to explicitly define the column type as a string during data ingestion:

```python
   "source": {
        "file_path": "/path/to/file.csv",
        "schema_overrides": {
            "column": "String"
        }
    },
```

In such cases, column type transformations (e.g., `cast(column as text)` or `column::text`) typically do not work because of the specifics of Polars' lazy execution and logical plan optimization: it may still use the originally inferred type.

---

Another common issue that causes `ComputeError` is a date or datetime type column being inferred as a string type. In these cases, column type transformation (e.g., `DATE(column, '%Y-%m-%d %H:%M:%S')`) will work if timestamp values are uniform and match the ISO 8601 standard supported by Polars.

If the column has a mixed format, e.g., with and without fractional seconds, but is still consistent with the ISO 8601 standard, it is better to handle it by specifying the column type as date or datetime in `schema_overrides`.

In cases of complex mixed time formats raising `ComputeError`, manual data cleaning to standardize the formats remains the best solution, although such cases are uncommon.

---

The last identified issue relates to processing large CSV files exceeding available RAM. In such cases, the following helps:

- Convert string columns to Polars' categorical type by assigning the `Categorical` value in `schema_overrides`,
- In the configuration, assign the `streaming` value to `engine` and start with a value of approximately 50000 for `streaming_chunk_size` (readjust based on observed memory usage if necessary).

[Back to table of contents](#table-of-contents)

## Dataset reading examples

This tool was tested using publicly available datasets. Full configurations for evaluating these datasets are in the [**examples**](examples) directory. Ready-to-use configuration extracts that require adjusting only a few parameters in the existing [**config.json**](config.json) are listed below:

### [Kaggle](https://www.kaggle.com/datasets?search=fraud&sort=votes&tags=13302-Classification&minUsabilityRating=9.00+or+higher)

- [Metaverse Financial Transactions Dataset](https://www.kaggle.com/datasets/faizaniftikharjanjua/metaverse-financial-transactions-dataset)

```json
    "source": {
        "file_path": "/datasets/metaverse_transactions_dataset.csv",
        "schema_overrides": {
            "timestamp": "Datetime"
        }
    },
    "output": "metaverse-financial-transactions-dataset",
    "transformations": {
        "target_column": "(anomaly = 'high_risk')::int"
    },
    "date_column": "timestamp",
```

- [Credit Card Fraud Prediction](https://www.kaggle.com/datasets/kelvinkelue/credit-card-fraud-prediction)

```json
    "source": {
        "file_path": "/datasets/fraud test.csv",
        "schema_overrides": {
            "trans_date_trans_time": "Datetime",
            "cc_num": "String",
            "zip": "String"
        }
    },
    "output": "credit-card-fraud-prediction",
    "date_column": "trans_date_trans_time",
    "target_column": "is_fraud",
    "columns_to_exclude": ["", "unix_time"],
```

- [Is this a bad transaction?](https://www.kaggle.com/datasets/podsyp/fraud-transactions-detection)

```json
    "source": {
        "file_path": "/datasets/fraud.csv"
    },
    "output": "fraud-transactions-detection",
    "date_column": "rep_loan_date",
    "target_column": "bad_flag",
```

- [Ecommerce Counterfeit Products Dataset](https://www.kaggle.com/datasets/aimlveera/counterfeit-product-detection-dataset)

```json
    "source": {
        "file_path": "/datasets/_counterfeit_transactions.csv"
    },
    "output": "counterfeit-product-detection-dataset",
    "date_column": "transaction_date",
    "target_column": "involves_counterfeit",
    "transformations": {
        "transaction_date": "DATE(transaction_date, '%Y-%m-%d %H:%M:%S')",
        "involves_counterfeit": "involves_counterfeit::int"
    },
```

It is also possible to replace `"transaction_date": "DATE(transaction_date, '%Y-%m-%d %H:%M:%S')",` with:

```json
    "source": {
        "file_path": "/datasets/_counterfeit_transactions.csv",
        "schema_overrides": {
            "transaction_date": "Datetime"
        }
    },
```

- [Financial Transactions Dataset for Fraud Detection](https://www.kaggle.com/datasets/aryan208/financial-transactions-dataset-for-fraud-detection)

```json
    "source": {
        "file_path": "/datasets/financial_fraud_detection_dataset.csv",
        "schema_overrides": {
            "timestamp": "Datetime"
        }
    },
    "output": "financial-transactions-dataset-for-fraud-detection",
    "filter": "select * from self where timestamp::date > '2023-01-01' and timestamp::date < '2024-01-01'",
    "date_column": "timestamp",
    "target_column": "is_fraud",
```

- [IBM Transactions for Anti Money Laundering](https://www.kaggle.com/datasets/ealtman2019/ibm-transactions-for-anti-money-laundering-aml)

```json
    "source": {
        "file_path": "/datasets/LI-Large_Trans.csv",
        "schema_overrides": {
            "Timestamp": "Datetime",
            "From Bank": "Categorical",
            "Account": "Categorical",
            "To Bank": "Categorical",
            "Account_duplicated_0": "Categorical",
            "Receiving Currency": "Categorical",
            "Payment Currency": "Categorical",
            "Payment Format": "Categorical"
        }
    },
    "output": "ibm-transactions-for-anti-money-laundering",
    "engine": "streaming",
    "streaming_chunk_size": 2026,
    "date_column": "Timestamp",
    "target_column": "Is Laundering",
    "transformations": {
        "sender_account": "Account",
        "receiver_account": "Account_duplicated_0"
    },
    "columns_to_exclude": ["Account", "Account_duplicated_0"],
```

**Note:**

- Extracted by `unzip ibm-transactions-for-anti-money-laundering-aml.zip LI-Large_Trans.csv`,
- CSV file weighs 16GB and has 176M rows,
- `schema_overrides` specifications convert string columns to categorical to reduce memory usage during aggregation,
- `engine` and `streaming_chunk_size` values ensure processing on a machine with small RAM; specified values allow processing on a laptop with 8GB RAM (any value of `streaming_chunk_size` below 50000 worked).

[Back to table of contents](#table-of-contents)

### [Hugging Face](https://huggingface.co/datasets?size_categories=or:%28size_categories:10K%3Cn%3C100K,size_categories:100K%3Cn%3C1M,size_categories:1M%3Cn%3C10M,size_categories:10M%3Cn%3C100M,size_categories:100M%3Cn%3C1B,size_categories:1B%3Cn%3C10B,size_categories:10B%3Cn%3C100B,size_categories:100B%3Cn%3C1T,size_categories:n%3E1T%29&sort=trending&search=fraud)

- [CiferAI/Cifer-Fraud-Detection-Dataset-AF](https://huggingface.co/datasets/CiferAI/Cifer-Fraud-Detection-Dataset-AF)

```json
    "source": {
        "file_path": "hf://datasets/CiferAI/Cifer-Fraud-Detection-Dataset-AF/**/*.csv",
        "schema_overrides": {
            "type": "Categorical",
            "nameOrig": "Categorical",
            "nameDest": "Categorical"
        }
    },
    "output": "cifer-fraud-detection-dataset",
    "engine": "streaming",
    "filter": "select * from self where step > 1 and step < 743",
    "transformations": {
        "date_column": "CAST(step AS DATE)"
    },
    "target_column": "isFraud",
    "columns_to_exclude": ["isFlaggedFraud", "step"],
```

**Note:** The specified `transformations` convert integer type column `step`, which is a unit of time (1 step = 1 hour), into a date type column with values starting from 1970-01-01.

- [Tichies/card-fraud](https://huggingface.co/datasets/Tichies/card-fraud)

```json
    "source": {
        "file_path": "hf://datasets/Tichies/card-fraud/saske.csv",
        "schema_overrides": {
            "Transaction_Date": "Datetime",
            "User_ID": "String",
            "Device_ID": "String",
            "Merchant_ID": "String"
        }
    },
    "output": "hf-card-fraud",
    "date_column": "Transaction_Date",
    "target_column": "isFraud",
    "columns_to_exclude": ["Transaction_ID"],
```

- [saifhmb/FraudPaymentData](https://huggingface.co/datasets/saifhmb/FraudPaymentData)

```json
    "source": {
        "file_path": "/datasets/FraudPaymentData.parquet"
    },
    "date_column": "Time_step",
    "target_column": "Label",
    "transformations": {
        "Time_step": "DATE(Time_step, '%m/%d/%Y %H:%M')"
    },
```

**Note:** Downloaded by `ds = load_dataset("saifhmb/FraudPaymentData")`, then saved `ds["train"].to_parquet("FraudPaymentData.parquet")`.

- [Ransaka/fraud_prediction_300K](https://huggingface.co/datasets/Ransaka/fraud_prediction_300K)

```json
    "source": {
        "file_path": "hf://datasets/Ransaka/fraud_prediction_300K/data_50K.parquet"
    },
    "output": "fraud-prediction-300K",
    "date_column": "S_2",
    "target_column": "target",
    "columns_to_exclude": ["customer_ID"],
```

- [Phoenix21/mock_fraud-detection-dataset](https://huggingface.co/datasets/Phoenix21/mock_fraud-detection-dataset)

```json
    "source": {
        "file_path": "hf://datasets/Phoenix21/mock_fraud-detection-dataset/transactions.csv",
        "schema_overrides": {
            "timestamp": "Datetime"
        }
    },
    "output": "mock-fraud-detection-dataset",
    "date_column": "timestamp",
    "target_column": "is_fraud",
    "columns_to_exclude": ["transaction_id"],
```

- [Nooha/cc_fraud_detection_dataset](https://huggingface.co/datasets/Nooha/cc_fraud_detection_dataset)

```json
    "source": {
        "file_path": "/datasets/cc_fraud_detection_dataset.parquet"
    },
    "output": "hf-cc-fraud_detection_dataset",
    "date_column": "trans_date",
    "target_column": "is_fraud",
    "transformations": {
        "acct_num": "acct_num::text"
    },
    "columns_to_exclude": ["trans_time", "unix_time"],
```

**Note:** Downloaded by `ds = load_dataset("Nooha/cc_fraud_detection_dataset")`, then saved `ds["train"].to_parquet("cc_fraud_detection_dataset.parquet")`.

[Back to table of contents](#table-of-contents)
