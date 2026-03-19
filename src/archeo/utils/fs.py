import os

import pandas as pd

from archeo.utils.logger import get_logger


LOGGER = get_logger(__name__)


def check_and_create_dir(dirpath: str) -> None:
    """Check if the directory exists, if not create it.
    NOTE: We recursively create the directory if it does not exist.

    Args:
        dirpath (str): The directory path
    """

    if not os.path.exists(dirpath):
        os.makedirs(dirpath, exist_ok=True)

    LOGGER.info("Created directory: %s", dirpath)


def load_dataframe(filepath: str) -> pd.DataFrame:
    """Load a dataframe from a given filepath. The function supports both parquet and csv formats.

    Args:
        filepath (str): The path to the data file.

    Returns:
        pd.DataFrame: The loaded dataframe.
    """

    if filepath.lower().endswith(".parquet"):
        df = pd.read_parquet(filepath)
        LOGGER.info("Loaded dataframe from parquet file: %s", filepath)
        return df

    if filepath.lower().endswith(".csv"):
        df = pd.read_csv(filepath)
        LOGGER.info("Loaded dataframe from csv file: %s", filepath)
        return df

    if filepath.lower().endswith(".json"):
        df = pd.read_json(filepath)
        LOGGER.info("Loaded dataframe from json file: %s", filepath)
        return df

    if filepath.lower().endswith(".ipc") or filepath.lower().endswith(".feather"):
        df = pd.read_feather(filepath)
        LOGGER.info("Loaded dataframe from feather file: %s", filepath)
        return df

    if filepath.lower().endswith(".xlsx") or filepath.lower().endswith(".xls"):
        df = pd.read_excel(filepath)
        LOGGER.info("Loaded dataframe from excel file: %s", filepath)
        return df

    raise ValueError(
        f"Unsupported file format for filepath: {filepath}. "
        "Supported formats are: .parquet, .csv, .json, .ipc, .feather, .xlsx, .xls"
    )
