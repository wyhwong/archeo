import os

import pandas as pd

import archeo.logger


local_logger = archeo.logger.get_logger(__name__)


def check_and_create_dir(dirpath: str) -> None:
    """Check if the directory exists, if not create it.
    NOTE: We recursively create the directory if it does not exist.

    Args:
        dirpath (str): The directory path
    """

    if not os.path.exists(dirpath):
        os.makedirs(dirpath, exist_ok=True)

    local_logger.info("Created directory: %s", dirpath)


def read_data(filepath: str) -> pd.DataFrame:
    """Read data from a file according to the file extension.

    Args:
        filepath (str): The file path

    Returns:
        pd.DataFrame: The data in a pandas DataFrame
    """

    filepath = filepath.lower()

    if filepath.endswith(".feather") or filepath.endswith(".ipc"):
        return pd.read_feather(filepath)

    if filepath.endswith(".csv"):
        return pd.read_csv(filepath)

    if filepath.endswith(".parquet"):
        return pd.read_parquet(filepath)

    if filepath.endswith(".json"):
        return pd.read_json(filepath)

    raise ValueError(f"File extension not supported: {filepath}")
