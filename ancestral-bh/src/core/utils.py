import json
import os

import h5py
import yaml

import logger


local_logger = logger.get_logger(__name__)


def check_and_create_dir(dirpath: str) -> None:
    """
    Check if the directory exists, if not, create it.

    Args:
    -----
        dirpath (str):
            The directory path.

    Returns:
    -----
        None
    """

    if not os.path.isdir(dirpath):
        os.mkdir(dirpath)
    local_logger.info("Created directory: %s", dirpath)


def load_yml(filepath: str) -> dict:
    """
    Load yml file.

    Args:
    -----
        filepath (str):
            Filepath of the yml file.

    Returns:
    -----
        content (dict):
            Content of the yml file.
    """

    local_logger.info("Read yml: %s", filepath)
    with open(filepath, "r", encoding="utf-8") as file:
        content = yaml.load(file, Loader=yaml.SafeLoader)
    return content


def save_as_yml(filepath: str, content: dict) -> None:
    """
    Save dict as yml file.

    Args:
    -----
        filepath (str):
            Filepath of the yml file.
        content (dict):
            Content of the yml file.

    Returns:
    -----
        None
    """

    with open(filepath, "w", encoding="utf-8") as file:
        yaml.dump(content, file)
    local_logger.info("Saved config at %s", filepath)


def load_h5(filepath: str) -> dict:
    """
    Load h5 file.

    Args:
    -----
        filepath (str):
            Filepath of the h5 file.

    Returns:
    -----
        content (dict):
            Content of the h5 file.
    """

    local_logger.info("Read h5: %s", filepath)
    return h5py.File(filepath, "r")


def load_json(filepath: str) -> dict:
    """
    Load json file.

    Args:
    -----
        filepath (str):
            Filepath of the json file.

    Returns:
    -----
        content (dict):
            Content of the json file.
    """

    local_logger.info("Read json: %s", filepath)
    with open(filepath, "r", encoding="utf-8") as file:
        return json.load(file)
