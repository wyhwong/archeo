import os
import json
import yaml
import utils.logger


logger = utils.logger.get_logger(logger_name="utils|common")


def read_dict_from_yml(filepath: str) -> dict:
    """
    Read a dictionary from a yaml file.

    Parameters
    ----------
    filepath : str
        Path to the yaml file.

    Returns
    -------
    output_dict : dict
        Dictionary read from the yaml file.
    """
    with open(filepath, "r") as file:
        return yaml.load(file, Loader=yaml.SafeLoader)


def read_dict_from_json(filepath: str) -> dict:
    """
    Read a dictionary from a json file.

    Parameters
    ----------
    filepath : str
        Path to the json file.

    Returns
    -------
    output_dict : dict
        Dictionary read from the json file.
    """
    with open(filepath, "r") as file:
        return json.load(file)


def save_dict_as_yml(savepath: str, input_dict: dict) -> None:
    """
    Save a dictionary as a yaml file.

    Parameters
    ----------
    savepath : str
        Path to save the yaml file.

    input_dict : dict
        Dictionary to save.
    """
    with open(savepath, "w") as file:
        yaml.dump(input_dict, file)


def check_and_create_dir(dirpath: str) -> None:
    """
    Check if a directory exists, if not, create one.

    Parameters
    ----------
    dirpath : str
        Path to the directory.

    Returns
    -------
    None
    """
    if not os.path.isdir(dirpath):
        logger.info(f"{dirpath} does not exist, creating one...")
        os.mkdir(dirpath)
