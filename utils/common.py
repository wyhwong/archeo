import os
import json
import h5py
import yaml
from .logger import get_logger

LOGGER = get_logger(logger_name="Utils | Common")


def read_dict_from_yml(filepath: str) -> dict:
    LOGGER.debug(f"Loading dict from {filepath}...")
    with open(filepath, "r") as file:
        output_dict = yaml.load(file, Loader=yaml.SafeLoader)
    LOGGER.debug(f"Loaded dict: {output_dict}.")
    return output_dict


def read_dict_from_json(filepath: str) -> dict:
    LOGGER.debug(f"Reading dict from {filepath}...")
    with open(filepath, "r") as file:
        output_dict = json.load(file)
    LOGGER.debug(f"Available keys in the Json file: {output_dict.keys()}")
    return output_dict


def read_posterior_from_json(filepath: str) -> dict:
    return read_dict_from_json(filepath=filepath)["posterior"]["content"]


def read_posterior_from_h5(filepath: str, fits="NRSur7dq4") -> dict:
    return h5py.File(filepath, "r")[fits]["posterior_samples"]


def get_main_config() -> dict:
    return read_dict_from_yml(filepath="config/main_config.yml")


def get_prior_config() -> dict:
    return read_dict_from_yml(filepath="config/prior_config.yml")


def save_dict_as_yml(savepath: str, input_dict: dict) -> None:
    LOGGER.debug(f"Saving dict: {input_dict}...")
    with open(savepath, "w") as file:
        yaml.dump(input_dict, file)
    LOGGER.debug(f"Saved config at {savepath}.")


def check_and_create_dir(dirpath: str) -> bool:
    exist = os.path.isdir(dirpath)
    LOGGER.debug(f"{dirpath} exists: {exist}.")
    if not exist:
        LOGGER.info(f"{dirpath} does not exist, creating one...")
        os.mkdir(dirpath)
        LOGGER.debug(f"Created {dirpath}")
    return exist
