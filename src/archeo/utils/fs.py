import os

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
