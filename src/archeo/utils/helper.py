from typing import Callable

from archeo.env import ENABLE_PRERELEASE_WARNING
from archeo.logger import get_logger


local_logger = get_logger(__name__)


def pre_release(func: Callable) -> Callable:
    """Decorator to mark a function as a pre-release feature."""

    def wrapper(*args, **kwargs):
        """Wrapper function to log a warning message before executing the decorated function."""

        if ENABLE_PRERELEASE_WARNING:
            local_logger.warning(
                "%s is a pre-release feature. Correctness is not guaranteed.",
                func.__name__,
            )

        return func(*args, **kwargs)

    return wrapper
