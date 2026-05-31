from typing import Callable

from archeo.utils.env import ENABLE_PRERELEASE_WARNING
from archeo.utils.logger import get_logger


LOGGER = get_logger(__name__)


def pre_release(func: Callable) -> Callable:
    """Mark a callable as pre-release and emit warning on invocation.

    Args:
        func (Callable): Function to wrap.

    Returns:
        Callable: Wrapped function.
    """

    def wrapper(*args, **kwargs):
        """Execute wrapped function with optional pre-release warning.

        Args:
            *args: Positional arguments for wrapped function.
            **kwargs: Keyword arguments for wrapped function.

        Returns:
            Any: Wrapped function return value.
        """

        if ENABLE_PRERELEASE_WARNING:
            LOGGER.warning(
                "%s is a pre-release feature. Correctness is not guaranteed.",
                func.__name__,
            )

        return func(*args, **kwargs)

    return wrapper
