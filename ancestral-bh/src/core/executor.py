from abc import ABC, abstractmethod
from concurrent.futures import ProcessPoolExecutor, wait
from typing import Any, Callable, Optional

from tqdm import tqdm

import env
import logger


local_logger = logger.get_logger(__name__)


class MultiExecutor(ABC):
    """
    Multi-executor base class.
    """

    def __init__(self, max_executors: Optional[int] = None) -> None:
        """
        Constructor.

        Args:
        -----
            max_executors (Optional[int]):
                The maximum number of executors.

        Returns:
        -----
            None
        """

        self.max_executors = max_executors

    @abstractmethod
    def run(self, func: Callable, input_kwargs: list[dict[str, Any]]) -> list[Any]:
        """
        Run the function with the arguments.

        Args:
        -----
            func (Callable):
                The function to be executed.

            input_kwargs (list[dict[str, Any]]):
                The input arguments.

        Returns:
        -----
            results (list[Any]):
                The results of the function.
        """

    def _check_args(self, func: Callable, input_kwargs: list[dict[str, Any]]) -> None:
        """
        Check the input arguments.

        Args:
        -----
            func (Callable):
                The function to be executed.

            input_kwargs (list[dict[str, Any]]):
                The input arguments.

        Returns:
        -----
            None
        """

        if not callable(func):
            local_logger.error("func must be a Callable, not %s", type(func))
            raise TypeError(f"func must be a Callable, not {type(func)}")

        if not isinstance(input_kwargs, list):
            local_logger.error("input_kwargs must be a list, not %s", type(input_kwargs))
            raise TypeError(f"input_kwargs must be a list, not {type(input_kwargs)}")

        func_keywords = set(func.__code__.co_varnames)
        for kwargs in input_kwargs:
            input_keywords = set(kwargs.keys())
            if not func_keywords.issubset(input_keywords):
                local_logger.error(
                    "input_kwargs must contain all keywords of function, not %s.",
                    input_keywords,
                )
                raise ValueError(f"input_kwargs must contain all keywords of func, not {input_keywords}")

            if not input_keywords.issubset(func_keywords):
                local_logger.error(
                    "input_kwargs must contain only keywords of function, not %s.",
                    input_keywords,
                )
                raise ValueError(f"input_kwargs must contain only keywords of func, not {input_keywords}")


class MultiProcessExecutor(MultiExecutor):
    """
    Multi-process executor.
    """

    def __init__(self, max_executors: Optional[int] = None) -> None:
        """
        Constructor.

        Args:
        -----
            max_executors (Optional[int]):
                The maximum number of executors.

        Returns:
        -----
            None
        """

        super().__init__(max_executors if max_executors else env.MAX_MULTIPROCESS_WORKER)

        local_logger.info("Initialized MultiProcessExecutor, max_executors: %d.", self.max_executors)

    def run(self, func: Callable, input_kwargs: list[dict[str, Any]]) -> list[Any]:
        """
        Run the function with the arguments.

        Args:
        -----
            func (Callable):
                The function to be executed.

            input_kwargs (list[dict[str, Any]]):
                The input arguments.

        Returns:
        -----
            results (list[Any]):
                The results of the function.
        """

        self._check_args(func, input_kwargs)

        with ProcessPoolExecutor(max_workers=self.max_executors) as executor:
            futures = [executor.submit(func, **kwargs) for kwargs in input_kwargs]
            wait(tqdm(futures))

        results = [future.result() for future in futures]

        return results
