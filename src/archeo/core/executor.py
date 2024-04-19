from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor, wait
from typing import Any, Callable, Optional

from tqdm import tqdm

import archeo.env
import archeo.logger


local_logger = archeo.logger.get_logger(__name__)


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


class MultiThreadExecutor(MultiExecutor):
    """
    Multi-thread executor.
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

        super().__init__(max_executors if max_executors else archeo.env.MAX_MULTITHREAD_WORKER)

        local_logger.info("Initialized MultiThreadExecutor, max_executors: %d.", self.max_executors)

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

        with ThreadPoolExecutor(max_workers=self.max_executors) as executor:
            futures = [executor.submit(func, **kwargs) for kwargs in input_kwargs]
            wait(tqdm(futures))

        results = [future.result() for future in futures]

        return results
