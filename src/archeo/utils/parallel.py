import multiprocessing
import warnings
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from typing import Any, Callable, Optional

from tqdm import tqdm

from archeo.utils.logger import get_logger


LOGGER = get_logger(__name__)


def get_available_cores() -> int:
    """Get the number of available CPU cores.

    Returns:
        int: The number of available CPU cores.
    """

    return multiprocessing.cpu_count()


def get_n_workers(n_workers: int) -> int:
    """Get the number of workers to use for parallel processing."""

    max_workers = get_available_cores()

    if n_workers == -1:
        return max_workers

    _n_workers = max(1, min(n_workers, max_workers))
    if _n_workers != n_workers:
        LOGGER.warning(
            "Requested number of workers (%d) is not valid. Using %d / %d workers instead.",
            n_workers,
            _n_workers,
            max_workers,
        )
    return _n_workers


def multithread_run(
    func: Callable,
    input_kwargs: list[dict[str, Any]],
    n_threads: Optional[int] = None,
) -> list[Any]:
    """Run the function with the arguments.

    Args:
        func (Callable): The function to be executed.
        input_kwargs (list[dict[str, Any]]): The input arguments.
        n_threads (Optional[int]): The number of threads to be used.

    Returns:
        results (list[Any]): The results of the function.
    """

    results = [None] * len(input_kwargs)

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=UserWarning)
        warnings.filterwarnings("ignore", category=RuntimeWarning)
        warnings.filterwarnings("ignore", category=FutureWarning)

        with ThreadPoolExecutor(max_workers=n_threads) as exc:
            futures = [exc.submit(func, **kwargs) for kwargs in input_kwargs]
            results = [future.result() for future in tqdm(futures, total=len(futures))]

    return results


def multiprocess_run(
    func: Callable,
    input_kwargs: list[dict[str, Any]],
    n_processes: Optional[int] = None,
) -> list[Any]:
    """Run the function with the arguments.

    Args:
        func (Callable): The function to be executed.
        input_kwargs (list[dict[str, Any]]): The input arguments.
        n_processes (Optional[int]): The number of processes to be used.

    Returns:
        results (list[Any]): The results of the function.
    """

    results = [None] * len(input_kwargs)

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=UserWarning)
        warnings.filterwarnings("ignore", category=RuntimeWarning)
        warnings.filterwarnings("ignore", category=FutureWarning)

        with ProcessPoolExecutor(max_workers=n_processes) as exc:
            futures = [exc.submit(func, **kwargs) for kwargs in input_kwargs]
            results = [future.result() for future in tqdm(futures, total=len(futures))]

    return results
