import multiprocessing
import os
import warnings
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from typing import Any, Callable, Optional

from tqdm import tqdm

from archeo.utils.logger import get_logger


LOGGER = get_logger(__name__)


def multiprocessing_disabled() -> bool:
    """Return whether multiprocessing is disabled by environment variable."""

    return os.getenv("ARCHEO_DISABLE_MULTIPROCESSING", "false").lower() == "true"


def threading_disabled() -> bool:
    """Return whether threading is disabled by environment variable."""

    return os.getenv("ARCHEO_DISABLE_THREADING", "false").lower() == "true"


def get_available_cores() -> int:
    """Return number of available CPU cores.

    Returns:
        int: CPU core count visible to the process.
    """

    return multiprocessing.cpu_count()


def get_n_workers(n_workers: int) -> int:
    """Normalize requested worker count to a valid value.

    Args:
        n_workers (int): Requested workers. Use `-1` for all available cores.

    Returns:
        int: Effective worker count clipped to valid range.
    """

    if multiprocessing_disabled():
        return 1

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
    """Execute a function over kwargs payloads using a thread pool.

    Args:
        func (Callable): Callable to execute.
        input_kwargs (list[dict[str, Any]]): List of keyword-argument dictionaries.
        n_threads (Optional[int]): Maximum thread count.

    Returns:
        list[Any]: Results in submission order.
    """

    if threading_disabled() or n_threads == 1:
        return [func(**kwargs) for kwargs in tqdm(input_kwargs, total=len(input_kwargs))]

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=UserWarning)
        warnings.filterwarnings("ignore", category=RuntimeWarning)
        warnings.filterwarnings("ignore", category=FutureWarning)

        with ThreadPoolExecutor(max_workers=n_threads) as exc:
            futures = [exc.submit(func, **kwargs) for kwargs in input_kwargs]
            return [future.result() for future in tqdm(futures, total=len(futures))]


def multiprocess_run(
    func: Callable,
    input_kwargs: list[dict[str, Any]],
    n_processes: Optional[int] = None,
    timeout: Optional[float] = None,
) -> list[Any]:
    """Execute a function over kwargs payloads using a process pool.

    Args:
        func (Callable): Callable to execute.
        input_kwargs (list[dict[str, Any]]): List of keyword-argument dictionaries.
        n_processes (Optional[int]): Maximum process count.
        timeout (Optional[float]): Maximum wait time, in seconds, for all futures.

    Returns:
        list[Any]: Results in submission order.
    """

    if not input_kwargs:
        return []

    if multiprocessing_disabled() or n_processes == 1:
        return [func(**kwargs) for kwargs in tqdm(input_kwargs, total=len(input_kwargs))]

    n_processes = max(1, min(n_processes or get_available_cores(), len(input_kwargs)))

    results = [None] * len(input_kwargs)
    ctx = multiprocessing.get_context("spawn")

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=UserWarning)
        warnings.filterwarnings("ignore", category=RuntimeWarning)
        warnings.filterwarnings("ignore", category=FutureWarning)

        with ProcessPoolExecutor(max_workers=n_processes, mp_context=ctx) as exc:
            future_to_index = {exc.submit(func, **kwargs): i for i, kwargs in enumerate(input_kwargs)}

            for future in tqdm(as_completed(future_to_index, timeout=timeout), total=len(future_to_index)):
                index = future_to_index[future]
                results[index] = future.result()

    return results
