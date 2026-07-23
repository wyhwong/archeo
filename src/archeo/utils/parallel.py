import multiprocessing
import warnings
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from typing import Any, Callable, Optional

from tqdm import tqdm

from archeo.utils.logger import get_logger


LOGGER = get_logger(__name__)


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

    if n_threads == 1:
        return [func(**kwargs) for kwargs in tqdm(input_kwargs, total=len(input_kwargs))]

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
    """Execute a function over kwargs payloads using a process pool.

    Args:
        func (Callable): Callable to execute.
        input_kwargs (list[dict[str, Any]]): List of keyword-argument dictionaries.
        n_processes (Optional[int]): Maximum process count.

    Returns:
        list[Any]: Results in submission order.
    """

    if n_processes == 1:
        return [func(**kwargs) for kwargs in tqdm(input_kwargs, total=len(input_kwargs))]

    results = [None] * len(input_kwargs)

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=UserWarning)
        warnings.filterwarnings("ignore", category=RuntimeWarning)
        warnings.filterwarnings("ignore", category=FutureWarning)

        with ProcessPoolExecutor(max_workers=n_processes) as exc:
            futures = [exc.submit(func, **kwargs) for kwargs in input_kwargs]
            results = [future.result() for future in tqdm(futures, total=len(futures))]

    return results
