from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from typing import Any, Callable, Optional

from tqdm import tqdm


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

    with ThreadPoolExecutor(max_workers=n_threads) as exc:
        futures = [exc.submit(func, **kwargs) for kwargs in input_kwargs]
        results = [future.result() for future in tqdm(as_completed(futures), total=len(futures))]

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

    with ProcessPoolExecutor(max_workers=n_processes) as exc:
        futures = [exc.submit(func, **kwargs) for kwargs in input_kwargs]
        results = [future.result() for future in tqdm(as_completed(futures), total=len(futures))]

    return results
