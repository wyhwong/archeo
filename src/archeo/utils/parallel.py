from concurrent.futures import ThreadPoolExecutor, wait
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

    with ThreadPoolExecutor(max_workers=n_threads) as exc:
        futures = [exc.submit(func, **kwargs) for kwargs in input_kwargs]
        wait(tqdm(futures))

    results = [future.result() for future in futures]

    return results
