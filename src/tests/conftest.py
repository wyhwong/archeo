import pytest


@pytest.fixture(autouse=True)
def prevent_real_multiprocessing_in_unit_tests(monkeypatch, request):
    if request.node.get_closest_marker("allow_multiprocessing"):
        return

    # Only patch project-level helper, not multiprocessing globally.
    def _fail_multiprocess_run(*args, **kwargs):
        raise RuntimeError(
            "Real multiprocessing is disabled in tests. "
            "Monkeypatch archeo.utils.parallel.multiprocess_run or mark test with "
            "@pytest.mark.allow_multiprocessing."
        )

    monkeypatch.setattr("archeo.utils.parallel.multiprocess_run", _fail_multiprocess_run)
