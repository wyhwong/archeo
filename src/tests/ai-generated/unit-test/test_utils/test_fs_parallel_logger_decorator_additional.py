import logging

import pandas as pd
import pytest

from archeo.utils.decorator import pre_release
from archeo.utils.fs import check_and_create_dir, load_dataframe
from archeo.utils.logger import get_logger
from archeo.utils.parallel import get_n_workers, multithread_run


def _plus_one(x: int) -> int:
    return x + 1


def test_check_and_create_dir_idempotent(tmp_path):
    d = tmp_path / "abc"
    check_and_create_dir(str(d))
    assert d.exists()
    check_and_create_dir(str(d))
    assert d.exists()


def test_load_dataframe_csv_json(tmp_path):
    df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
    csv_path = tmp_path / "x.csv"
    json_path = tmp_path / "x.json"
    df.to_csv(csv_path, index=False)
    df.to_json(json_path, orient="records")

    out_csv = load_dataframe(str(csv_path))
    out_json = load_dataframe(str(json_path))

    assert out_csv.shape == (2, 2)
    assert out_json.shape[0] == 2


def test_load_dataframe_unsupported(tmp_path):
    p = tmp_path / "x.txt"
    p.write_text("x", encoding="utf-8")
    with pytest.raises(ValueError, match="Unsupported file format"):
        load_dataframe(str(p))


def test_get_n_workers_bounds(monkeypatch):
    monkeypatch.setattr("archeo.utils.parallel.get_available_cores", lambda: 4)
    assert get_n_workers(-1) == 4
    assert get_n_workers(0) == 1
    assert get_n_workers(100) == 4
    assert get_n_workers(2) == 2


def test_multithread_run_happy_path():
    out = multithread_run(_plus_one, input_kwargs=[{"x": 1}, {"x": 2}, {"x": 3}], n_threads=2)
    assert out == [2, 3, 4]


def test_get_logger_reuse_and_invalid_file(tmp_path):
    logger = get_logger("qa_logger_reuse")
    logger2 = get_logger("qa_logger_reuse")
    assert logger is logger2

    bad_file = tmp_path / "missing_dir" / "x.log"
    with pytest.raises(FileNotFoundError):
        get_logger("qa_bad_file", log_filepath=str(bad_file))


def test_pre_release_decorator_invokes_function(monkeypatch):
    called = {"n": 0}

    @pre_release
    def f(a, b):
        called["n"] += 1
        return a + b

    monkeypatch.setattr("archeo.utils.decorator.ENABLE_PRERELEASE_WARNING", False)
    assert f(2, 3) == 5
    assert called["n"] == 1
