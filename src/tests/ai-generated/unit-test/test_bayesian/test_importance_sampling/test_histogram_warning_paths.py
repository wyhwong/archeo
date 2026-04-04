import numpy as np
import pandas as pd

from archeo.bayesian.importance_sampling.resampler.assume_independence import get_histogram_1d
from archeo.bayesian.importance_sampling.resampler.generic import get_histogram_dd
from archeo.data_structures.math import Domain


def test_get_histogram_1d_warning_branch(monkeypatch):
    calls = {"warn": 0}
    monkeypatch.setattr(
        "archeo.bayesian.importance_sampling.resampler.assume_independence.LOGGER.warning",
        lambda msg: calls.__setitem__("warn", calls["warn"] + 1),
    )
    monkeypatch.setattr("numpy.isclose", lambda *args, **kwargs: False)

    s = pd.Series(np.linspace(0, 1, 100), name="a_1")
    _ = get_histogram_1d(s, nbins=10, bounds=Domain(low=0, high=1))
    assert calls["warn"] == 1


def test_get_histogram_dd_warning_branch(monkeypatch):
    calls = {"warn": 0}
    monkeypatch.setattr(
        "archeo.bayesian.importance_sampling.resampler.generic.LOGGER.warning",
        lambda msg: calls.__setitem__("warn", calls["warn"] + 1),
    )
    monkeypatch.setattr("numpy.isclose", lambda *args, **kwargs: False)

    X = np.column_stack([np.linspace(0, 1, 200), np.linspace(0, 2, 200)])
    _ = get_histogram_dd(X, nbins=[10, 10], bounds=[Domain(low=0, high=1), Domain(low=0, high=2)])
    assert calls["warn"] == 1
