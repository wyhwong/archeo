import pandas as pd

from archeo.bayesian.importance_sampling import ImportanceSamplingData as ISData


def _df():
    return pd.DataFrame({"m_1": [30.0, 31.0, 32.0], "a_1": [0.2, 0.3, 0.4]})


def test_sample_bayes_factor_parallel_independence_branch(monkeypatch):
    data = ISData(
        prior_samples=_df(), posterior_samples=_df(), new_prior_samples=_df(), assume_parameter_independence=True
    )

    monkeypatch.setattr(
        "archeo.bayesian.importance_sampling.resampler.interface.multithread_run",
        lambda func, input_kwargs, n_threads=None: [1.23 for _ in input_kwargs],
    )
    out = data.sample_bayes_factor(n=4, is_parallel=True, n_threads=2)
    assert out.samples == [1.23, 1.23, 1.23, 1.23]


def test_sample_bayes_factor_parallel_generic_branch(monkeypatch):
    data = ISData(
        prior_samples=_df(), posterior_samples=_df(), new_prior_samples=_df(), assume_parameter_independence=False
    )

    monkeypatch.setattr(
        "archeo.bayesian.importance_sampling.resampler.interface.multithread_run",
        lambda func, input_kwargs, n_threads=None: [0.77 for _ in input_kwargs],
    )
    out = data.sample_bayes_factor(n=3, is_parallel=True, n_threads=2)
    assert out.samples == [0.77, 0.77, 0.77]


def test_get_reweighted_samples_dispatches_dd_branch():
    data = ISData(
        prior_samples=_df(), posterior_samples=_df(), new_prior_samples=_df(), assume_parameter_independence=False
    )
    out = data.get_reweighted_samples(random_state=1)
    assert len(out) == len(_df())
