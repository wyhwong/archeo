import numpy as np
import pandas as pd
import pytest

from archeo.bayesian.ancestral_posterior import _retrieve_sample, infer_ancestral_posterior_distribution
from archeo.bayesian.importance_sampling import ImportanceSamplingData as ISData
from archeo.bayesian.importance_sampling.bayes_factor_curve import BayesFactorCurve, CandidatePrior
from archeo.data_structures.bayesian.bayes_factor import BayesFactor


def _prior_df(n=5000):
    rng = np.random.default_rng(42)
    return pd.DataFrame(
        {
            "m_1": rng.uniform(5, 65, n),
            "a_1": rng.uniform(0, 1, n),
            "m_f": rng.uniform(10, 120, n),
            "a_f": rng.uniform(0, 1, n),
            "k_f": rng.uniform(0, 2000, n),
        }
    )


def _posterior_df(n=2000):
    rng = np.random.default_rng(43)
    return pd.DataFrame(
        {
            "m_1": np.clip(rng.normal(35, 5, n), 5, 65),
            "a_1": rng.uniform(0, 1, n),
        }
    )


def test_retrieve_sample_no_match_returns_logL_minus_inf():
    df = pd.DataFrame({"m_f": [10.0], "a_f": [0.1], "x": [1]})
    s = _retrieve_sample(df_binaries=df, mass_measure=999.0, spin_measure=0.99, binsize_mass=0.01, binsize_spin=0.001)
    assert len(s) == 1
    assert np.isneginf(s["logL"].iloc[0])


def test_infer_ancestral_length_mismatch():
    with pytest.raises(ValueError, match="must be the same"):
        infer_ancestral_posterior_distribution(
            df_binaries=_prior_df(100),
            mass_posterior_samples=[1.0, 2.0],
            spin_posterior_samples=[0.1],
        )


def test_infer_ancestral_parallel_and_single_worker_lengths():
    df = _prior_df(1000)
    masses = [30.0, 31.0, 29.0, 35.0]
    spins = [0.2, 0.3, 0.4, 0.5]

    out1 = infer_ancestral_posterior_distribution(df, masses, spins, n_workers=1)
    out2 = infer_ancestral_posterior_distribution(df, masses, spins, n_workers=2)
    assert len(out1) == 4
    assert len(out2) == 4


def test_isdata_base_properties_and_safe_divide():
    prior = pd.DataFrame({"m_1": np.linspace(5, 65, 1000), "a_1": np.linspace(0, 1, 1000)})
    posterior = pd.DataFrame({"m_1": np.linspace(10, 60, 1000), "a_1": np.linspace(0.1, 0.9, 1000)})
    new_prior = pd.DataFrame({"m_1": np.linspace(5, 65, 1000), "a_1": np.linspace(0, 1, 1000)})

    data = ISData(prior_samples=prior, posterior_samples=posterior, new_prior_samples=new_prior)
    cols = data.common_columns
    assert "m_1" in cols and "a_1" in cols
    assert data.get_binsize("m_1") > 0
    assert data.get_binsize("a_1") > 0

    arr = data._safe_divide(np.array([1.0, 2.0]), np.array([1.0, 0.0]))
    assert arr[0] == 1.0
    assert arr[1] == 0.0


def test_isdata_get_binsize_unknown_column():
    prior = pd.DataFrame({"m_1": np.linspace(5, 65, 10), "a_1": np.linspace(0, 1, 10)})
    data = ISData(prior_samples=prior, posterior_samples=prior, new_prior_samples=prior)
    with pytest.raises(ValueError, match="Unknown column name"):
        data.get_binsize("q")


def test_isdata_empty_new_prior_branches():
    prior = _posterior_df(1000)
    posterior = _posterior_df(800)
    empty_new = pd.DataFrame(columns=["m_1", "a_1"])

    data = ISData(prior_samples=prior, posterior_samples=posterior, new_prior_samples=empty_new)
    assert data.get_bayes_factor() == 0.0
    sampled = data.sample_bayes_factor(n=5)
    assert sampled.samples == [0.0] * 5


def test_isdata_sampling_paths_1d_and_dd():
    prior = _posterior_df(3000)
    posterior = _posterior_df(2500)
    new_prior = _posterior_df(3000)

    data_1d = ISData(
        prior_samples=prior,
        posterior_samples=posterior,
        new_prior_samples=new_prior,
        assume_parameter_independence=True,
    )
    out_1d = data_1d.get_likelihood_samples(random_state=1)
    rw_1d = data_1d.get_reweighted_samples(random_state=2)
    assert len(out_1d) == len(posterior)
    assert len(rw_1d) == len(posterior)

    data_dd = ISData(
        prior_samples=prior,
        posterior_samples=posterior,
        new_prior_samples=new_prior,
        assume_parameter_independence=False,
    )
    out_dd = data_dd.get_likelihood_samples(random_state=1)
    rw_dd = data_dd.get_reweighted_samples(random_state=2)
    assert len(out_dd) == len(posterior)
    assert len(rw_dd) == len(posterior)


def test_candidate_prior_validation_and_methods():
    df1 = pd.DataFrame({"v_esc": [10.0, 100.0], "m_1": [30.0, 40.0]})
    df2 = pd.DataFrame({"v_esc": [20.0, 200.0], "m_2": [10.0, 20.0]})
    cp = CandidatePrior(df_bh1=df1, df_bh2=df2)

    c = cp.get_conditional_prior(v_esc=50.0, n_min=10)
    assert len(c) == 10
    assert "m_1" in c.columns and "m_2" in c.columns

    v_log = cp.get_host_escape_velocities(n_pts=5, log_scale=True)
    v_lin = cp.get_host_escape_velocities(n_pts=5, log_scale=False)
    assert len(v_log) >= 5
    assert len(v_lin) >= 5
    assert v_log[-1] >= 5000.0
    assert v_lin[-1] >= 5000.0


def test_candidate_prior_invalid_missing_v_esc():
    with pytest.raises(ValueError, match="must have 'v_esc'"):
        CandidatePrior(
            df_bh1=pd.DataFrame({"v_esc": [1.0], "m_1": [2.0]}),
            df_bh2=pd.DataFrame({"m_2": [3.0]}),
        )


def test_candidate_prior_invalid_overlap_columns():
    with pytest.raises(ValueError, match="overlapping columns"):
        CandidatePrior(
            df_bh1=pd.DataFrame({"v_esc": [1.0], "m_1": [2.0]}),
            df_bh2=pd.DataFrame({"v_esc": [2.0], "m_1": [3.0]}),
        )


def test_bayes_factor_curve_serial(monkeypatch):
    curve = BayesFactorCurve(n_bootstrapping=3, n_pts=3, log_scale=False)

    def fake_sample(*args, **kwargs):
        return BayesFactor(samples=[1.0, 2.0, 3.0])

    monkeypatch.setattr(BayesFactorCurve, "_sample_bayes_factor", fake_sample)

    prior = pd.DataFrame({"m_1": [30.0], "a_1": [0.5]})
    posterior = pd.DataFrame({"m_1": [32.0], "a_1": [0.6]})
    cp = CandidatePrior(
        df_bh1=pd.DataFrame({"v_esc": [10.0, 100.0], "m_1": [30.0, 40.0]}),
        df_bh2=pd.DataFrame({"v_esc": [10.0, 100.0], "m_2": [20.0, 25.0]}),
    )

    out = curve.get_bayes_factor_over_escape_velocity(prior, posterior, cp, n_workers=1)
    assert len(out) >= 3


def test_bayes_factor_curve_parallel(monkeypatch):
    curve = BayesFactorCurve(n_bootstrapping=2, n_pts=2, log_scale=False)

    monkeypatch.setattr(
        "archeo.bayesian.importance_sampling.bayes_factor_curve.multiprocess_run",
        lambda func, input_kwargs, n_processes: [BayesFactor(samples=[1.0, 1.0]) for _ in input_kwargs],
    )
    monkeypatch.setattr(BayesFactorCurve, "_sample_bayes_factor", lambda *args, **kwargs: BayesFactor(samples=[1.0]))

    prior = pd.DataFrame({"m_1": [30.0], "a_1": [0.5]})
    posterior = pd.DataFrame({"m_1": [32.0], "a_1": [0.6]})
    cp = CandidatePrior(
        df_bh1=pd.DataFrame({"v_esc": [10.0, 100.0], "m_1": [30.0, 40.0]}),
        df_bh2=pd.DataFrame({"v_esc": [10.0, 100.0], "m_2": [20.0, 25.0]}),
    )

    out = curve.get_bayes_factor_over_escape_velocity(prior, posterior, cp, n_workers=2)
    assert len(out) >= 2
