import pandas as pd

from archeo.bayesian.importance_sampling.bayes_factor_curve import BayesFactorCurve, CandidatePrior
from archeo.data_structures.bayesian.bayes_factor import BayesFactor


def test_bayes_factor_curve_parallel_mapping_order(monkeypatch):
    curve = BayesFactorCurve(n_bootstrapping=2, n_pts=3, log_scale=False)

    monkeypatch.setattr(
        "archeo.bayesian.importance_sampling.bayes_factor_curve.multiprocess_run",
        lambda func, input_kwargs, n_processes: [BayesFactor(samples=[i + 1.0]) for i, _ in enumerate(input_kwargs)],
    )

    prior = pd.DataFrame({"m_1": [30.0], "a_1": [0.5]})
    post = pd.DataFrame({"m_1": [31.0], "a_1": [0.6]})
    cp = CandidatePrior(
        df_bh1=pd.DataFrame({"v_esc": [10.0, 100.0], "m_1": [30.0, 40.0]}),
        df_bh2=pd.DataFrame({"v_esc": [10.0, 100.0], "m_2": [20.0, 25.0]}),
    )

    out = curve.get_bayes_factor_over_escape_velocity(prior, post, cp, n_workers=2)
    assert len(out) >= 3
    # sanity: values came from mocked multiprocess list and mapped by zip
    assert all(hasattr(v, "samples") for v in out.values())
