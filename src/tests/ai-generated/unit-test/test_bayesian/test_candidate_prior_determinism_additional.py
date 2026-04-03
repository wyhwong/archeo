import pandas as pd

from archeo.bayesian.importance_sampling.bayes_factor_curve import CandidatePrior


def test_candidate_prior_conditional_prior_is_deterministic_with_same_seed():
    df1 = pd.DataFrame({"v_esc": [10.0, 20.0, 30.0], "m_1": [30.0, 40.0, 50.0]})
    df2 = pd.DataFrame({"v_esc": [10.0, 20.0, 30.0], "m_2": [15.0, 25.0, 35.0]})
    cp = CandidatePrior(df_bh1=df1, df_bh2=df2)

    out1 = cp.get_conditional_prior(v_esc=25.0, n_min=20, random_state=123)
    out2 = cp.get_conditional_prior(v_esc=25.0, n_min=20, random_state=123)

    assert out1.equals(out2)
    assert len(out1) == 20
    assert {"m_1", "m_2", "v_esc"}.issubset(set(out1.columns))


def test_candidate_prior_host_escape_velocities_no_forced_5000_if_already_above():
    df1 = pd.DataFrame({"v_esc": [100.0, 6000.0], "m_1": [30.0, 40.0]})
    df2 = pd.DataFrame({"v_esc": [200.0, 5500.0], "m_2": [20.0, 25.0]})
    cp = CandidatePrior(df_bh1=df1, df_bh2=df2)

    v_escs = cp.get_host_escape_velocities(n_pts=5, log_scale=False)

    assert v_escs[-1] == 6000.0  # max is already > 5000, so no forced append
