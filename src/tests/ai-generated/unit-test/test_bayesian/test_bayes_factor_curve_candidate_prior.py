import pandas as pd
import pytest

from archeo.bayesian.importance_sampling.bayes_factor_curve import CandidatePrior


def test_candidate_prior_requires_v_esc():
    df1 = pd.DataFrame({"m_1": [1.0], "v_esc": [50.0]})
    df2 = pd.DataFrame({"m_2": [2.0]})  # missing v_esc
    with pytest.raises(ValueError, match="must have 'v_esc'"):
        CandidatePrior(df_bh1=df1, df_bh2=df2)


def test_candidate_prior_rejects_overlapping_columns():
    df1 = pd.DataFrame({"m_1": [1.0], "v_esc": [50.0]})
    df2 = pd.DataFrame({"m_1": [2.0], "v_esc": [70.0]})  # overlap not allowed
    with pytest.raises(ValueError, match="overlapping columns"):
        CandidatePrior(df_bh1=df1, df_bh2=df2)


def test_get_conditional_prior_empty_case():
    df1 = pd.DataFrame({"v_esc": [1000.0], "m_1": [30.0]})
    df2 = pd.DataFrame({"v_esc": [1000.0], "m_2": [20.0]})
    cp = CandidatePrior(df_bh1=df1, df_bh2=df2)

    out = cp.get_conditional_prior(v_esc=10.0)
    assert out.empty
