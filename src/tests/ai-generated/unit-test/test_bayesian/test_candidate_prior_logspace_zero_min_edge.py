import numpy as np
import pandas as pd

from archeo.bayesian.importance_sampling.bayes_factor_curve import CandidatePrior


def test_candidate_prior_host_escape_velocities_logscale_with_zero_min_is_finite():
    df1 = pd.DataFrame({"v_esc": [0.0, 10.0], "m_1": [30.0, 35.0]})
    df2 = pd.DataFrame({"v_esc": [0.0, 20.0], "m_2": [20.0, 22.0]})
    cp = CandidatePrior(df_bh1=df1, df_bh2=df2)

    v_escs = cp.get_host_escape_velocities(n_pts=4, log_scale=True)

    assert len(v_escs) >= 4
    assert np.isfinite(v_escs).all()
    assert v_escs[0] >= 0.1  # lower clamp used by implementation
    assert v_escs[-1] >= 5000.0  # forced append behavior when max < 5000
