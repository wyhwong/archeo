import numpy as np
import pandas as pd

from archeo.constants.physics import TypicalHostEscapeVelocity


def test_compute_p2g_happy_path():
    df = pd.DataFrame(
        {
            "k_f": [10, 40, 55, 700],
            "m_1": [30, 40, 30, 20],
            "m_2": [20, 50, 70, 20],  # third row fails m_2 <= PISN_LB
        }
    )
    # For GC (50): rows 0,1 satisfy k_f<=50 and m_1,m_2 <= PISN_LB => 2/4 = 50%
    p2g = TypicalHostEscapeVelocity.GLOBULAR_CLUSTER.compute_p2g(df)
    assert np.isclose(p2g, 50.0)


def test_compute_p2g_empty_df():
    df = pd.DataFrame(columns=["k_f", "m_1", "m_2"])
    assert TypicalHostEscapeVelocity.MILKY_WAY.compute_p2g(df) == 0.0
