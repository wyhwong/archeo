import pandas as pd

from archeo.visualization.estimation import filter_unmapped_samples, table_estimates


def test_filter_unmapped_samples():
    df = pd.DataFrame({"k_f": [10.0, None, 30.0], "m_1": [1, 2, 3]})
    out = filter_unmapped_samples(df)
    assert len(out) == 2
    assert out["k_f"].isna().sum() == 0


def test_table_estimates_basic(tmp_path):
    df = pd.DataFrame(
        {
            "m_1": [40, 50, 60],
            "m_2": [20, 25, 30],
            "q": [2.0, 2.0, 2.0],
            "m_f": [55, 68, 80],
            "a_f": [0.3, 0.5, 0.7],
            "k_f": [100, 200, 300],
            "chi_p": [0.1, 0.2, 0.3],
            "chi_eff": [0.05, 0.1, 0.15],
        }
    )

    out = table_estimates({"test": df}, output_dir=str(tmp_path), fmt="csv")
    assert "Recovery Rate" in out.columns
    assert len(out) == 1
    assert (tmp_path / "table_estimates.csv").exists()
