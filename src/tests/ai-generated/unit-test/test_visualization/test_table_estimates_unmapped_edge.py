import pandas as pd

from archeo.visualization.estimation import table_estimates


def test_table_estimates_handles_all_unmapped_rows_gracefully(tmp_path):
    # k_f all NaN => filter_unmapped_samples makes per-parameter quantiles NaN
    df = pd.DataFrame(
        {
            "m_1": [30.0, 31.0],
            "m_2": [20.0, 21.0],
            "q": [1.5, 1.6],
            "m_f": [45.0, 46.0],
            "a_f": [0.6, 0.7],
            "k_f": [float("nan"), float("nan")],
            "chi_p": [0.2, 0.3],
            "chi_eff": [0.1, 0.2],
        }
    )

    out = table_estimates({"caseA": df}, output_dir=str(tmp_path), fmt="csv")
    assert len(out) == 1
    assert "Recovery Rate" in out.columns
    assert (tmp_path / "table_estimates.csv").exists()
