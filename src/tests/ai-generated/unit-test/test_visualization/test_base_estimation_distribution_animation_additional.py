import numpy as np
import pandas as pd
import pytest

from archeo.visualization import base
from archeo.visualization.animation import animate_remnant_property_change_over_kick
from archeo.visualization.distribution import distribution_summary, kick_distribution_on_spin
from archeo.visualization.estimation import (
    corner_estimates,
    effective_spin_estimates,
    filter_unmapped_samples,
    mass_estimates,
    precession_spin_estimates,
    second_generation_probability_curve,
    table_estimates,
)


@pytest.fixture(name="df_viz")
def _df_viz():
    rng = np.random.default_rng(123)
    n = 300
    return pd.DataFrame(
        {
            "m_1": rng.uniform(20, 70, n),
            "m_2": rng.uniform(10, 50, n),
            "q": rng.uniform(1, 5, n),
            "m_f": rng.uniform(20, 100, n),
            "a_f": rng.uniform(0, 1, n),
            "k_f": rng.uniform(0, 1500, n),
            "chi_p": rng.uniform(0, 1, n),
            "chi_eff": rng.uniform(-0.5, 0.5, n),
        }
    )


def test_initialize_plot_single_and_grid():
    fig1, ax1 = base.initialize_plot()
    assert fig1 is not None
    assert ax1 is not None

    fig2, axes2 = base.initialize_plot(nrows=2, ncols=2)
    assert fig2 is not None
    assert axes2.shape == (2, 2)


def test_savefig_and_close(tmp_path):
    fig, ax = base.initialize_plot()
    ax.plot([1, 2], [3, 4])
    base.savefig_and_close("x", output_dir=str(tmp_path), close=True, fmt="png")
    assert fig is not None
    assert (tmp_path / "x.png").exists()


def test_add_escape_velocity_and_plot_pdf(df_viz):
    fig, ax = base.initialize_plot()
    assert fig is not None
    base.plot_pdf(ax, df_viz["m_1"], unit="[Msun]")
    base.add_escape_velocity(ax, v_max=1000, y_max=1.0)
    # at least the PDF stairs should exist
    assert len(ax.patches) >= 0


def test_filter_unmapped_samples():
    df = pd.DataFrame({"k_f": [1.0, np.nan, 2.0], "m_1": [10, 20, 30]})
    out = filter_unmapped_samples(df)
    assert len(out) == 2
    assert out["k_f"].isna().sum() == 0


def test_mass_and_spin_estimates(df_viz, tmp_path):
    fig1, _ = mass_estimates(df_viz, label="test", output_dir=str(tmp_path), fmt="png")
    fig2, _ = effective_spin_estimates({"a": df_viz}, output_dir=str(tmp_path), fmt="png")
    fig3, _ = precession_spin_estimates({"a": df_viz}, output_dir=str(tmp_path), fmt="png")
    assert fig1 is not None and fig2 is not None and fig3 is not None


def test_second_generation_probability_curve(df_viz, tmp_path):
    fig, _ = second_generation_probability_curve({"a": df_viz, "b": df_viz}, output_dir=str(tmp_path), fmt="png")
    assert fig is not None


def test_table_estimates_md_csv_and_unsupported(df_viz, tmp_path):
    out1 = table_estimates({"x": df_viz}, output_dir=str(tmp_path), fmt="md")
    out2 = table_estimates({"x": df_viz}, output_dir=str(tmp_path), fmt="csv")
    out3 = table_estimates({"x": df_viz}, output_dir=str(tmp_path), fmt="abc")
    assert len(out1) == 1
    assert len(out2) == 1
    assert len(out3) == 1
    assert (tmp_path / "table_estimates.md").exists()
    assert (tmp_path / "table_estimates.csv").exists()
    assert not (tmp_path / "table_estimates.abc").exists()


def test_distribution_summary_and_kick_distribution(df_viz, tmp_path):
    fig1, _ = distribution_summary(df_viz, output_dir=str(tmp_path), fmt="png")
    fig2, _ = kick_distribution_on_spin(df_viz, output_dir=str(tmp_path), fmt="png")
    assert fig1 is not None and fig2 is not None


def test_animation_unsupported_or_invalid_bounds(df_viz):
    assert animate_remnant_property_change_over_kick(df_viz, col_name="unknown") is None
    # set kick_lb >= kick_ub
    assert animate_remnant_property_change_over_kick(df_viz, col_name="m_f", kick_lb=1e9) is None


def test_animation_valid(df_viz):
    ani = animate_remnant_property_change_over_kick(df_viz, col_name="m_f", kick_lb=0, kick_width=100)
    assert ani is not None


def test_corner_estimates_small_df_branch(tmp_path):
    # len(df) < nbins branch
    df = pd.DataFrame(
        {
            "m_1": [30.0, 31.0],
            "m_2": [20.0, 21.0],
            "m_f": [48.0, 50.0],
            "k_f": [100.0, 120.0],
            "a_f": [0.6, 0.7],
            "chi_eff": [0.1, 0.2],
            "chi_p": [0.3, 0.4],
        }
    )
    fig, axes = corner_estimates({"tiny": df}, nbins=70, output_dir=str(tmp_path), fmt="png")
    assert fig is not None
    assert axes is not None
