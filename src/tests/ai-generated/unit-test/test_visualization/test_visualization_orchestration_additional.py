import pandas as pd

import archeo.visualization as viz


def test_visualize_prior_distribution_orchestrates_all_subplots(monkeypatch, tmp_path):
    calls = {
        "check_and_create_dir": 0,
        "animate": [],
        "distribution_summary": 0,
        "kick_against_spin_cmap": 0,
        "kick_distribution_on_spin": 0,
    }

    monkeypatch.setattr(
        "archeo.visualization.check_and_create_dir",
        lambda output_dir: calls.__setitem__("check_and_create_dir", calls["check_and_create_dir"] + 1),
    )
    monkeypatch.setattr(
        "archeo.visualization.animate_remnant_property_change_over_kick",
        lambda df, col_name, output_dir=None: calls["animate"].append((col_name, output_dir)),
    )
    monkeypatch.setattr(
        "archeo.visualization.distribution_summary",
        lambda *args, **kwargs: calls.__setitem__("distribution_summary", calls["distribution_summary"] + 1),
    )
    monkeypatch.setattr(
        "archeo.visualization.kick_against_spin_cmap",
        lambda *args, **kwargs: calls.__setitem__("kick_against_spin_cmap", calls["kick_against_spin_cmap"] + 1),
    )
    monkeypatch.setattr(
        "archeo.visualization.kick_distribution_on_spin",
        lambda *args, **kwargs: calls.__setitem__("kick_distribution_on_spin", calls["kick_distribution_on_spin"] + 1),
    )

    df = pd.DataFrame({"k_f": [10.0], "m_f": [30.0], "a_f": [0.5], "q": [2.0], "chi_eff": [0.1], "chi_p": [0.2]})
    viz.visualize_prior_distribution(df, output_dir=str(tmp_path), fmt="png")

    assert calls["check_and_create_dir"] == 1
    assert calls["distribution_summary"] == 1
    assert calls["kick_against_spin_cmap"] == 1
    assert calls["kick_distribution_on_spin"] == 1
    assert [name for name, _ in calls["animate"]] == ["m_f", "a_f", "q", "chi_eff", "chi_p"]


def test_visualize_posterior_estimation_orchestrates_per_label_and_global(monkeypatch, tmp_path):
    calls = {
        "check_and_create_dir": [],
        "mass_estimates": 0,
        "corner_estimates": 0,
        "second_generation_probability_curve": 0,
        "effective_spin_estimates": 0,
        "precession_spin_estimates": 0,
        "table_estimates": 0,
    }

    monkeypatch.setattr("archeo.visualization.check_and_create_dir", lambda d: calls["check_and_create_dir"].append(d))
    monkeypatch.setattr(
        "archeo.visualization.mass_estimates",
        lambda *args, **kwargs: calls.__setitem__("mass_estimates", calls["mass_estimates"] + 1),
    )
    monkeypatch.setattr(
        "archeo.visualization.corner_estimates",
        lambda *args, **kwargs: calls.__setitem__("corner_estimates", calls["corner_estimates"] + 1),
    )
    monkeypatch.setattr(
        "archeo.visualization.second_generation_probability_curve",
        lambda *args, **kwargs: calls.__setitem__(
            "second_generation_probability_curve", calls["second_generation_probability_curve"] + 1
        ),
    )
    monkeypatch.setattr(
        "archeo.visualization.effective_spin_estimates",
        lambda *args, **kwargs: calls.__setitem__("effective_spin_estimates", calls["effective_spin_estimates"] + 1),
    )
    monkeypatch.setattr(
        "archeo.visualization.precession_spin_estimates",
        lambda *args, **kwargs: calls.__setitem__("precession_spin_estimates", calls["precession_spin_estimates"] + 1),
    )
    monkeypatch.setattr(
        "archeo.visualization.table_estimates",
        lambda *args, **kwargs: calls.__setitem__("table_estimates", calls["table_estimates"] + 1),
    )

    df = pd.DataFrame(
        {"k_f": [1.0], "m_1": [30.0], "m_2": [20.0], "m_f": [45.0], "a_f": [0.6], "chi_eff": [0.1], "chi_p": [0.2]}
    )
    viz.visualize_posterior_estimation({"A": df, "B": df}, output_dir=str(tmp_path), fmt="png")

    # per-label calls
    assert calls["mass_estimates"] == 2
    # corner/2g/eff/precess each called once per label + once globally
    assert calls["corner_estimates"] == 3
    assert calls["second_generation_probability_curve"] == 3
    assert calls["effective_spin_estimates"] == 3
    assert calls["precession_spin_estimates"] == 3
    assert calls["table_estimates"] == 1
