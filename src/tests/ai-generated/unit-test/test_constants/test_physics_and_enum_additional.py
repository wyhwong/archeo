import os
import sys
import types

import numpy as np
import pandas as pd
import pytest

from archeo.constants.enum import Fits
from archeo.constants.physics import PISN_LB, TypicalHostEscapeVelocity


def test_compute_p2g_happy_path():
    df = pd.DataFrame(
        {
            "k_f": [10.0, 40.0, 55.0, 700.0],
            "m_1": [30.0, 40.0, 30.0, 20.0],
            "m_2": [20.0, 50.0, PISN_LB + 1.0, 20.0],
        }
    )
    # GC v_esc=50 -> rows 0 and 1 only => 2/4 = 50%
    out = TypicalHostEscapeVelocity.GLOBULAR_CLUSTER.compute_p2g(df)
    assert np.isclose(out, 50.0)


def test_compute_p2g_empty_df():
    df = pd.DataFrame(columns=["k_f", "m_1", "m_2"])
    assert TypicalHostEscapeVelocity.MILKY_WAY.compute_p2g(df) == 0.0


def test_escape_velocity_lookup_maps():
    latex_map = TypicalHostEscapeVelocity.latex_to_values()
    short_map = TypicalHostEscapeVelocity.short_to_values()

    assert r"$v_{esc, GC}$" in latex_map
    assert "GC" in short_map
    assert short_map["MW"] == 600.0


def test_fits_cleanup_surfinbh_data(tmp_path, monkeypatch):
    fake_surfinbh = types.SimpleNamespace(__file__=str(tmp_path / "surfinBH.py"))
    data_dir = tmp_path / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    f1 = data_dir / "a.h5"
    f2 = data_dir / "b.h5"
    f1.write_text("x", encoding="utf-8")
    f2.write_text("y", encoding="utf-8")

    monkeypatch.setitem(sys.modules, "surfinBH", fake_surfinbh)
    Fits.clean_up_surfinbh_data()

    assert not f1.exists()
    assert not f2.exists()


def test_fits_load_recovers_from_oserror(monkeypatch):
    calls = {"n": 0}

    class FakeFitsInfo:
        desc = "fake fit"

    fake_surfinbh = types.SimpleNamespace(fits_collection={Fits.NRSUR3DQ8REMNANT.value: FakeFitsInfo()})

    def fake_loadfits(name):
        calls["n"] += 1
        if calls["n"] == 1:
            raise OSError("broken cache")
        return {"loaded": name}

    fake_surfinbh.LoadFits = fake_loadfits

    monkeypatch.setitem(sys.modules, "surfinBH", fake_surfinbh)
    monkeypatch.setattr(Fits, "clean_up_surfinbh_data", staticmethod(lambda: None))

    out = Fits.NRSUR3DQ8REMNANT.load()
    assert out["loaded"] == Fits.NRSUR3DQ8REMNANT.value
    assert calls["n"] == 2
