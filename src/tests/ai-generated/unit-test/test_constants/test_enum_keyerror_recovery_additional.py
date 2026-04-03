import sys
import types

from archeo.constants.enum import Fits


def test_fits_load_recovers_from_keyerror(monkeypatch):
    calls = {"loadfits": 0, "cleanup": 0}

    class FakeFitsInfo:
        desc = "fake"

    fake_surfinbh = types.SimpleNamespace(fits_collection={Fits.NRSUR3DQ8REMNANT.value: FakeFitsInfo()})

    def fake_loadfits(name):
        calls["loadfits"] += 1
        if calls["loadfits"] == 1:
            raise KeyError("corrupt metadata")
        return {"loaded": name}

    fake_surfinbh.LoadFits = fake_loadfits
    monkeypatch.setitem(sys.modules, "surfinBH", fake_surfinbh)
    monkeypatch.setattr(
        Fits,
        "clean_up_surfinbh_data",
        staticmethod(lambda: calls.__setitem__("cleanup", calls["cleanup"] + 1)),
    )

    out = Fits.NRSUR3DQ8REMNANT.load()
    assert out["loaded"] == Fits.NRSUR3DQ8REMNANT.value
    assert calls["cleanup"] == 1
    assert calls["loadfits"] == 2
