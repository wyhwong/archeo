import runpy


def test_main_invokes_simulation_cli(monkeypatch):
    called = {"n": 0}

    monkeypatch.setattr(
        "archeo.preset.cli.simulation_cli",
        lambda: called.__setitem__("n", called["n"] + 1),
    )

    runpy.run_module("archeo.__main__", run_name="__main__")
    assert called["n"] == 1
