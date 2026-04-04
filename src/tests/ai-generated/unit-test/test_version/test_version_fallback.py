import importlib
import importlib.metadata as ilm


def test_version_fallback_when_package_not_installed(monkeypatch):
    def _raise(_name: str):
        raise ilm.PackageNotFoundError

    monkeypatch.setattr(ilm, "version", _raise)

    import archeo.version as v

    importlib.reload(v)
    assert v.__version__ == "0.0.0"
