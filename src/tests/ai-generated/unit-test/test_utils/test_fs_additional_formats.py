import pandas as pd

from archeo.utils.fs import load_dataframe


def test_load_dataframe_feather_branch(monkeypatch, tmp_path):
    p = tmp_path / "x.feather"
    p.write_text("dummy", encoding="utf-8")

    expected = pd.DataFrame({"a": [1]})
    monkeypatch.setattr(pd, "read_feather", lambda fp: expected)

    out = load_dataframe(str(p))
    assert out.equals(expected)


def test_load_dataframe_excel_branch(monkeypatch, tmp_path):
    p = tmp_path / "x.xlsx"
    p.write_text("dummy", encoding="utf-8")

    expected = pd.DataFrame({"a": [1], "b": [2]})
    monkeypatch.setattr(pd, "read_excel", lambda fp: expected)

    out = load_dataframe(str(p))
    assert out.equals(expected)
