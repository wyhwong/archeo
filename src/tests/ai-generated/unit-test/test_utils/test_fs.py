import pandas as pd
import pytest

from archeo.utils.fs import check_and_create_dir, load_dataframe


def test_check_and_create_dir_idempotent(tmp_path):
    d = tmp_path / "new_dir"
    check_and_create_dir(str(d))
    assert d.exists()
    # second call should not fail
    check_and_create_dir(str(d))
    assert d.exists()


def test_load_dataframe_csv_and_json(tmp_path):
    df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})

    csv_path = tmp_path / "data.csv"
    json_path = tmp_path / "data.json"
    df.to_csv(csv_path, index=False)
    df.to_json(json_path, orient="records")

    loaded_csv = load_dataframe(str(csv_path))
    loaded_json = load_dataframe(str(json_path))

    assert list(loaded_csv.columns) == ["a", "b"]
    assert len(loaded_json) == 2


def test_load_dataframe_unsupported_extension(tmp_path):
    bad_path = tmp_path / "data.unsupported"
    bad_path.write_text("x")
    with pytest.raises(ValueError, match="Unsupported file format"):
        load_dataframe(str(bad_path))
