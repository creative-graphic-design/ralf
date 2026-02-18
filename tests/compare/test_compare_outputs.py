import dataclasses
import pickle

import numpy as np
import torch

from ralf.compare.compare_outputs import (
    _as_float,
    _is_nan,
    _to_numpy,
    compare_files,
    compare_trees,
    compare_values,
    main,
)


def test_compare_values_numeric_tolerance():
    mismatches: list = []
    assert compare_values(1.0, 1.00005, 1e-4, "root", mismatches)
    assert mismatches == []

    mismatches = []
    assert not compare_values(1.0, 1.1, 1e-4, "root", mismatches)
    assert any(m.reason == "numeric_mismatch" for m in mismatches)


def test_compare_files_yaml_tolerance(tmp_path):
    left_path = tmp_path / "left.yaml"
    right_path = tmp_path / "right.yaml"
    left_path.write_text("score: 1.0\n", encoding="utf-8")
    right_path.write_text("score: 1.00005\n", encoding="utf-8")

    mismatches: list = []
    assert compare_files(
        str(left_path), str(right_path), 1e-4, "score.yaml", mismatches
    )
    assert mismatches == []

    right_path.write_text("score: 1.2\n", encoding="utf-8")
    mismatches = []
    assert not compare_files(
        str(left_path), str(right_path), 1e-4, "score.yaml", mismatches
    )
    assert any(m.reason == "numeric_mismatch" for m in mismatches)


def test_compare_files_pickle_tensor(tmp_path):
    left_path = tmp_path / "left.pkl"
    right_path = tmp_path / "right.pkl"

    payload = {
        "arr": np.array([1.0, 2.0], dtype=np.float32),
        "tensor": torch.tensor([3.0, 4.0]),
    }
    with left_path.open("wb") as file_obj:
        pickle.dump(payload, file_obj)
    with right_path.open("wb") as file_obj:
        pickle.dump(payload, file_obj)

    mismatches: list = []
    assert compare_files(str(left_path), str(right_path), 1e-4, "data.pkl", mismatches)
    assert mismatches == []


def test_compare_trees_missing_file(tmp_path):
    left_root = tmp_path / "left"
    right_root = tmp_path / "right"
    left_root.mkdir()
    right_root.mkdir()

    (left_root / "only_left.txt").write_text("a", encoding="utf-8")
    (right_root / "only_right.txt").write_text("b", encoding="utf-8")

    report = compare_trees(str(left_root), str(right_root), 1e-4, ignore_exts=[])
    assert report["pass"] is False
    reasons = {m["reason"] for m in report["mismatches"]}
    assert "missing_files" in reasons
    assert "extra_files" in reasons


def test_compare_values_structures_and_nan():
    mismatches: list = []

    @dataclasses.dataclass
    class Item:
        value: int

    assert compare_values(Item(1), Item(1), 1e-4, "root", mismatches)
    assert mismatches == []

    assert compare_values(float("nan"), float("nan"), 1e-4, "root", mismatches)

    mismatches = []
    assert not compare_values({"a": 1}, {"b": 1}, 1e-4, "root", mismatches)
    assert any(m.reason == "key_mismatch" for m in mismatches)

    mismatches = []
    assert not compare_values([1, 2], [1], 1e-4, "root", mismatches)
    assert any(m.reason == "length_mismatch" for m in mismatches)

    mismatches = []
    assert not compare_values({1, 2}, {2, 3}, 1e-4, "root", mismatches)
    assert any(m.reason == "set_mismatch" for m in mismatches)


def test_compare_values_misc_branches():
    mismatches: list = []
    assert compare_values(1, 1, 1e-4, "root.pkl_path", mismatches)
    assert mismatches == []

    mismatches = []
    assert not compare_values(torch.tensor([1.0]), [1.0], 1e-4, "root", mismatches)
    assert any(m.reason == "type_mismatch" for m in mismatches)

    mismatches = []
    assert compare_values({1, 2}, {1, 2}, 1e-4, "root", mismatches)
    assert mismatches == []

    mismatches = []
    assert not compare_values("left", "right", 1e-4, "root", mismatches)
    assert any(m.reason == "value_mismatch" for m in mismatches)


def test_compare_array_mismatch_cases():
    mismatches: list = []
    left = np.array([1.0, 2.0], dtype=np.float32)
    right = np.array([[1.0, 2.0]], dtype=np.float32)
    assert not compare_values(left, right, 1e-4, "root", mismatches)
    assert any(m.reason == "shape_mismatch" for m in mismatches)

    mismatches = []
    right = np.array([1, 2], dtype=np.int64)
    assert not compare_values(left, right, 1e-4, "root", mismatches)
    assert any(m.reason == "dtype_mismatch" for m in mismatches)

    mismatches = []
    right = np.array([1.0, 2.2], dtype=np.float32)
    assert not compare_values(left, right, 1e-4, "root", mismatches)
    assert any(m.reason == "array_mismatch" for m in mismatches)

    mismatches = []
    empty = np.array([], dtype=np.float32)
    assert compare_values(empty, empty, 1e-4, "root", mismatches)
    assert mismatches == []


def test_internal_helpers_for_numpy_nan():
    assert _as_float(np.float32(1.5)) == 1.5
    assert _is_nan(object()) is False
    output = _to_numpy([1, 2, 3])
    assert isinstance(output, np.ndarray)


def test_compare_files_json_csv_npy_and_hash(tmp_path):
    left_json = tmp_path / "left.json"
    right_json = tmp_path / "right.json"
    left_json.write_text('{"score": 1.0}', encoding="utf-8")
    right_json.write_text('{"score": 1.00001}', encoding="utf-8")
    mismatches: list = []
    assert compare_files(str(left_json), str(right_json), 1e-4, "data.json", mismatches)

    left_csv = tmp_path / "left.csv"
    right_csv = tmp_path / "right.csv"
    left_csv.write_text("1.0,2.0\n", encoding="utf-8")
    right_csv.write_text("1.0,2.0\n", encoding="utf-8")
    mismatches = []
    assert compare_files(str(left_csv), str(right_csv), 1e-4, "data.csv", mismatches)

    left_csv.write_text("a,2.0\n", encoding="utf-8")
    right_csv.write_text("a,2.0\n", encoding="utf-8")
    mismatches = []
    assert compare_files(str(left_csv), str(right_csv), 1e-4, "data.csv", mismatches)

    left_npy = tmp_path / "left.npy"
    right_npy = tmp_path / "right.npy"
    np.save(left_npy, np.array([1.0, 2.0], dtype=np.float32))
    np.save(right_npy, np.array([1.0, 2.0], dtype=np.float32))
    mismatches = []
    assert compare_files(str(left_npy), str(right_npy), 1e-4, "data.npy", mismatches)

    left_bin = tmp_path / "left.bin"
    right_bin = tmp_path / "right.bin"
    left_bin.write_bytes(b"abc")
    right_bin.write_bytes(b"abcd")
    mismatches = []
    assert not compare_files(
        str(left_bin), str(right_bin), 1e-4, "data.bin", mismatches
    )
    assert any(m.reason == "hash_mismatch" for m in mismatches)

    right_bin.write_bytes(b"abc")
    mismatches = []
    assert compare_files(str(left_bin), str(right_bin), 1e-4, "data.bin", mismatches)


def test_compare_files_pickle_fallback_and_errors(tmp_path):
    left_pt = tmp_path / "left.pt"
    right_pt = tmp_path / "right.pt"
    payload = {"tensor": torch.tensor([1.0, 2.0])}
    torch.save(payload, left_pt)
    torch.save(payload, right_pt)
    mismatches: list = []
    assert compare_files(str(left_pt), str(right_pt), 1e-4, "data.pt", mismatches)

    bad_yaml = tmp_path / "bad.yaml"
    bad_yaml.write_text("a: [1,2", encoding="utf-8")
    mismatches = []
    assert not compare_files(str(bad_yaml), str(bad_yaml), 1e-4, "bad.yaml", mismatches)
    assert any(m.reason == "load_error" for m in mismatches)


def test_compare_trees_ignore_ext(tmp_path):
    left_root = tmp_path / "left"
    right_root = tmp_path / "right"
    left_root.mkdir()
    right_root.mkdir()

    (left_root / "only.txt").write_text("a", encoding="utf-8")
    (right_root / "only.txt").write_text("b", encoding="utf-8")

    report = compare_trees(str(left_root), str(right_root), 1e-4, ignore_exts=[".txt"])
    assert report["pass"] is True


def test_compare_trees_with_common_file(tmp_path):
    left_root = tmp_path / "left"
    right_root = tmp_path / "right"
    left_root.mkdir()
    right_root.mkdir()

    (left_root / "shared.json").write_text('{"value": 1}', encoding="utf-8")
    (right_root / "shared.json").write_text('{"value": 1}', encoding="utf-8")

    report = compare_trees(str(left_root), str(right_root), 1e-4, ignore_exts=[])
    assert report["pass"] is True


def test_compare_main_generates_report(tmp_path):
    left_root = tmp_path / "left"
    right_root = tmp_path / "right"
    left_root.mkdir()
    right_root.mkdir()

    (left_root / "data.json").write_text('{"score": 1.0}', encoding="utf-8")
    (right_root / "data.json").write_text('{"score": 1.0}', encoding="utf-8")

    report_path = tmp_path / "report.json"
    exit_code = main(
        [
            "--original",
            str(left_root),
            "--migrated",
            str(right_root),
            "--report-path",
            str(report_path),
        ]
    )
    assert exit_code == 0
    assert report_path.exists()
