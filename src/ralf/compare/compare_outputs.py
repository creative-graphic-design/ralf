import argparse
import csv
import dataclasses
import hashlib
import json
import math
import os
import pickle
from dataclasses import dataclass
from typing import Any, Iterable

import numpy as np
import torch
import yaml

IGNORED_KEY_SUFFIXES = (".pkl_path", ".test_cfg.result_dir")


@dataclass(frozen=True)
class Mismatch:
    path: str
    reason: str
    detail: str


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float, np.number))


def _as_float(value: Any) -> float:
    if isinstance(value, np.number):
        return float(value)
    return float(value)


def _is_nan(value: Any) -> bool:
    try:
        return math.isnan(_as_float(value))
    except Exception:
        return False


def _to_numpy(value: Any) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    if isinstance(value, np.ndarray):
        return value
    return np.asarray(value)


def _compare_arrays(
    left: np.ndarray,
    right: np.ndarray,
    tol: float,
    path: str,
    mismatches: list[Mismatch],
) -> bool:
    if left.shape != right.shape:
        mismatches.append(
            Mismatch(
                path=path,
                reason="shape_mismatch",
                detail=f"{left.shape} != {right.shape}",
            )
        )
        return False
    if left.dtype != right.dtype:
        mismatches.append(
            Mismatch(
                path=path,
                reason="dtype_mismatch",
                detail=f"{left.dtype} != {right.dtype}",
            )
        )
        return False
    if left.size == 0 and right.size == 0:
        return True
    close = np.isclose(left, right, atol=tol, rtol=0.0, equal_nan=True)
    if bool(np.all(close)):
        return True
    diff = np.abs(left.astype(np.float64) - right.astype(np.float64))
    max_diff = float(np.nanmax(diff))
    mismatches.append(
        Mismatch(
            path=path,
            reason="array_mismatch",
            detail=f"max_abs_diff={max_diff}",
        )
    )
    return False


def compare_values(
    left: Any,
    right: Any,
    tol: float,
    path: str,
    mismatches: list[Mismatch],
) -> bool:
    if any(path.endswith(suffix) for suffix in IGNORED_KEY_SUFFIXES):
        return True
    if left is right:
        return True

    if dataclasses.is_dataclass(left):
        left = dataclasses.asdict(left)
    if dataclasses.is_dataclass(right):
        right = dataclasses.asdict(right)

    if isinstance(left, torch.Tensor) or isinstance(left, np.ndarray):
        left_arr = _to_numpy(left)
        if not isinstance(right, (torch.Tensor, np.ndarray)):
            mismatches.append(
                Mismatch(
                    path=path,
                    reason="type_mismatch",
                    detail=f"{type(left).__name__} != {type(right).__name__}",
                )
            )
            return False
        right_arr = _to_numpy(right)
        return _compare_arrays(left_arr, right_arr, tol, path, mismatches)

    if _is_number(left) and _is_number(right):
        if _is_nan(left) and _is_nan(right):
            return True
        diff = abs(_as_float(left) - _as_float(right))
        if diff <= tol:
            return True
        mismatches.append(
            Mismatch(
                path=path,
                reason="numeric_mismatch",
                detail=f"{left} != {right} (diff={diff})",
            )
        )
        return False

    if isinstance(left, dict) and isinstance(right, dict):
        left_keys = set(left.keys())
        right_keys = set(right.keys())
        ok = True
        if left_keys != right_keys:
            missing = sorted(right_keys - left_keys)
            extra = sorted(left_keys - right_keys)
            mismatches.append(
                Mismatch(
                    path=path,
                    reason="key_mismatch",
                    detail=f"missing={missing} extra={extra}",
                )
            )
            ok = False
        for key in sorted(left_keys & right_keys):
            child_path = f"{path}.{key}"
            ok = (
                compare_values(left[key], right[key], tol, child_path, mismatches)
                and ok
            )
        return ok

    if isinstance(left, (list, tuple)) and isinstance(right, (list, tuple)):
        if len(left) != len(right):
            mismatches.append(
                Mismatch(
                    path=path,
                    reason="length_mismatch",
                    detail=f"{len(left)} != {len(right)}",
                )
            )
            return False
        ok = True
        for idx, (l_item, r_item) in enumerate(zip(left, right)):
            child_path = f"{path}[{idx}]"
            ok = compare_values(l_item, r_item, tol, child_path, mismatches) and ok
        return ok

    if isinstance(left, (set, frozenset)) and isinstance(right, (set, frozenset)):
        if left == right:
            return True
        mismatches.append(
            Mismatch(
                path=path,
                reason="set_mismatch",
                detail=f"{sorted(left)} != {sorted(right)}",
            )
        )
        return False

    if left == right:
        return True

    mismatches.append(
        Mismatch(
            path=path,
            reason="value_mismatch",
            detail=f"{left!r} != {right!r}",
        )
    )
    return False


def _load_yaml(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as file_obj:
        return yaml.safe_load(file_obj)


def _load_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as file_obj:
        return json.load(file_obj)


def _load_pickle(path: str) -> Any:
    with open(path, "rb") as file_obj:
        try:
            return pickle.load(file_obj)
        except Exception:
            file_obj.seek(0)
            return torch.load(file_obj, map_location="cpu")


def _load_csv(path: str) -> list[list[Any]]:
    rows: list[list[Any]] = []
    with open(path, newline="", encoding="utf-8") as file_obj:
        reader = csv.reader(file_obj)
        for row in reader:
            parsed_row: list[Any] = []
            for value in row:
                try:
                    parsed_row.append(float(value))
                except ValueError:
                    parsed_row.append(value)
            rows.append(parsed_row)
    return rows


def _hash_file(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as file_obj:
        for chunk in iter(lambda: file_obj.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def compare_files(
    left_path: str,
    right_path: str,
    tol: float,
    rel_path: str,
    mismatches: list[Mismatch],
) -> bool:
    ext = os.path.splitext(left_path)[1].lower()
    try:
        if ext in {".yaml", ".yml"}:
            left = _load_yaml(left_path)
            right = _load_yaml(right_path)
            return compare_values(left, right, tol, rel_path, mismatches)
        if ext == ".json":
            left = _load_json(left_path)
            right = _load_json(right_path)
            return compare_values(left, right, tol, rel_path, mismatches)
        if ext in {".pkl", ".pickle", ".pt", ".pth"}:
            left = _load_pickle(left_path)
            right = _load_pickle(right_path)
            return compare_values(left, right, tol, rel_path, mismatches)
        if ext == ".npy":
            left = np.load(left_path, allow_pickle=True)
            right = np.load(right_path, allow_pickle=True)
            return compare_values(left, right, tol, rel_path, mismatches)
        if ext == ".csv":
            left = _load_csv(left_path)
            right = _load_csv(right_path)
            return compare_values(left, right, tol, rel_path, mismatches)
    except Exception as exc:
        mismatches.append(
            Mismatch(
                path=rel_path,
                reason="load_error",
                detail=str(exc),
            )
        )
        return False

    left_hash = _hash_file(left_path)
    right_hash = _hash_file(right_path)
    if left_hash == right_hash:
        return True
    mismatches.append(
        Mismatch(
            path=rel_path,
            reason="hash_mismatch",
            detail=f"{left_hash} != {right_hash}",
        )
    )
    return False


def _collect_files(root: str, ignore_exts: Iterable[str]) -> set[str]:
    ignore_set = {ext.lower() for ext in ignore_exts}
    files: set[str] = set()
    for dirpath, _, filenames in os.walk(root):
        for filename in filenames:
            ext = os.path.splitext(filename)[1].lower()
            if ext in ignore_set:
                continue
            full_path = os.path.join(dirpath, filename)
            rel_path = os.path.relpath(full_path, root)
            files.add(rel_path)
    return files


def compare_trees(
    left_root: str,
    right_root: str,
    tol: float,
    ignore_exts: Iterable[str],
) -> dict[str, Any]:
    mismatches: list[Mismatch] = []
    left_files = _collect_files(left_root, ignore_exts)
    right_files = _collect_files(right_root, ignore_exts)
    ok = True

    missing = sorted(right_files - left_files)
    extra = sorted(left_files - right_files)
    if missing:
        mismatches.append(
            Mismatch(
                path="",
                reason="missing_files",
                detail=",".join(missing),
            )
        )
        ok = False
    if extra:
        mismatches.append(
            Mismatch(
                path="",
                reason="extra_files",
                detail=",".join(extra),
            )
        )
        ok = False

    for rel_path in sorted(left_files & right_files):
        left_path = os.path.join(left_root, rel_path)
        right_path = os.path.join(right_root, rel_path)
        ok = compare_files(left_path, right_path, tol, rel_path, mismatches) and ok

    return {
        "pass": ok,
        "tolerance": tol,
        "left_root": left_root,
        "right_root": right_root,
        "file_count_left": len(left_files),
        "file_count_right": len(right_files),
        "mismatches": [dataclasses.asdict(m) for m in mismatches],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--original", required=True, help="Original output root")
    parser.add_argument("--migrated", required=True, help="Migrated output root")
    parser.add_argument("--tolerance", type=float, default=1e-4)
    parser.add_argument("--report-path", default="comparison_report.json")
    parser.add_argument(
        "--ignore-ext",
        action="append",
        default=[],
        help="File extensions to ignore (e.g. .log). Can be provided multiple times.",
    )
    args = parser.parse_args(argv)

    report = compare_trees(
        left_root=args.original,
        right_root=args.migrated,
        tol=args.tolerance,
        ignore_exts=args.ignore_ext,
    )
    os.makedirs(os.path.dirname(args.report_path) or ".", exist_ok=True)
    with open(args.report_path, "w", encoding="utf-8") as file_obj:
        json.dump(report, file_obj, indent=2, sort_keys=True)

    return 0 if report["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
