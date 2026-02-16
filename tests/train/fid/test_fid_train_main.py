import shutil
import sys
from pathlib import Path

import datasets as ds

from ralf.train.fid import train as fid_train


def _write_split(dataset: ds.Dataset, out_dir: Path, split: str) -> None:
    if "transforms" in dataset.column_names:
        dataset = dataset.remove_columns("transforms")
    subset = dataset.select(range(min(2, len(dataset))))
    out_path = out_dir / f"{split}-00000-of-00001.parquet"
    subset.to_parquet(str(out_path))


def test_fid_train_main(tmp_path: Path, dataset, dataset_dir: str) -> None:
    ds_dict, _ = dataset
    root_dir = tmp_path / "dataset"
    cgl_dir = root_dir / "cgl"
    cgl_dir.mkdir(parents=True, exist_ok=True)

    for split in ["train", "val", "test", "with_no_annotation"]:
        _write_split(ds_dict[split], cgl_dir, split)

    shutil.copy(Path(dataset_dir) / "vocabulary.json", cgl_dir / "vocabulary.json")

    out_dir = tmp_path / "fid_out"
    argv = [
        "fid_train",
        "--dataset",
        "cgl",
        "--data-dir",
        str(root_dir),
        "--out-dir",
        str(out_dir),
        "--batch-size",
        "2",
        "--max_epoch",
        "1",
        "--iteration",
        "1",
    ]
    old_argv = sys.argv
    sys.argv = argv
    try:
        fid_train.main()
    finally:
        sys.argv = old_argv

    assert (out_dir / "model_best.pth.tar").exists()
