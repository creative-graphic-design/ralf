import json
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from ralf.hfds_builder import dump_dataset
from ralf.hfds_builder.dump_dataset import _make_record, _unpack_elements
from ralf.hfds_builder.helpers.global_variables import (
    HEIGHT_RESIZE_IMAGE,
    WIDTH_RESIZE_IMAGE,
)
from ralf.hfds_builder.helpers.util import Coordinates, Element, Sample


def _write_gray_image(path: Path, image: Image.Image) -> None:
    gray = image.convert("L")
    path.parent.mkdir(parents=True, exist_ok=True)
    gray.save(path)


def _make_minimal_cgl_root(root: Path) -> str:
    root.mkdir(parents=True, exist_ok=True)
    annotation_dir = root / "annotation"
    annotation_dir.mkdir(parents=True, exist_ok=True)

    train_id = Path("data_splits/splits/cgl/train.txt").read_text().splitlines()[0]
    image_size = (8, 8)
    image = Image.fromarray(np.zeros((*image_size, 3), dtype=np.uint8))

    train_json = {
        "images": [
            {
                "id": 1,
                "file_name": f"{train_id}.png",
                "width": image_size[0],
                "height": image_size[1],
            }
        ],
        "annotations": [
            [
                {
                    "category_id": 1,
                    "image_id": 1,
                    "bbox": [0, 0, 2, 2],
                }
            ]
        ],
    }
    (annotation_dir / "layout_train_6w_fixed_v2.json").write_text(
        json.dumps(train_json)
    )
    (annotation_dir / "layout_test_6w_fixed_v2.json").write_text(
        json.dumps({"images": [], "annotations": []})
    )
    (annotation_dir / "yinhe.json").write_text(
        json.dumps({"images": [], "annotations": []})
    )

    input_dir = root / "image" / "train" / "input"
    saliency_dir = root / "image" / "train" / "saliency"
    saliency_sub_dir = root / "image" / "train" / "saliency_sub"
    input_dir.mkdir(parents=True, exist_ok=True)
    saliency_dir.mkdir(parents=True, exist_ok=True)
    saliency_sub_dir.mkdir(parents=True, exist_ok=True)

    image.save(input_dir / f"{train_id}.png")
    _write_gray_image(saliency_dir / f"{train_id}.png", image)
    _write_gray_image(saliency_sub_dir / f"{train_id}.png", image)

    return str(root)


def test_make_record_and_unpack_elements(tmp_path, dataset) -> None:
    ds_dict, _ = dataset
    sample_raw = ds_dict["train"][0]
    image_raw = sample_raw["image"]
    if isinstance(image_raw, torch.Tensor):
        image = Image.fromarray(
            (image_raw.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
        )
    else:
        image = image_raw
    sample_id = str(sample_raw["id"])

    input_dir = tmp_path / "image" / "train" / "input"
    saliency_dir = tmp_path / "image" / "train" / "saliency"
    saliency_sub_dir = tmp_path / "image" / "train" / "saliency_sub"
    input_dir.mkdir(parents=True, exist_ok=True)
    saliency_dir.mkdir(parents=True, exist_ok=True)
    saliency_sub_dir.mkdir(parents=True, exist_ok=True)

    input_path = input_dir / f"{sample_id}.png"
    image.save(input_path)
    _write_gray_image(saliency_dir / f"{sample_id}.png", image)
    _write_gray_image(saliency_sub_dir / f"{sample_id}.png", image)

    cx = float(sample_raw["center_x"][0])
    cy = float(sample_raw["center_y"][0])
    w = float(sample_raw["width"][0])
    h = float(sample_raw["height"][0])
    coord = Coordinates(
        left=cx - w / 2,
        center_x=cx,
        right=cx + w / 2,
        width=w,
        top=cy - h / 2,
        center_y=cy,
        bottom=cy + h / 2,
        height=h,
    )
    elements = [Element(label=sample_raw["label"][0], coordinates=coord)]
    sample = Sample(
        id=sample_id,
        identifier=f"train/{sample_id}.png",
        image_width=int(sample_raw["image_width"]),
        image_height=int(sample_raw["image_height"]),
        elements=elements,
        split="train",
    )

    unpacked = _unpack_elements(elements)
    assert unpacked["label"] == [sample_raw["label"][0]]
    record = _make_record(
        sample,
        dataset_root=str(tmp_path),
        image_size=(WIDTH_RESIZE_IMAGE, HEIGHT_RESIZE_IMAGE),
    )
    assert record["id"] == sample_id
    assert record["image"].size == (WIDTH_RESIZE_IMAGE, HEIGHT_RESIZE_IMAGE)
    assert record["saliency"].size == (WIDTH_RESIZE_IMAGE, HEIGHT_RESIZE_IMAGE)
    assert record["label"][0] == sample_raw["label"][0]


def test_dump_dataset_main(tmp_path) -> None:
    dataset_root = _make_minimal_cgl_root(tmp_path / "cgl_root")
    output_dir = tmp_path / "out"
    argv = [
        "dump_dataset",
        "--dataset_type",
        "cgl",
        "--dataset_root",
        dataset_root,
        "--output_dir",
        str(output_dir),
        "--num-shards",
        "1",
        "--log_level",
        "debug",
    ]
    old_argv = sys.argv
    sys.argv = argv
    try:
        dump_dataset.main()
    finally:
        sys.argv = old_argv

    assert (output_dir / "vocabulary.json").exists()
    assert any(output_dir.glob("train-*.parquet"))
