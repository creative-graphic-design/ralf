import json
import os
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

from ralf.hfds_builder import inpainting
from ralf.hfds_builder.helpers.util import Coordinates, Element, Sample
from ralf.hfds_builder.inpainting import _dilate_mask, _get_mask


def _make_dummy_lama(path: Path) -> None:
    class Dummy(torch.nn.Module):
        def forward(self, image, mask):
            return image

    model = Dummy()
    example = (
        torch.zeros(1, 3, 8, 8),
        torch.zeros(1, 3, 8, 8),
    )
    scripted = torch.jit.trace(model, example)
    scripted.save(str(path))


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

    original_dir = root / "image" / "train" / "original"
    original_dir.mkdir(parents=True, exist_ok=True)
    image.save(original_dir / f"{train_id}.png")

    return str(root)


def test_dilate_mask_basic() -> None:
    mask = np.zeros((10, 10, 1), dtype=np.float32)
    mask[4:6, 4:6, 0] = 1.0
    dilated = _dilate_mask(mask, kernel_size=3, iterations=1)
    assert dilated.dtype == np.float32
    assert dilated.shape[:2] == mask.shape[:2]
    assert dilated.max() >= mask.max()


def test_get_mask_from_sample() -> None:
    pytest.importorskip("cv2", exc_type=ImportError)
    coord = Coordinates(
        left=0.1,
        center_x=0.2,
        right=0.3,
        width=0.2,
        top=0.1,
        center_y=0.2,
        bottom=0.3,
        height=0.2,
    )
    sample = Sample(
        id="1",
        identifier="train/1.png",
        image_width=100,
        image_height=120,
        elements=[Element(label="logo", coordinates=coord)],
        split="train",
    )
    mask = _get_mask(sample)
    assert isinstance(mask, Image.Image)
    assert mask.size == (sample.image_width, sample.image_height)


def test_inpainting_main(tmp_path) -> None:
    pytest.importorskip("cv2", exc_type=ImportError)
    dataset_root = _make_minimal_cgl_root(tmp_path / "cgl_root")
    dummy_model = tmp_path / "dummy_lama.pt"
    _make_dummy_lama(dummy_model)
    os.environ["LAMA_MODEL"] = str(dummy_model)

    argv = [
        "inpainting",
        "--dataset_type",
        "cgl",
        "--dataset_root",
        dataset_root,
        "--log_level",
        "debug",
    ]
    old_argv = sys.argv
    sys.argv = argv
    try:
        inpainting.main()
    finally:
        sys.argv = old_argv
        del os.environ["LAMA_MODEL"]

    train_id = Path("data_splits/splits/cgl/train.txt").read_text().splitlines()[0]
    output_path = Path(dataset_root) / "image" / "train" / "input" / f"{train_id}.png"
    assert output_path.exists()
