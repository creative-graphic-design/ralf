import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from ralf.hfds_builder.helpers import util as hfds_util
from ralf.hfds_builder.helpers.cgl import CGL_ID_NAME_MAPPING, CGL_JSON_FILES, read_cgl
from ralf.hfds_builder.helpers.global_variables import HEIGHT, WIDTH
from ralf.hfds_builder.helpers.image import draw, get_image_object, write_image_to_bytes
from ralf.hfds_builder.helpers.pku import PKU_CSV_FILES, read_pku


def test_helper_utils() -> None:
    assert hfds_util.with_key("foo", 1) == ("foo", 1)
    out = hfds_util.list_of_dict_to_dict_of_list([{"a": 1, "b": 2}, {"a": 3, "b": 4}])
    assert out == {"a": [1, 3], "b": [2, 4]}
    assert hfds_util.find_new_split("id1", {"train": ["id1"]}) == "train"
    assert hfds_util.find_new_split("id2", {"train": ["id1"]}) is None
    assert hfds_util.is_area_valid(0.2, 0.2, thresh=0.01)


def test_coordinates_and_clamp() -> None:
    coord = hfds_util.Coordinates.load_from_cgl_ltwh(
        ltwh=(10.0, 20.0, 30.0, 40.0), global_width=100, global_height=200
    )
    assert coord.has_valid_area()
    coord2 = hfds_util.Coordinates.load_from_pku_ltrb(
        box=(10.0, 20.0, 50.0, 60.0), global_width=100, global_height=200
    )
    assert coord2.has_valid_area()
    assert hfds_util.clamp_w_tol(1.0) == 1.0


def _write_cgl_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump(data, f)


def test_read_cgl_with_temp_data(tmp_path, dataset) -> None:
    ds_dict, _ = dataset
    sample = ds_dict["train"][0]
    label = sample["label"][0]
    if isinstance(label, int):
        cat_id = label + 1
    else:
        inv_map = {v: k for k, v in CGL_ID_NAME_MAPPING.items()}
        cat_id = inv_map[label]
    width = int(sample["image_width"])
    height = int(sample["image_height"])
    cx = float(sample["center_x"][0])
    cy = float(sample["center_y"][0])
    bw = float(sample["width"][0])
    bh = float(sample["height"][0])
    bbox = [
        (cx - bw / 2) * width,
        (cy - bh / 2) * height,
        bw * width,
        bh * height,
    ]
    base_json = {
        "images": [
            {
                "id": 1,
                "file_name": "1.png",
                "width": width,
                "height": height,
            }
        ],
        "annotations": [[{"image_id": 1, "category_id": cat_id, "bbox": bbox}]],
    }
    test_json = {
        "images": [
            {
                "id": 2,
                "file_name": "2.png",
                "width": width,
                "height": height,
            }
        ],
        "annotations": [],
    }
    annotation_dir = tmp_path / "annotation"
    _write_cgl_json(annotation_dir / CGL_JSON_FILES["train"], base_json)
    _write_cgl_json(annotation_dir / CGL_JSON_FILES["validation"], base_json)
    _write_cgl_json(annotation_dir / CGL_JSON_FILES["test"], test_json)

    samples = read_cgl(str(tmp_path), max_seq_length=10)
    assert len(samples) >= 2
    assert any(s.split == "test" and len(s.elements) == 0 for s in samples)


def test_read_pku_with_temp_data(tmp_path) -> None:
    pd = pytest.importorskip("pandas", exc_type=ImportError)
    annotation_dir = tmp_path / "annotation"
    annotation_dir.mkdir(parents=True, exist_ok=True)

    train_df = pd.DataFrame(
        {
            "poster_path": ["train/0001.png", "train/0001.png"],
            "cls_elem": [1, 2],
            "box_elem": ["(0, 0, 100, 100)", "(10, 10, 120, 140)"],
        }
    )
    train_df.to_csv(annotation_dir / PKU_CSV_FILES["train"], index=False)
    test_df = pd.DataFrame({"poster_path": ["0002.png"]})
    test_df.to_csv(annotation_dir / PKU_CSV_FILES["test"], index=False)

    samples = read_pku(str(tmp_path), max_seq_length=10)
    assert any(s.split == "train" for s in samples)
    assert any(s.split == "test" for s in samples)


def test_image_helpers() -> None:
    image = Image.fromarray(np.zeros((10, 10, 3), dtype=np.uint8))
    bytes_data = write_image_to_bytes(image)
    assert isinstance(bytes_data, bytes)
    obj = get_image_object(image, (8, 8))
    assert obj[0][0] == "bytes"
    assert obj[1][1] == "png"


def test_draw_image_overlay() -> None:
    image = Image.fromarray(np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8))
    example = {
        "image_width": WIDTH,
        "image_height": HEIGHT,
        "label": [0, 1],
        "center_x": [0.25, 0.75],
        "center_y": [0.25, 0.75],
        "width": [0.1, 0.2],
        "height": [0.1, 0.2],
    }
    output = draw(image, example)
    assert output.size == image.size
