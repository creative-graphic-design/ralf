import numpy as np
import torch

from ralf.train.helpers.util import (
    argsort,
    batch_shuffle_index,
    box_cxcywh_to_xyxy,
    convert_xywh_to_ltrb,
    dict_of_list_to_list_of_dict,
    is_dict_of_list,
    is_list_of_dict,
    list_of_dict_to_dict_of_list,
    pad,
)


def test_box_conversion_roundtrip() -> None:
    boxes = np.array([[0.5, 0.5, 0.2, 0.2]])
    xyxy = box_cxcywh_to_xyxy(torch.tensor(boxes))
    ltrb = convert_xywh_to_ltrb(boxes[0])
    assert len(ltrb) == 4


def test_convert_xywh_to_ltrb() -> None:
    boxes = np.array([0.5, 0.5, 0.2, 0.2])
    ltrb = convert_xywh_to_ltrb(boxes)
    assert len(ltrb) == 4


def test_util_helpers() -> None:
    assert argsort([3, 1, 2]) == [1, 2, 0]

    data = {"a": [1, 2], "b": [3, 4]}
    assert is_dict_of_list(data)
    list_of_dicts = dict_of_list_to_list_of_dict(data)
    assert is_list_of_dict(list_of_dicts)
    back = list_of_dict_to_dict_of_list(list_of_dicts)
    assert back == data

    assert pad([True], 3) == [True, False, False]
    assert pad([1], 2) == [1, 0]
    assert pad([1.0], 2) == [1.0, 0.0]

    mask = torch.tensor([[True, False, True]])
    idx = batch_shuffle_index(1, 3, mask=mask)
    assert idx.shape == (1, 3)
