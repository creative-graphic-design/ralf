import torch

from ralf.train.helpers.relationships import (
    RelLoc,
    RelSize,
    detect_loc_relation_between_element_and_canvas,
    detect_loc_relation_between_elements,
    detect_size_relation,
)


def test_get_rel_loc_simple() -> None:
    a = torch.tensor([0.2, 0.2, 0.2, 0.2])
    b = torch.tensor([0.8, 0.8, 0.2, 0.2])
    rel = detect_loc_relation_between_elements(a.tolist(), b.tolist())
    assert rel in [RelLoc.LEFT, RelLoc.RIGHT, RelLoc.TOP, RelLoc.BOTTOM, RelLoc.CENTER]


def test_get_rel_size_simple() -> None:
    a = torch.tensor([0.2, 0.2, 0.1, 0.1])
    b = torch.tensor([0.2, 0.2, 0.4, 0.4])
    rel = detect_size_relation(a.tolist(), b.tolist())
    assert rel in [RelSize.SMALLER, RelSize.LARGER, RelSize.EQUAL]


def test_rel_size_branches() -> None:
    base = [0.5, 0.5, 0.2, 0.2]
    larger = [0.5, 0.5, 0.4, 0.4]
    smaller = [0.5, 0.5, 0.1, 0.1]
    almost_equal = [0.5, 0.5, 0.21, 0.19]
    assert detect_size_relation(base, larger) == RelSize.LARGER
    assert detect_size_relation(base, smaller) == RelSize.SMALLER
    assert detect_size_relation(base, almost_equal) == RelSize.EQUAL


def test_rel_loc_branches() -> None:
    center = [0.5, 0.5, 0.2, 0.2]
    top = [0.5, 0.1, 0.2, 0.2]
    bottom = [0.5, 0.9, 0.2, 0.2]
    left = [0.1, 0.5, 0.2, 0.2]
    right = [0.9, 0.5, 0.2, 0.2]
    assert detect_loc_relation_between_elements(center, top) == RelLoc.TOP
    assert detect_loc_relation_between_elements(center, bottom) == RelLoc.BOTTOM
    assert detect_loc_relation_between_elements(center, left) == RelLoc.LEFT
    assert detect_loc_relation_between_elements(center, right) == RelLoc.RIGHT
    assert detect_loc_relation_between_elements(center, center) == RelLoc.CENTER


def test_rel_loc_canvas_branches() -> None:
    assert (
        detect_loc_relation_between_element_and_canvas([0.5, 0.1, 0.2, 0.2])
        == RelLoc.TOP
    )
    assert (
        detect_loc_relation_between_element_and_canvas([0.5, 0.5, 0.2, 0.2])
        == RelLoc.CENTER
    )
    assert (
        detect_loc_relation_between_element_and_canvas([0.5, 0.9, 0.2, 0.2])
        == RelLoc.BOTTOM
    )
