import numpy as np
import torch

from ralf.train.models.common_gan.design_seq import (
    box_cxcywh_to_xyxy,
    reorder,
)
from ralf.train.models.common_gan.layout_initializer import (
    box_xyxy_to_cxcywh_with_pad,
    preprocess_layout,
    random_init_layout,
)


def test_design_seq_reorder_and_box_conversion() -> None:
    boxes = np.array(
        [
            [0.5, 0.5, 0.2, 0.2],
            [0.5, 0.5, 0.2, 0.2],
            [0.5, 0.5, 0.2, 0.2],
            [0.5, 0.5, 0.2, 0.2],
        ]
    )
    cls = [0.0, 1.0, 2.0, 3.0]
    order = reorder(cls, boxes, o="cxcywh", max_elem=8)
    assert len(order) == len(cls)

    tensor_boxes = torch.tensor(boxes).float()
    xyxy = box_cxcywh_to_xyxy(tensor_boxes)
    assert xyxy.shape[-1] == 4


def test_layout_initializer_reorder_and_pad(small_batch) -> None:
    batch = {
        k: (v.clone() if isinstance(v, torch.Tensor) else v)
        for k, v in small_batch.items()
    }
    out = preprocess_layout(
        batch,
        max_elem=10,
        num_classes=5,
        use_reorder=True,
    )
    assert "layout" in out
    assert out["image_saliency"].shape[1] == 4

    padded = box_xyxy_to_cxcywh_with_pad(torch.tensor([0.1, 0.2, 0.3, 0.4, 0.0]))
    assert padded.numel() == 5

    init_layout = random_init_layout(
        batch_size=1,
        seq_length=4,
        coef=[1.0, 1.0, 1.0, 1.0],
        use_reorder=True,
        num_classes=4,
    )
    assert init_layout.shape[:3] == (1, 4, 2)
