import numpy as np
import torch

from ralf.train.models.common_gan.rec_loss import (
    HungarianMatcher,
    SetCriterion,
    linear_sum_assignment_with_inf,
)


def test_linear_sum_assignment_with_nan() -> None:
    cost = np.array([[np.nan, 1.0], [2.0, np.nan]])
    row, col = linear_sum_assignment_with_inf(cost)
    assert len(row) == len(col) == 2


def test_set_criterion_aux_outputs() -> None:
    matcher = HungarianMatcher()
    criterion = SetCriterion(
        num_classes=3,
        matcher=matcher,
        weight_dict={"loss_ce": 1.0, "loss_bbox": 1.0, "loss_giou": 1.0},
        coef=[1.0, 1.0, 1.0, 1.0],
        losses=["boxes"],
    )
    outputs = {
        "pred_logits": torch.randn(1, 2, 4),
        "pred_boxes": torch.rand(1, 2, 4),
        "aux_outputs": [
            {
                "pred_logits": torch.randn(1, 2, 4),
                "pred_boxes": torch.rand(1, 2, 4),
            }
        ],
    }
    targets = [{"labels": torch.tensor([1]), "boxes": torch.rand(1, 4)}]
    indices = matcher(outputs, targets)
    batch_idx, tgt_idx = criterion._get_tgt_permutation_idx(indices)
    assert batch_idx.numel() == tgt_idx.numel()

    losses = criterion(outputs, targets)
    assert any(key.endswith("_0") for key in losses)
