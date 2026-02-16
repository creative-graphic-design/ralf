import torch

from ralf.train.models.detr.main import SetCriterion
from ralf.train.models.detr.matcher import HungarianMatcher, build_matcher


class _Args:
    set_cost_class = 1
    set_cost_bbox = 1
    set_cost_giou = 1


def test_detr_matcher_and_criterion() -> None:
    outputs = {
        "pred_logits": torch.randn(1, 2, 4),
        "pred_boxes": torch.rand(1, 2, 4),
    }
    targets = [
        {"labels": torch.tensor([1]), "boxes": torch.rand(1, 4)},
    ]
    matcher = HungarianMatcher()
    indices = matcher(outputs, targets)
    assert len(indices) == 1

    matcher2 = build_matcher(_Args())
    criterion = SetCriterion(
        num_classes=3,
        matcher=matcher2,
        weight_dict={"loss_ce": 1.0, "loss_bbox": 1.0, "loss_giou": 1.0},
        eos_coef=0.1,
        losses=["labels", "boxes", "cardinality"],
    )
    losses = criterion(outputs, targets)
    assert "loss_ce" in losses


def test_detr_criterion_aux_outputs_and_get_loss_error() -> None:
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
    targets = [
        {"labels": torch.tensor([1]), "boxes": torch.rand(1, 4)},
    ]
    matcher = HungarianMatcher()
    criterion = SetCriterion(
        num_classes=3,
        matcher=matcher,
        weight_dict={"loss_ce": 1.0},
        eos_coef=0.1,
        losses=["labels"],
    )
    losses = criterion(outputs, targets)
    assert "loss_ce" in losses
    assert any(key.endswith("_0") for key in losses)

    indices = matcher(outputs, targets)
    num_boxes = torch.tensor([1.0])
    try:
        criterion.get_loss("unknown", outputs, targets, indices, num_boxes)
    except AssertionError as exc:
        assert "compute unknown loss" in str(exc)
