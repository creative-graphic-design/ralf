import torch

from ralf.train.models.common_gan.argmax import ArgMax, ArgMaxWithReorder
from ralf.train.models.common_gan.base_model import BaseGANGenerator
from ralf.train.models.common_gan.rec_loss import HungarianMatcher, SetCriterion


class DummyGAN(BaseGANGenerator):
    def __init__(self, features, auxilary_task: str):
        num_classes = features["label"].feature.num_classes + 1
        coef = [1.0] * num_classes
        super().__init__(
            d_model=8,
            apply_weight=False,
            use_reorder=False,
            use_reorder_for_random=False,
            features=features,
            max_seq_length=10,
            coef=coef,
            auxilary_task=auxilary_task,
        )

    def _encode_into_memory(self, inputs: dict) -> tuple[torch.Tensor, torch.Tensor]:
        return inputs["image"], inputs["layout"]

    def decode(self, img_feature: torch.Tensor, layout: torch.Tensor) -> dict:
        b, s = layout.shape[:2]
        return {
            "pred_logits": torch.zeros(b, s, self.d_label),
            "pred_boxes": torch.zeros(b, s, 4),
        }


def test_argmax_ops() -> None:
    x = torch.zeros(1, 2, 2, 4)
    x[0, 0, 0, 1] = 1
    out = ArgMax.apply(x.clone())
    assert out.shape == x.shape
    out2 = ArgMaxWithReorder.apply(x.clone())
    assert out2.shape == x.shape


def test_base_gan_preprocess_branches(small_batch, features) -> None:
    for task in ["uncond", "c", "cwh", "partial", "refinement"]:
        model = DummyGAN(features=features, auxilary_task=task)
        inputs, _targets = model.preprocess(small_batch)
        outputs = model(inputs)
        post = model.postprocess(outputs)
        assert "label" in post and "mask" in post

    model = DummyGAN(features=features, auxilary_task="uncond")
    sampled = model.sample(cond=small_batch)
    assert "label" in sampled


def test_common_gan_rec_loss() -> None:
    matcher = HungarianMatcher()
    outputs = {
        "pred_logits": torch.randn(1, 2, 4),
        "pred_boxes": torch.rand(1, 2, 4),
    }
    targets = [
        {"labels": torch.tensor([1]), "boxes": torch.rand(1, 4)},
    ]
    indices = matcher(outputs, targets)
    assert len(indices) == 1
    criterion = SetCriterion(
        num_classes=3,
        matcher=matcher,
        weight_dict={"loss_ce": 1.0, "loss_bbox": 1.0, "loss_giou": 1.0},
        coef=[1.0, 1.0, 1.0, 1.0],
        losses=["labels", "boxes"],
    )
    losses = criterion(outputs, targets)
    assert "loss_ce" in losses and "loss_bbox" in losses
