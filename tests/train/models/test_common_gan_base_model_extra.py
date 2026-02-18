import torch
import torch.nn as nn

from ralf.train.models.common_gan.base_model import BaseGANGenerator


class DummyGAN(BaseGANGenerator):
    def __init__(self, features, max_seq_length: int = 10) -> None:
        coef = [1.0] * (features["label"].feature.num_classes + 1)
        super().__init__(
            d_model=32,
            apply_weight=True,
            use_reorder=False,
            use_reorder_for_random=False,
            features=features,
            max_seq_length=max_seq_length,
            coef=coef,
            auxilary_task="cwh",
        )
        self.adv_weight = 1.0

    def _encode_into_memory(self, inputs):  # type: ignore
        bsz = inputs["image"].size(0)
        return torch.zeros(bsz, 1), torch.zeros(bsz, 1)

    def decode(self, img_feature, layout_feature):  # type: ignore
        bsz = img_feature.size(0)
        pred_logits = torch.randn(bsz, self.max_seq_length, self.d_label)
        pred_boxes = torch.rand(bsz, self.max_seq_length, 4)
        return {"pred_logits": pred_logits, "pred_boxes": pred_boxes}


class DummyDiscriminator(nn.Module):
    def forward(self, image, layout):  # type: ignore
        return torch.zeros(image.size(0), 1, device=image.device)


def test_common_gan_train_loss_and_dis_loss(small_batch, features) -> None:
    model = DummyGAN(features=features, max_seq_length=10)
    model.eval()
    inputs, targets = model.preprocess(small_batch)
    outputs, losses = model.train_loss(
        inputs, targets, discriminator=DummyDiscriminator()
    )
    assert "adv_fake" in losses
    assert "loss_ce" in losses

    _, dis_losses = model.train_dis_loss(inputs, targets, outputs, DummyDiscriminator())
    assert "adv_fake" in dis_losses
    assert "adv_real" in dis_losses


def test_common_gan_postprocess_and_sample(small_batch, features) -> None:
    model = DummyGAN(features=features, max_seq_length=10)
    model.eval()
    outputs = {"bbox": torch.zeros(1, model.max_seq_length, 4)}
    processed = model.postprocess(outputs)
    assert "bbox" not in processed
    assert "center_x" in processed

    sampled, violation = model.sample(cond=small_batch, return_violation=True)
    assert "label" in sampled
    assert violation is None
