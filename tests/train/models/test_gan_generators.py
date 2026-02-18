import torch

from ralf.train.models.cgl import CGLGenerator
from ralf.train.models.dsgan import DSGenerator


def test_dsgan_forward(small_batch, features) -> None:
    num_classes = features["label"].feature.num_classes
    in_channels = 2 * (num_classes + 1)
    model = DSGenerator(features=features, max_seq_length=10, in_channels=in_channels)
    model.eval()
    inputs, _targets = model.preprocess(small_batch)
    with torch.no_grad():
        outputs = model(inputs)
    assert outputs["pred_logits"].shape[0] == small_batch["image"].shape[0]
    assert outputs["pred_boxes"].shape[-1] == 4


def test_cgl_generator_forward(small_batch, features) -> None:
    num_classes = features["label"].feature.num_classes
    in_channels = 2 * (num_classes + 1)
    model = CGLGenerator(features=features, max_seq_length=10, in_channels=in_channels)
    model.eval()
    inputs, _targets = model.preprocess(small_batch)
    with torch.no_grad():
        outputs = model(inputs)
    assert outputs["pred_logits"].shape[0] == small_batch["image"].shape[0]
    assert outputs["pred_boxes"].shape[-1] == 4
