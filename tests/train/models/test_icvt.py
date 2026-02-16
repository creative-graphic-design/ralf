import torch

from ralf.train.models.icvt import (
    ICVTGenerator,
    Tokenizer,
    VAEModule,
    _make_grid_like_layout,
)


def test_icvt_tokenizer_roundtrip() -> None:
    tokenizer = Tokenizer(num_classes=3, n_boundaries=16)
    inputs = {
        "label": torch.tensor([[0, 1]]),
        "center_x": torch.tensor([[0.2, 0.8]]),
        "center_y": torch.tensor([[0.3, 0.7]]),
        "width": torch.tensor([[0.1, 0.2]]),
        "height": torch.tensor([[0.1, 0.2]]),
        "mask": torch.tensor([[True, True]]),
    }
    encoded = tokenizer.encode(inputs)
    decoded = tokenizer.decode(encoded)
    assert decoded["mask"].shape == inputs["mask"].shape


def test_icvt_helpers() -> None:
    layout = _make_grid_like_layout(2, 3)
    assert layout["center_x"].shape == (3, 2)
    vae = VAEModule(dim_input=4, dim_latent=2)
    out = vae(torch.zeros(1, 4))
    assert "z" in out and "mu" in out and "logvar" in out


def test_icvt_preprocess_and_update(small_batch, features) -> None:
    model = ICVTGenerator(features=features, d_model=40, backbone="resnet18")
    model.eval()
    inputs, targets = model.preprocess(small_batch)
    assert "image" in inputs and "label" in inputs
    model.update_per_epoch(epoch=1, warmup_dis_epoch=2, max_epoch=4)
    assert "kl" in model.loss_weight_dict


def test_icvt_train_loss(small_batch, features) -> None:
    model = ICVTGenerator(features=features, d_model=40, backbone="resnet18")
    model.eval()
    inputs, targets = model.preprocess(small_batch)
    with torch.no_grad():
        outputs, losses = model.train_loss(inputs, targets)
    assert "loss_kl" in losses
    assert outputs["label"].shape[0] == small_batch["image"].shape[0]
