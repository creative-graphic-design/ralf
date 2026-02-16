import os
from tempfile import TemporaryDirectory

import torch
from omegaconf import OmegaConf

from ralf.train.helpers.io import get_dim_model, load_model, save_model, shrink


def test_save_model_creates_file() -> None:
    with TemporaryDirectory() as tmpdir:
        model = torch.nn.Linear(2, 2)
        out_path = os.path.join(tmpdir, "model.pt")
        save_model(model, out_path)
        assert os.path.exists(out_path)


def test_io_helpers_roundtrip(tmp_path) -> None:
    cfg = OmegaConf.create({"encoder_layer": {"d_model": 16, "dim_feedforward": 32}})
    assert get_dim_model(cfg) == 16
    shrunk = shrink(cfg, mult=0.5)
    assert shrunk.encoder_layer.d_model == 8

    model = torch.nn.Linear(3, 4)
    ckpt_dir = tmp_path / "ckpt"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    save_model(model, str(ckpt_dir), best_or_final="final", prefix="unit")
    loaded = torch.nn.Linear(3, 4)
    load_model(
        loaded, str(ckpt_dir), torch.device("cpu"), best_or_final="final", prefix="unit"
    )
    for p1, p2 in zip(model.parameters(), loaded.parameters()):
        assert torch.allclose(p1, p2)
