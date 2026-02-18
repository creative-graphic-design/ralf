import os
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from ralf.hfds_builder.models.inpainting import (
    SimpleLama,
    ceil_modulo,
    get_cache_path_by_url,
    get_image,
    pad_img_to_modulo,
    prepare_img_and_mask,
    scale_image,
)


def _make_dummy_lama(path: Path) -> None:
    class Dummy(torch.nn.Module):
        def forward(self, image, mask):
            return image

    model = Dummy()
    example = (
        torch.zeros(1, 3, 8, 8),
        torch.zeros(1, 3, 8, 8),
    )
    scripted = torch.jit.trace(model, example)
    scripted.save(str(path))


def test_inpainting_helpers(tmp_path) -> None:
    image = Image.fromarray(np.zeros((8, 8, 3), dtype=np.uint8))
    array = get_image(image)
    assert array.shape[0] == 3
    assert ceil_modulo(9, 8) == 16
    scaled = scale_image(array, factor=0.5)
    assert scaled.ndim == 3
    padded = pad_img_to_modulo(array, mod=8)
    assert padded.shape[1] % 8 == 0
    img_t, mask_t = prepare_img_and_mask(image, image, device=torch.device("cpu"))
    assert img_t.shape[0] == 1
    assert mask_t.max() <= 1
    assert get_cache_path_by_url("https://example.com/model.pt").endswith("model.pt")


def test_simple_lama_with_dummy_model(tmp_path) -> None:
    model_path = tmp_path / "dummy_lama.pt"
    _make_dummy_lama(model_path)
    os.environ["LAMA_MODEL"] = str(model_path)
    lama = SimpleLama(device=torch.device("cpu"))
    image = Image.fromarray(np.zeros((8, 8, 3), dtype=np.uint8))
    mask = Image.fromarray(np.zeros((8, 8, 3), dtype=np.uint8))
    out = lama(image, mask)
    del os.environ["LAMA_MODEL"]
    assert isinstance(out, Image.Image)


def test_get_image_invalid_input_raises() -> None:
    try:
        get_image("not-an-image")
    except Exception as exc:
        assert "Input image should be either" in str(exc)


def test_prepare_img_and_mask_with_scale_factor() -> None:
    image = Image.fromarray(np.zeros((8, 8, 3), dtype=np.uint8))
    img_t, mask_t = prepare_img_and_mask(
        image, image, device=torch.device("cpu"), scale_factor=0.5
    )
    assert img_t.shape[0] == 1
    assert mask_t.shape[0] == 1
