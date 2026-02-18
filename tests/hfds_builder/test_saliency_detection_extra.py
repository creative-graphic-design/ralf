import os
from pathlib import Path

import pytest
import torch
from PIL import Image

from ralf.hfds_builder.saliency_detection import _norm_pred


def test_norm_pred_range() -> None:
    data = torch.tensor([0.0, 1.0, 2.0]).view(1, 1, 3)
    out = _norm_pred(data)
    assert torch.isfinite(out).all()
    assert out.min().item() >= 0.0
    assert out.max().item() <= 1.0


def test_basnet_tester_smoke() -> None:
    pytest.importorskip("cv2", exc_type=ImportError)
    from ralf.hfds_builder.saliency_detection import BASNetSaliencyTester

    cache_root = os.environ.get(
        "RALF_CACHE_DIR", "/cvl-gen-vsfs2/public_dataset/layout/RALF_cache"
    )
    weight_root = Path(cache_root) / "hfds_builder" / "saliency_detection"
    if not (weight_root / "gdi-basnet.pth").exists():
        pytest.skip("BASNet weight not found")

    tester = BASNetSaliencyTester()
    image = Image.new("RGB", (64, 64), color=(0, 0, 0))
    pred = tester(image)
    assert pred.ndim == 4
