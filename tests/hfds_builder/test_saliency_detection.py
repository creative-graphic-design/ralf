import os
from pathlib import Path

import numpy as np
import pytest
from PIL import Image


def test_saliency_detection_smoke(tmp_path) -> None:
    pytest.importorskip("cv2", exc_type=ImportError)
    from ralf.hfds_builder.saliency_detection import main as saliency_main

    cache_root = os.environ.get(
        "RALF_CACHE_DIR", "/cvl-gen-vsfs2/public_dataset/layout/RALF_cache"
    )
    weight_root = Path(cache_root) / "hfds_builder" / "saliency_detection"
    assert weight_root.exists()

    img = Image.fromarray(np.zeros((240, 350, 3), dtype=np.uint8))
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"
    input_dir.mkdir()
    output_dir.mkdir()
    img.save(input_dir / "sample.png")

    saliency_main(
        input_dir=str(input_dir),
        output_dir=str(output_dir),
        algorithm="isnet",
        weight_dir=str(weight_root),
    )
    assert len(list(output_dir.glob("*.png"))) > 0
