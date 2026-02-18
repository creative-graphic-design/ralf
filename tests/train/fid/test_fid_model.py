import os

import pytest
import torch


@pytest.mark.cuda
def test_fid_model_load_and_forward() -> None:
    cache_root = os.environ.get(
        "RALF_CACHE_DIR", "/cvl-gen-vsfs2/public_dataset/layout/RALF_cache"
    )
    weight_dir = os.path.join(cache_root, "PRECOMPUTED_WEIGHT_DIR", "fidnet", "cgl")
    from ralf.train.fid.model import FIDNetV3, load_fidnet_v3

    model = FIDNetV3(num_label=4, max_bbox=10)
    model = load_fidnet_v3(model, weight_dir)
    model.eval()
    dummy = {
        "label": torch.zeros((1, 10), dtype=torch.long),
        "center_x": torch.zeros((1, 10)),
        "center_y": torch.zeros((1, 10)),
        "width": torch.zeros((1, 10)),
        "height": torch.zeros((1, 10)),
        "mask": torch.ones((1, 10), dtype=torch.bool),
    }
    with torch.no_grad():
        feats = model.extract_features(dummy)
    assert feats.shape[0] == 1
