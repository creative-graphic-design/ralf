import os

import pytest
import torch

from ralf.train.config import get_mock_train_cfg
from ralf.train.helpers.layout_tokenizer import init_layout_tokenizer


@pytest.mark.cuda
def test_autoreg_forward() -> None:
    pytest.importorskip("cv2", exc_type=ImportError)
    from ralf.train.models.autoreg import Autoreg

    cache_root = os.environ.get(
        "RALF_CACHE_DIR", "/cvl-gen-vsfs2/public_dataset/layout/RALF_cache"
    )
    dataset_path = os.path.join(cache_root, "dataset", "cgl")
    from ralf.train.data import collate_fn, get_dataset

    train_cfg = get_mock_train_cfg(10, dataset_path)
    train_cfg.tokenizer = {"num_bin": 128, "geo_quantization": "linear"}
    datasets, features = get_dataset(
        dataset_cfg=train_cfg.dataset,
        transforms=list(train_cfg.data.transforms),
        remove_column_names=["image_width", "image_height"],
    )
    tokenizer = init_layout_tokenizer(
        train_cfg.tokenizer, train_cfg.dataset, features["label"].feature
    )

    batch = collate_fn([datasets["train"][0], datasets["train"][1]], max_seq_length=10)
    batch = {k: v for k, v in batch.items() if k not in {"id"}}
    model = Autoreg(
        features=features,
        tokenizer=tokenizer,
        d_model=256,
    )
    model.eval()
    seq = tokenizer.encode(batch)["seq"]
    image = torch.cat([batch["image"], batch["saliency"]], dim=1)
    inputs = {
        "image": image,
        "seq": seq[:, :-1],
        "tgt_key_padding_mask": seq[:, :-1] == tokenizer.name_to_id("pad"),
    }
    with torch.no_grad():
        out = model.forward(inputs)
    assert out["logits"].shape[0] == seq.shape[0]
