import os

import pytest
import torch

from ralf.train.helpers.retrieval_dataset_wrapper import RetrievalDatasetWrapper


def _dataset_root() -> str:
    cache_root = os.environ.get(
        "RALF_CACHE_DIR", "/cvl-gen-vsfs2/public_dataset/layout/RALF_cache"
    )
    return os.path.join(cache_root, "dataset", "cgl")


@pytest.mark.cuda
def test_retrieval_dataset_wrapper() -> None:
    dataset_dir = _dataset_root()
    from ralf.train.data import get_dataset

    cfg = type("Cfg", (), {"data_dir": dataset_dir, "path": "", "data_type": "parquet"})
    dataset, _ = get_dataset(dataset_cfg=cfg, transforms=["image"])
    wrapper = RetrievalDatasetWrapper(
        dataset_name="cgl",
        dataset=dataset["train"].select(range(2)),
        db_dataset=dataset["train"],
        split="train",
        top_k=1,
        num_cache_indexes_per_sample=32,
        max_seq_length=10,
        retrieval_backbone="dreamsim",
        random_retrieval=False,
        saliency_k=0,
        inference_num_saliency=0,
    )
    item = wrapper[0]
    assert "retrieved" in item
    assert torch.is_tensor(item["retrieved"][0]["mask"])


# cross-dataset wrapper requires special cached indices and with_no_annotation split
