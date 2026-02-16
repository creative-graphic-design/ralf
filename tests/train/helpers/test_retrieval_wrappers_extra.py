import importlib
import os

import torch

from ralf.train.helpers.random_retrieval_dataset_wrapper import (
    RandomRetrievalDatasetWrapper,
)
from ralf.train.helpers.retrieval_dataset_wrapper import load_cache_table


def test_random_retrieval_dataset_wrapper(dataset) -> None:
    ds_dict, _ = dataset
    wrapper = RandomRetrievalDatasetWrapper(
        dataset_name="cgl",
        dataset=ds_dict["train"],
        db_dataset=ds_dict["train"],
        split="train",
        top_k=1,
        max_seq_length=10,
        retrieval_backbone="saliency",
        random_retrieval=True,
        saliency_k=1,
    )
    item = wrapper[0]
    assert "retrieved" in item
    retrieved = item["retrieved"][0]
    assert retrieved["label"].shape[0] == 1
    assert retrieved["mask"].shape[0] == 1


def test_retrieval_cross_dataset_wrapper(tmp_path, dataset) -> None:
    ds_dict, _ = dataset
    split = "with_no_annotation"
    data_ids = [ds_dict[split][i]["id"] for i in range(2)]
    table_idx = {data_id: [0] for data_id in data_ids}
    cache_name = (
        f"source_cgl_reference_cgl_{split}_dreamsim_cross_dataset_indexes_top_k16.pt"
    )
    cache_path = tmp_path / cache_name
    torch.save(table_idx, cache_path)

    old_cache_dir = os.environ.get("RALF_CACHE_DIR")
    try:
        os.environ["RALF_CACHE_DIR"] = str(tmp_path)
        import ralf.train.global_variables as gv
        import ralf.train.helpers.retrieval_cross_dataset_wrapper as rcdw

        importlib.reload(gv)
        rcdw = importlib.reload(rcdw)

        wrapper = rcdw.RetrievalCrossDatasetWrapper(
            dataset_source_name="cgl",
            dataset_reference_name="cgl",
            dataset=ds_dict[split],
            db_dataset=ds_dict["train"],
            split=split,
            top_k=1,
            max_seq_length=10,
            retrieval_backbone="dreamsim",
        )
        item = wrapper[0]
        assert "retrieved" in item
        retrieved = item["retrieved"][0]
        assert retrieved["label"].shape[0] == 1
    finally:
        if old_cache_dir is None:
            os.environ.pop("RALF_CACHE_DIR", None)
        else:
            os.environ["RALF_CACHE_DIR"] = old_cache_dir
        import ralf.train.global_variables as gv

        importlib.reload(gv)


def test_load_cache_table_missing(tmp_path) -> None:
    missing_path = tmp_path / "missing_cache.pt"
    try:
        load_cache_table(str(missing_path), top_k=1)
    except ValueError as exc:
        assert "Cache not found" in str(exc)
