import os

import datasets as ds
import faiss
import numpy as np
import pytest
import torch

from ralf.train.config import get_mock_train_cfg
from ralf.train.models.common.base_model import ConditionalInputsForDiscreteLayout
from ralf.train.models.retrieval.retriever import Retriever


def _build_small_dataset() -> ds.Dataset:
    saliency0 = np.zeros((1, 350, 240), dtype=np.float32)
    saliency1 = np.ones((1, 350, 240), dtype=np.float32) * 0.5
    data = {
        "id": ["0", "1"],
        "saliency": [saliency0, saliency1],
        "label": [[0], [1]],
        "center_x": [[0.5], [0.6]],
        "center_y": [[0.5], [0.6]],
        "width": [[0.1], [0.2]],
        "height": [[0.1], [0.2]],
        "mask": [[True], [True]],
    }
    dataset = ds.Dataset.from_dict(data)
    return dataset.with_format("torch")


@pytest.mark.cuda
def test_retriever_sample() -> None:
    cache_root = os.environ.get(
        "RALF_CACHE_DIR", "/cvl-gen-vsfs2/public_dataset/layout/RALF_cache"
    )
    dataset_path = os.path.join(cache_root, "dataset")
    from ralf.train.data import get_dataset

    train_cfg = get_mock_train_cfg(10, os.path.join(dataset_path, "cgl"))
    datasets, features = get_dataset(
        dataset_cfg=train_cfg.dataset,
        transforms=list(train_cfg.data.transforms),
        remove_column_names=["image_width", "image_height"],
    )
    retriever = Retriever(
        features=features,
        db_dataset=datasets["train"].select(range(4)),
        max_seq_length=train_cfg.dataset.max_seq_length,
        dataset_name="cgl",
        retrieval_backbone="saliency",
        save_index=False,
    )
    sample = datasets["val"][0]
    image = torch.cat([sample["image"], sample["saliency"]], dim=0)
    cond = ConditionalInputsForDiscreteLayout(
        image=image.unsqueeze(0),
        id=None,
    )
    outputs, _ = retriever.sample(cond)
    assert "label" in outputs


def test_retriever_cache_small(tmp_path, features) -> None:
    dataset = _build_small_dataset()
    retriever = Retriever(
        features=features,
        db_dataset=dataset,
        max_seq_length=1,
        dataset_name="cgl",
        retrieval_backbone="saliency",
        save_index=False,
        cache_dir=str(tmp_path),
    )
    table = retriever.preprocess_retrieval_cache(
        split="train",
        dataset=dataset,
        top_k=1,
        run_on_local=True,
        save_scores=True,
    )
    assert isinstance(table, dict)


def test_retriever_saves_index(tmp_path, features) -> None:
    dataset = _build_small_dataset()
    retriever = Retriever(
        features=features,
        db_dataset=dataset,
        max_seq_length=1,
        dataset_name="dummy",
        retrieval_backbone="saliency",
        save_index=True,
        cache_dir=str(tmp_path),
    )
    expected = tmp_path / "dummy_saliency_wo_head_index.faiss"
    assert expected.exists()
    assert "search_feat" in dataset.list_indexes()
    _ = retriever


def test_retriever_loads_cache(tmp_path, features) -> None:
    dataset = _build_small_dataset()
    dim = 16 * 16
    vectors = np.random.randn(len(dataset), dim).astype("float32")
    index = faiss.IndexFlatIP(dim)
    index.add(vectors)
    cache_path = tmp_path / "dummy_saliency_wo_head_index.faiss"
    faiss.write_index(index, str(cache_path))

    Retriever(
        features=features,
        db_dataset=dataset,
        max_seq_length=1,
        dataset_name="dummy",
        retrieval_backbone="saliency",
        save_index=False,
        cache_dir=str(tmp_path),
    )
    assert "search_feat" in dataset.list_indexes()


def test_retriever_merge_cache(tmp_path, features) -> None:
    dataset = _build_small_dataset()
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)

    # Prepare a tiny FAISS index for saliency backbone
    dim = 16 * 16
    vectors = np.random.randn(len(dataset), dim).astype("float32")
    index = faiss.IndexFlatIP(dim)
    index.add(vectors)
    faiss.write_index(index, str(tmp_path / "dummy_saliency_wo_head_index.faiss"))

    # Prepare paired id -> db index cache
    table_paired = {dataset[i]["id"]: i for i in range(len(dataset))}
    torch.save(
        table_paired,
        cache_dir / "dummy_train_saliency_cache_table_paired.pt",
    )

    retriever = Retriever(
        features=features,
        db_dataset=dataset,
        max_seq_length=1,
        dataset_name="dummy",
        retrieval_backbone="merge_saliency",
        save_index=False,
        cache_dir=str(cache_dir),
    )

    cwd = os.getcwd()
    os.chdir(tmp_path)
    try:
        table = retriever.preprocess_to_merge_retrieval_cache(
            dataset_name="dummy",
            split="train",
            dataset=dataset,
            top_k=1,
            run_on_local=True,
            where_norm="before_concat",
        )
    finally:
        os.chdir(cwd)
    assert table is None
    cache_path = cache_dir / "dummy_train_merge_saliency_before_concat__topk1.pt"
    assert cache_path.exists()
