import os

import datasets as ds
import numpy as np
import torch

from ralf.train.models.common.base_model import ConditionalInputsForDiscreteLayout
from ralf.train.models.retrieval.cross_retriever import CrossRetriever
from ralf.train.models.retrieval.image import coarse_saliency


def _build_dataset() -> ds.Dataset:
    saliency0 = torch.zeros(1, 350, 240)
    saliency1 = torch.ones(1, 350, 240) * 0.5
    search_feat = [coarse_saliency(saliency0), coarse_saliency(saliency1)]
    data = {
        "id": ["0", "1"],
        "saliency": [saliency0.numpy(), saliency1.numpy()],
        "search_feat": [np.array(search_feat[0]), np.array(search_feat[1])],
        "label": [[0], [1]],
        "center_x": [[0.5], [0.6]],
        "center_y": [[0.5], [0.6]],
        "width": [[0.1], [0.2]],
        "height": [[0.1], [0.2]],
        "mask": [[True], [True]],
    }
    return ds.Dataset.from_dict(data)


def test_cross_retriever_sample(tmp_path) -> None:
    import ralf.train.models.retrieval.cross_retriever as cr

    cr.CACHE_DIR = str(tmp_path)
    dataset = _build_dataset()
    dataset.add_faiss_index("search_feat")
    dataset.save_faiss_index(
        "search_feat", os.path.join(cr.CACHE_DIR, "pku_saliency_wo_head_index.faiss")
    )
    dataset.save_faiss_index(
        "search_feat", os.path.join(cr.CACHE_DIR, "cgl_saliency_wo_head_index.faiss")
    )

    retriever = CrossRetriever(
        features_pku=dataset.features,
        features_cgl=dataset.features,
        db_dataset_pku=dataset,
        db_dataset_cgl=dataset,
        max_seq_length=1,
        top_k=1,
        retrieval_backbone="saliency",
        saliency_k=0,
    )
    retriever.db_dataset = dataset
    image = torch.zeros(1, 4, 350, 240)
    cond = ConditionalInputsForDiscreteLayout(image=image, id=None)
    outputs, violation = retriever.sample(cond)
    assert "label" in outputs and "mask" in outputs


def test_cross_retriever_preprocess_cache(tmp_path) -> None:
    import ralf.train.models.retrieval.cross_retriever as cr

    cr.CACHE_DIR = str(tmp_path)
    dataset = _build_dataset()
    dataset.add_faiss_index("search_feat")
    dataset.save_faiss_index(
        "search_feat", os.path.join(cr.CACHE_DIR, "pku_saliency_wo_head_index.faiss")
    )
    dataset.save_faiss_index(
        "search_feat", os.path.join(cr.CACHE_DIR, "cgl_saliency_wo_head_index.faiss")
    )

    for name in ["pku", "cgl"]:
        cache_path = os.path.join(
            cr.CACHE_DIR, f"{name}_saliency_cache_table_paired.pt"
        )
        torch.save({"0": 0, "1": 1}, cache_path)

    retriever = CrossRetriever(
        features_pku=dataset.features,
        features_cgl=dataset.features,
        db_dataset_pku=dataset,
        db_dataset_cgl=dataset,
        max_seq_length=1,
        top_k=1,
        retrieval_backbone="saliency",
        saliency_k=0,
    )

    dataset_torch = dataset.with_format("torch")
    retriever.preprocess_retrieval_cache(
        split="train",
        source="pku",
        reference="cgl",
        dataset_source=dataset_torch,
        dataset_reference=dataset_torch,
        top_k=1,
        run_on_local=True,
        save_scores=True,
    )
    cache_path = os.path.join(
        cr.CACHE_DIR,
        "source_pku_reference_cgl_train_saliency_cross_dataset_indexes_top_k1.pt",
    )
    assert os.path.exists(cache_path)
