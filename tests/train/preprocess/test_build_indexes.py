import os

import pytest


@pytest.mark.cuda
def test_build_retrieval_indexes_smoke(tmp_path) -> None:
    from ralf.preprocess.build_retrieval_indexes import preprocess_retriever

    cache_root = os.environ.get(
        "RALF_CACHE_DIR", "/cvl-gen-vsfs2/public_dataset/layout/RALF_cache"
    )
    dataset_path = os.path.join(cache_root, "dataset")
    preprocess_retriever(
        dataset_path=dataset_path,
        dataset_name="cgl",
        max_seq_length=10,
        retrieval_backbone="saliency",
        top_k=1,
        save_scores=False,
        max_items_per_split=2,
        save_index=False,
        cache_dir=str(tmp_path),
    )
