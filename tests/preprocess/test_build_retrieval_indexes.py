import os
import sys

from ralf.preprocess import build_retrieval_indexes
from ralf.preprocess.build_retrieval_indexes import preprocess_retriever


def test_preprocess_retriever_small(tmp_path, dataset_dir) -> None:
    dataset_root = os.path.dirname(dataset_dir)
    preprocess_retriever(
        dataset_path=dataset_root,
        dataset_name="cgl",
        retrieval_backbone="saliency",
        top_k=1,
        max_items_per_split=2,
        save_scores=False,
        save_index=False,
        cache_dir=str(tmp_path),
    )

    expected_cache = tmp_path / "cgl_saliency_cache_table_paired.pt"
    assert expected_cache.exists()


def test_build_retrieval_indexes_main(tmp_path, dataset_dir) -> None:
    dataset_root = os.path.dirname(dataset_dir)
    argv = [
        "build_retrieval_indexes",
        "--dataset_name",
        "cgl",
        "--dataset_path",
        dataset_root,
        "--retrieval_backbone",
        "saliency",
        "--top_k",
        "1",
        "--max_items_per_split",
        "1",
        "--cache_dir",
        str(tmp_path),
        "--no-save_index",
    ]
    original_argv = sys.argv
    try:
        sys.argv = argv
        build_retrieval_indexes.main()
    finally:
        sys.argv = original_argv

    expected_cache = tmp_path / "cgl_saliency_cache_table_paired.pt"
    assert expected_cache.exists()
