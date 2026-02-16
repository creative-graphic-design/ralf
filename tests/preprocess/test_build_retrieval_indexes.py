import os
import sys
from pathlib import Path

from ralf.preprocess import build_retrieval_indexes
from ralf.preprocess.build_retrieval_indexes import preprocess_retriever


def test_preprocess_retriever_small(tmp_path: Path, dataset_dir: str) -> None:
    dataset_root = os.path.dirname(dataset_dir)
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)

    preprocess_retriever(
        dataset_path=dataset_root,
        dataset_name="cgl",
        max_seq_length=10,
        retrieval_backbone="saliency",
        top_k=1,
        save_scores=False,
        max_items_per_split=2,
        save_index=False,
        cache_dir=str(cache_dir),
    )

    expected = [
        "cgl_train_saliency_wo_head_table_between_dataset_indexes_top_k1.pt",
        "cgl_test_saliency_wo_head_table_between_dataset_indexes_top_k1.pt",
        "cgl_val_saliency_wo_head_table_between_dataset_indexes_top_k1.pt",
        "cgl_with_no_annotation_saliency_wo_head_table_between_dataset_indexes_top_k1.pt",
    ]
    for name in expected:
        assert (cache_dir / name).exists()


def test_build_retrieval_indexes_main(tmp_path: Path, dataset_dir: str) -> None:
    dataset_root = os.path.dirname(dataset_dir)
    cache_dir = tmp_path / "cache_main"
    cache_dir.mkdir(parents=True, exist_ok=True)

    old_argv = sys.argv
    sys.argv = [
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
        "2",
        "--cache_dir",
        str(cache_dir),
        "--no-save_index",
    ]
    try:
        build_retrieval_indexes.main()
    finally:
        sys.argv = old_argv

    expected = (
        cache_dir / "cgl_train_saliency_wo_head_table_between_dataset_indexes_top_k1.pt"
    )
    assert expected.exists()
