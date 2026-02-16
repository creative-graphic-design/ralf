import argparse
import os

from ralf.train.config import get_mock_train_cfg
from ralf.train.data import get_dataset
from ralf.train.models.retrieval.retriever import Retriever

DATASETS = ["pku", "cgl"]
RETRIEVAL_BACKBONES = ["saliency", "clip", "vgg"]


def main():
    """
    Pre-compute and cache indexes (and optionally similarity scores) for nearest neighbour search.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", type=str, default="pku", choices=DATASETS)
    parser.add_argument("--dataset_path", type=str)
    parser.add_argument(
        "--retrieval_backbone",
        type=str,
        default="dreamsim",
        choices=RETRIEVAL_BACKBONES,
    )
    parser.add_argument("--top_k", type=int, default=32)
    parser.add_argument("--max_items_per_split", type=int, default=None)
    parser.add_argument("--cache_dir", type=str, default=None)
    parser.add_argument("--save_index", dest="save_index", action="store_true")
    parser.add_argument("--no-save_index", dest="save_index", action="store_false")
    parser.set_defaults(save_index=True)
    parser.add_argument(
        "--save_scores",
        action="store_true",
        help="some reranking methods needs similarity scores between query and retrieved data",
    )
    args = parser.parse_args()

    preprocess_retriever(
        dataset_path=args.dataset_path,
        dataset_name=args.dataset_name,
        retrieval_backbone=args.retrieval_backbone,
        top_k=args.top_k,
        save_scores=args.save_scores,
        max_items_per_split=args.max_items_per_split,
        save_index=args.save_index,
        cache_dir=args.cache_dir,
    )


def preprocess_retriever(
    dataset_path: str = "/datasets/PosterLayout",
    dataset_name: str = "pku",
    max_seq_length: int = 10,
    retrieval_backbone: str = "saliency",
    top_k: int = 32,
    save_scores: bool = False,
    max_items_per_split: int | None = None,
    save_index: bool = True,
    cache_dir: str | None = None,
) -> None:
    if dataset_name == "pku":
        _data_dir = f"{dataset_name}{max_seq_length}"
    else:
        _data_dir = dataset_name

    train_cfg = get_mock_train_cfg(
        max_seq_length, os.path.join(dataset_path, _data_dir)
    )

    datasets, features = get_dataset(
        dataset_cfg=train_cfg.dataset,
        transforms=list(train_cfg.data.transforms),
        remove_column_names=["image_width", "image_height"],
    )

    db_dataset = datasets["train"]
    if max_items_per_split is not None:
        db_dataset = db_dataset.select(range(min(len(db_dataset), max_items_per_split)))

    retriever = Retriever(
        features=features,
        db_dataset=db_dataset,
        max_seq_length=max_seq_length,
        dataset_name=dataset_name,
        retrieval_backbone=retrieval_backbone,
        save_index=save_index,
        cache_dir=cache_dir,
    )

    for split in datasets.keys():
        current_dataset = datasets[split]
        if max_items_per_split is not None:
            current_dataset = current_dataset.select(
                range(min(len(current_dataset), max_items_per_split))
            )
        retriever.preprocess_retrieval_cache(
            split=split,
            dataset=current_dataset,
            top_k=top_k,
            run_on_local=True,
            save_scores=save_scores,
        )


def preprocess_merged_retriever(
    dataset_path: str = "/datasets/PosterLayout",
    dataset_name: str = "pku",
    max_seq_length: int = 10,
    retrieval_backbone: str = "clip",
    where_norm: str = "after_concat",
    top_k: int = 32,
):
    if dataset_name == "pku":
        _data_dir = f"{dataset_name}{max_seq_length}"
    else:
        _data_dir = dataset_name
    train_cfg = get_mock_train_cfg(
        max_seq_length, os.path.join(dataset_path, _data_dir)
    )

    datasets, features = get_dataset(
        dataset_cfg=train_cfg.dataset,
        transforms=list(train_cfg.data.transforms),
        remove_column_names=["image_width", "image_height"],
    )

    retriever = Retriever(
        features=features,
        db_dataset=datasets["train"],
        max_seq_length=max_seq_length,
        dataset_name=dataset_name,
        retrieval_backbone=retrieval_backbone,
    )

    for split in datasets.keys():
        retriever.preprocess_to_merge_retrieval_cache(
            dataset_name=dataset_name,
            split=split,
            dataset=datasets[split],
            top_k=top_k,
            run_on_local=True,
            where_norm=where_norm,
        )


if __name__ == "__main__":
    main()

# OMP_NUM_THREADS=2 uv run python -m ralf.preprocess.build_retrieval_indexes --dataset pku --dataset_path /datasets/PosterLayout
