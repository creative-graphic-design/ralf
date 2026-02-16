import os
from typing import Iterator

_CACHE_ROOT = os.environ.get(
    "RALF_CACHE_DIR", "/cvl-gen-vsfs2/public_dataset/layout/RALF_cache"
)
os.environ.setdefault("RALF_CACHE_DIR", _CACHE_ROOT)
os.environ.setdefault("RALF_DATASET_DIR", os.path.join(_CACHE_ROOT, "dataset"))
os.environ.setdefault(
    "RALF_PRECOMPUTED_WEIGHT_DIR",
    os.path.join(_CACHE_ROOT, "PRECOMPUTED_WEIGHT_DIR"),
)

import datasets as ds
import pytest
import torch
from omegaconf import OmegaConf

from ralf.train.config import TokenizerConfig
from ralf.train.helpers.layout_tokenizer import LayoutSequenceTokenizer


@pytest.fixture(scope="session")
def cache_root() -> str:
    return _CACHE_ROOT


@pytest.fixture(scope="session", autouse=True)
def _set_env(cache_root: str) -> Iterator[None]:
    os.environ.setdefault("RALF_CACHE_DIR", cache_root)
    os.environ.setdefault("RALF_DATASET_DIR", os.path.join(cache_root, "dataset"))
    os.environ.setdefault(
        "RALF_PRECOMPUTED_WEIGHT_DIR",
        os.path.join(cache_root, "PRECOMPUTED_WEIGHT_DIR"),
    )
    yield


@pytest.fixture(scope="session")
def dataset_dir(cache_root: str) -> str:
    return os.path.join(cache_root, "dataset", "cgl")


@pytest.fixture(scope="session")
def dataset(dataset_dir: str) -> tuple[ds.DatasetDict, ds.Features]:
    from ralf.train.data import get_dataset

    cfg = OmegaConf.create(
        {
            "data_dir": dataset_dir,
            "path": "",
            "data_type": "parquet",
            "max_seq_length": 10,
            "name": "cgl",
        }
    )
    dataset, features = get_dataset(dataset_cfg=cfg, transforms=["image", "shuffle"])  # type: ignore
    dataset = ds.DatasetDict(
        {
            split: dataset[split].select(range(min(8, len(dataset[split]))))
            for split in dataset.keys()
        }
    )
    return dataset, features


@pytest.fixture(scope="session")
def features(dataset: tuple[ds.DatasetDict, ds.Features]) -> ds.Features:
    return dataset[1]


@pytest.fixture()
def small_batch(dataset: tuple[ds.DatasetDict, ds.Features]) -> dict[str, torch.Tensor]:
    from ralf.train.data import collate_fn

    ds_dict = dataset[0]
    batch = [ds_dict["train"][0], ds_dict["train"][1]]
    return collate_fn(batch, max_seq_length=10)


@pytest.fixture()
def layout_tokenizer(features: ds.Features) -> LayoutSequenceTokenizer:
    cfg = TokenizerConfig()
    return LayoutSequenceTokenizer(
        label_feature=features["label"].feature,
        max_seq_length=10,
        **cfg.__dict__,
    )
