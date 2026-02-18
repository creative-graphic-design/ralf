import os
from pathlib import Path

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf
from PIL import Image

from ralf.train import inference, inference_single_data, inference_unanno
from ralf.train.config import get_mock_train_cfg
from ralf.train.data import get_dataset


def _prepare_retrieval_cache(
    cache_dir: Path,
    dataset_dir: str,
    split: str,
    top_k: int = 32,
    retrieval_backbone: str = "saliency",
) -> None:
    train_cfg = get_mock_train_cfg(10, dataset_dir)
    datasets, _ = get_dataset(
        dataset_cfg=train_cfg.dataset,
        transforms=list(train_cfg.data.transforms),
        remove_column_names=["image_width", "image_height"],
    )
    ds_split = datasets[split]
    ids = [ds_split[i]["id"] for i in range(min(4, len(ds_split)))]
    table = {data_id: [0] * top_k for data_id in ids}
    cache_path = (
        cache_dir
        / f"cgl_{split}_{retrieval_backbone}_wo_head_table_between_dataset_indexes_top_k{top_k}.pt"
    )
    torch.save(table, cache_path)


def _write_train_config(
    job_dir: Path,
    dataset_dir: str,
    generator_target: str = "ralf.train.models.autoreg.Autoreg",
    generator_extra: dict | None = None,
    generator_override: dict | None = None,
    tokenization: bool = True,
    max_seq_length: int = 10,
) -> None:
    if generator_override is not None:
        generator = generator_override
    else:
        generator = {
            "_target_": generator_target,
            "_partial_": True,
            "d_model": 256,
        }
        if generator_extra:
            generator.update(generator_extra)
    cfg = OmegaConf.create(
        {
            "dataset": {
                "name": "cgl",
                "max_seq_length": max_seq_length,
                "data_dir": dataset_dir,
                "data_type": "parquet",
                "path": None,
            },
            "data": {"transforms": ["image", "shuffle"], "tokenization": tokenization},
            "tokenizer": {"num_bin": 128, "geo_quantization": "linear"},
            "generator": generator,
            "sampling": {"name": "random", "temperature": 1.0},
            "run_on_local": True,
        }
    )
    job_dir.mkdir(parents=True, exist_ok=True)
    (job_dir / "config.yaml").write_text(OmegaConf.to_yaml(cfg))


def _prepare_job(
    job_dir: Path,
    dataset_dir: str,
    features,
    generator_target: str = "ralf.train.models.autoreg.Autoreg",
    generator_extra: dict | None = None,
    generator_override: dict | None = None,
    tokenization: bool = True,
    db_dataset=None,
    max_seq_length: int = 10,
) -> None:
    _write_train_config(
        job_dir,
        dataset_dir,
        generator_target,
        generator_extra,
        generator_override,
        tokenization,
        max_seq_length,
    )
    train_cfg = OmegaConf.load(job_dir / "config.yaml")
    model, _ = inference.build_network(
        train_cfg=train_cfg,
        features=features,
        max_seq_length=max_seq_length,
        db_dataset=db_dataset,
    )
    ckpt_path = job_dir / "model_final.pt"
    torch.save(model.state_dict(), ckpt_path)


@pytest.fixture(scope="module")
def job_dir_with_ckpt(tmp_path_factory, dataset_dir, features) -> Path:
    base_dir = tmp_path_factory.mktemp("inference_jobs")
    job_dir = base_dir / "job"
    _prepare_job(job_dir, dataset_dir, features)
    return job_dir


@pytest.fixture(scope="module")
def job_dir_with_ckpt_retrieval_aug(
    tmp_path_factory, dataset_dir, features, dataset
) -> Path:
    base_dir = tmp_path_factory.mktemp("inference_jobs_retrieval")
    job_dir = base_dir / "job_retrieval"
    ds_dict, _ = dataset
    _prepare_job(
        job_dir,
        dataset_dir,
        features,
        generator_target="ralf.train.models.retrieval_augmented_autoreg.ConcateAuxilaryTaskConcateCrossAttnRetrievalAugmentedAutoreg",
        generator_extra={
            "top_k": 1,
            "retrieval_backbone": "saliency",
            "random_retrieval": True,
            "saliency_k": 1,
            "use_reference_image": False,
            "auxilary_task": "uncond",
            "use_multitask": False,
            "use_flag_embedding": True,
            "global_task_embedding": False,
        },
        db_dataset=ds_dict["train"],
    )
    return job_dir


def test_inference_main_paths(
    tmp_path: Path, dataset_dir: str, job_dir_with_ckpt
) -> None:
    job_dir = job_dir_with_ckpt
    result_dir = tmp_path / "results"

    test_cfg = OmegaConf.create(
        {
            "job_dir": str(job_dir),
            "dataset_path": dataset_dir,
            "result_dir": str(result_dir),
            "test_split": "train",
            "cond_type": "uncond",
            "batch_size": 1,
            "debug": True,
            "debug_num_samples": 0,
            "num_seeds": 1,
            "use_db_dataset": False,
            "ckpt_filter_substring": None,
            "sampling": {"name": "random", "temperature": 1.0},
            "generator": {"retrieval_backbone": "saliency"},
            "preload_data": False,
        }
    )
    inference.main.__wrapped__(test_cfg)


def test_inference_main_preload_use_db_dataset(
    tmp_path: Path, dataset_dir: str, job_dir_with_ckpt_retrieval_aug
) -> None:
    job_dir = job_dir_with_ckpt_retrieval_aug
    result_dir = tmp_path / "results_retrieval"
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    _prepare_retrieval_cache(cache_dir, dataset_dir, split="train", top_k=32)

    import ralf.train.helpers.retrieval_dataset_wrapper as rdw

    old_cache_dir = rdw.CACHE_DIR
    old_precomputed = rdw.PRECOMPUTED_WEIGHT_DIR
    rdw.CACHE_DIR = str(cache_dir)
    rdw.PRECOMPUTED_WEIGHT_DIR = str(cache_dir)

    test_cfg = OmegaConf.create(
        {
            "job_dir": str(job_dir),
            "dataset_path": dataset_dir,
            "result_dir": str(result_dir),
            "test_split": "train",
            "cond_type": "uncond",
            "batch_size": 1,
            "debug": True,
            "debug_num_samples": 0,
            "num_seeds": 1,
            "use_db_dataset": True,
            "ckpt_filter_substring": None,
            "sampling": {"name": "random", "temperature": 1.0},
            "generator": {"retrieval_backbone": "saliency"},
            "preload_data": True,
        }
    )
    try:
        inference.main.__wrapped__(test_cfg)
    finally:
        rdw.CACHE_DIR = old_cache_dir
        rdw.PRECOMPUTED_WEIGHT_DIR = old_precomputed


def test_inference_single_data_main(tmp_path: Path, dataset_dir: str, features) -> None:
    job_dir = tmp_path / "job_single"
    _prepare_job(
        job_dir,
        dataset_dir,
        features,
        generator_target="ralf.train.models.autoreg.ConcateAuxilaryTaskAutoreg",
        generator_extra={"auxilary_task": "uncond"},
    )

    test_cfg = OmegaConf.create(
        {
            "job_dir": str(job_dir),
            "dataset_path": dataset_dir,
            "result_dir": str(tmp_path / "results"),
            "test_split": "train",
            "cond_type": "uncond",
            "sample_id": 0,
            "batch_size": 1,
            "debug": True,
            "debug_num_samples": 1,
            "num_seeds": 1,
            "use_db_dataset": False,
            "ckpt_filter_substring": None,
            "sampling": {"name": "random", "temperature": 1.0},
            "generator": {"retrieval_backbone": "saliency"},
        }
    )
    inference_single_data.main.__wrapped__(test_cfg)


def test_inference_single_data_use_db_dataset(
    tmp_path: Path, dataset_dir: str, features, job_dir_with_ckpt_retrieval_aug
) -> None:
    job_dir = job_dir_with_ckpt_retrieval_aug
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    _prepare_retrieval_cache(cache_dir, dataset_dir, split="train", top_k=32)

    import ralf.train.helpers.retrieval_dataset_wrapper as rdw

    old_cache_dir = rdw.CACHE_DIR
    old_precomputed = rdw.PRECOMPUTED_WEIGHT_DIR
    rdw.CACHE_DIR = str(cache_dir)
    rdw.PRECOMPUTED_WEIGHT_DIR = str(cache_dir)

    test_cfg = OmegaConf.create(
        {
            "job_dir": str(job_dir),
            "dataset_path": dataset_dir,
            "result_dir": str(tmp_path / "results_single"),
            "test_split": "train",
            "cond_type": "uncond",
            "sample_id": 0,
            "batch_size": 1,
            "debug": True,
            "debug_num_samples": 0,
            "num_seeds": 1,
            "use_db_dataset": True,
            "ckpt_filter_substring": None,
            "sampling": {"name": "random", "temperature": 1.0},
            "generator": {"retrieval_backbone": "saliency"},
        }
    )
    try:
        inference_single_data.main.__wrapped__(test_cfg)
    finally:
        rdw.CACHE_DIR = old_cache_dir
        rdw.PRECOMPUTED_WEIGHT_DIR = old_precomputed


def test_inference_single_data_read_image(tmp_path: Path) -> None:
    img = Image.fromarray(np.zeros((10, 10, 3), dtype=np.uint8))
    img_path = tmp_path / "img.png"
    img.save(img_path)
    tensor = inference_single_data.read_image(str(img_path))
    assert tensor.shape[0] == 1
    assert tensor.shape[1] == 3


def test_inference_single_data_pku_path_relation_return(
    tmp_path: Path, dataset_dir: str, job_dir_with_ckpt
) -> None:
    pku_path = tmp_path / "pku_dataset"
    if not pku_path.exists():
        os.symlink(dataset_dir, pku_path)
    test_cfg = OmegaConf.create(
        {
            "job_dir": str(job_dir_with_ckpt),
            "dataset_path": str(pku_path),
            "result_dir": str(tmp_path / "results_pku"),
            "test_split": "val",
            "cond_type": "relation",
            "sample_id": 0,
            "batch_size": 1,
            "debug": True,
            "debug_num_samples": 0,
            "num_seeds": 1,
            "use_db_dataset": False,
            "ckpt_filter_substring": None,
            "sampling": {"name": "random", "temperature": 1.0},
            "generator": {"retrieval_backbone": "saliency"},
        }
    )
    inference_single_data.main.__wrapped__(test_cfg)


def test_inference_single_data_retriever_early_return(
    tmp_path: Path, dataset_dir: str
) -> None:
    job_dir = tmp_path / "job_retriever_early"
    _write_train_config(
        job_dir,
        dataset_dir,
        generator_override={
            "_target_": "ralf.train.models.retrieval.retriever.Retriever",
            "_partial_": True,
            "retrieval_backbone": "saliency",
            "top_k": 1,
            "save_index": False,
        },
        max_seq_length=-1,
    )
    test_cfg = OmegaConf.create(
        {
            "job_dir": str(job_dir),
            "dataset_path": dataset_dir,
            "result_dir": str(tmp_path / "results_retriever_early"),
            "test_split": "val",
            "cond_type": "relation",
            "sample_id": 0,
            "batch_size": 1,
            "debug": True,
            "debug_num_samples": 0,
            "num_seeds": 1,
            "use_db_dataset": False,
            "ckpt_filter_substring": None,
            "sampling": {"name": "random", "temperature": 1.0},
            "generator": {"retrieval_backbone": "saliency"},
        }
    )
    inference_single_data.main.__wrapped__(test_cfg)


def test_inference_single_data_cgl_aux_mismatch(
    tmp_path: Path, dataset_dir: str, features
) -> None:
    job_dir = tmp_path / "job_single_cgl_mismatch"
    _prepare_job(
        job_dir,
        dataset_dir,
        features,
        generator_override={
            "_target_": "ralf.train.models.cgl.CGLGenerator",
            "_partial_": True,
            "d_model": 256,
            "auxilary_task": "c",
        },
        tokenization=False,
    )
    test_cfg = OmegaConf.create(
        {
            "job_dir": str(job_dir),
            "dataset_path": dataset_dir,
            "result_dir": str(tmp_path / "results_single_cgl_mismatch"),
            "test_split": "train",
            "cond_type": "uncond",
            "sample_id": 0,
            "batch_size": 1,
            "debug": True,
            "debug_num_samples": 0,
            "num_seeds": 1,
            "use_db_dataset": False,
            "ckpt_filter_substring": None,
            "sampling": {"name": "random", "temperature": 1.0},
            "generator": {"retrieval_backbone": "saliency"},
        }
    )
    inference_single_data.main.__wrapped__(test_cfg)


def test_inference_single_data_val_relation_returns(
    tmp_path: Path, dataset_dir: str, job_dir_with_ckpt
) -> None:
    job_dir = job_dir_with_ckpt
    test_cfg = OmegaConf.create(
        {
            "job_dir": str(job_dir),
            "dataset_path": dataset_dir,
            "result_dir": str(tmp_path / "results_val_relation"),
            "test_split": "val",
            "cond_type": "relation",
            "sample_id": 0,
            "batch_size": 1,
            "debug": True,
            "debug_num_samples": 0,
            "num_seeds": 1,
            "use_db_dataset": False,
            "ckpt_filter_substring": None,
            "sampling": {"name": "random", "temperature": 1.0},
            "generator": {"retrieval_backbone": "saliency"},
        }
    )
    inference_single_data.main.__wrapped__(test_cfg)


def test_inference_single_data_aux_task_mismatch(
    tmp_path: Path, dataset_dir: str, features
) -> None:
    job_dir = tmp_path / "job_aux_mismatch"
    _prepare_job(
        job_dir,
        dataset_dir,
        features,
        generator_target="ralf.train.models.autoreg.ConcateAuxilaryTaskAutoreg",
        generator_extra={
            "auxilary_task": "c",
            "use_multitask": False,
            "use_flag_embedding": True,
            "global_task_embedding": False,
        },
    )
    test_cfg = OmegaConf.create(
        {
            "job_dir": str(job_dir),
            "dataset_path": dataset_dir,
            "result_dir": str(tmp_path / "results_aux_mismatch"),
            "test_split": "train",
            "cond_type": "uncond",
            "sample_id": 0,
            "batch_size": 1,
            "debug": True,
            "debug_num_samples": 0,
            "num_seeds": 1,
            "use_db_dataset": False,
            "ckpt_filter_substring": None,
            "sampling": {"name": "random", "temperature": 1.0},
            "generator": {"retrieval_backbone": "saliency"},
        }
    )
    inference_single_data.main.__wrapped__(test_cfg)


def test_inference_single_data_refinement_aux_task(
    tmp_path: Path, dataset_dir: str, features
) -> None:
    job_dir = tmp_path / "job_refinement"
    _prepare_job(
        job_dir,
        dataset_dir,
        features,
        generator_target="ralf.train.models.autoreg.ConcateAuxilaryTaskAutoreg",
        generator_extra={
            "auxilary_task": "refinement",
            "use_multitask": False,
            "use_flag_embedding": True,
            "global_task_embedding": False,
        },
    )
    test_cfg = OmegaConf.create(
        {
            "job_dir": str(job_dir),
            "dataset_path": dataset_dir,
            "result_dir": str(tmp_path / "results_refinement"),
            "test_split": "train",
            "cond_type": "refinement",
            "sample_id": 0,
            "batch_size": 1,
            "debug": True,
            "debug_num_samples": 0,
            "num_seeds": 1,
            "use_db_dataset": False,
            "ckpt_filter_substring": None,
            "sampling": {"name": "random", "temperature": 1.0},
            "refine_lambda": 1.0,
            "refine_mode": "add",
            "refine_offset_ratio": 0.5,
            "lambda": 0.2,
            "generator": {"retrieval_backbone": "saliency"},
        }
    )
    inference_single_data.main.__wrapped__(test_cfg)


def test_inference_single_data_cgl_no_tokenization(
    tmp_path: Path, dataset_dir: str, features
) -> None:
    job_dir = tmp_path / "job_single_cgl"
    _prepare_job(
        job_dir,
        dataset_dir,
        features,
        generator_override={
            "_target_": "ralf.train.models.cgl.CGLGenerator",
            "_partial_": True,
            "d_model": 256,
            "auxilary_task": None,
        },
        tokenization=False,
    )
    test_cfg = OmegaConf.create(
        {
            "job_dir": str(job_dir),
            "dataset_path": dataset_dir,
            "result_dir": str(tmp_path / "results_single_cgl"),
            "test_split": "train",
            "cond_type": None,
            "sample_id": 0,
            "batch_size": 1,
            "debug": True,
            "debug_num_samples": 0,
            "num_seeds": 1,
            "use_db_dataset": False,
            "ckpt_filter_substring": None,
            "sampling": {"name": "random", "temperature": 1.0},
            "generator": {"retrieval_backbone": "saliency"},
        }
    )
    inference_single_data.main.__wrapped__(test_cfg)


def test_inference_unanno_main(
    tmp_path: Path, dataset_dir: str, job_dir_with_ckpt
) -> None:
    job_dir = job_dir_with_ckpt

    test_cfg = OmegaConf.create(
        {
            "job_dir": str(job_dir),
            "dataset_path": dataset_dir,
            "result_dir": str(tmp_path / "results"),
            "test_split": "with_no_annotation",
            "cond_type": "uncond",
            "batch_size": 1,
            "debug": True,
            "debug_num_samples": 0,
            "num_seeds": 1,
            "use_db_dataset": False,
            "ckpt_filter_substring": None,
            "sampling": {"name": "random", "temperature": 1.0},
            "generator": {"retrieval_backbone": "saliency"},
            "no_anno_dataset_name": "cgl",
            "repeat_retrieved_layouts": False,
            "save_vis": False,
            "dynamic_topk": None,
        }
    )
    inference_unanno.main.__wrapped__(test_cfg)


def test_inference_unanno_read_image(tmp_path: Path) -> None:
    img = Image.fromarray(np.zeros((10, 10, 3), dtype=np.uint8))
    img_path = tmp_path / "img_unanno.png"
    img.save(img_path)
    tensor = inference_unanno.read_image(str(img_path))
    assert tensor.shape[0] == 1
    assert tensor.shape[1] == 3


def test_inference_unanno_dynamic_topk_and_save_vis(
    tmp_path: Path, dataset_dir: str, job_dir_with_ckpt_retrieval_aug
) -> None:
    job_dir = job_dir_with_ckpt_retrieval_aug
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    _prepare_retrieval_cache(
        cache_dir, dataset_dir, split="with_no_annotation", top_k=32
    )

    import ralf.train.helpers.retrieval_dataset_wrapper as rdw

    old_cache_dir = rdw.CACHE_DIR
    old_precomputed = rdw.PRECOMPUTED_WEIGHT_DIR
    rdw.CACHE_DIR = str(cache_dir)
    rdw.PRECOMPUTED_WEIGHT_DIR = str(cache_dir)

    test_cfg = OmegaConf.create(
        {
            "job_dir": str(job_dir),
            "dataset_path": dataset_dir,
            "result_dir": str(tmp_path / "results_unanno"),
            "test_split": "with_no_annotation",
            "cond_type": "uncond",
            "batch_size": 1,
            "debug": True,
            "debug_num_samples": 0,
            "num_seeds": 1,
            "use_db_dataset": True,
            "ckpt_filter_substring": None,
            "sampling": {"name": "random", "temperature": 1.0},
            "generator": {"retrieval_backbone": "saliency"},
            "no_anno_dataset_name": "cgl",
            "repeat_retrieved_layouts": True,
            "save_vis": True,
            "dynamic_topk": 1,
        }
    )
    try:
        inference_unanno.main.__wrapped__(test_cfg)
    finally:
        rdw.CACHE_DIR = old_cache_dir
        rdw.PRECOMPUTED_WEIGHT_DIR = old_precomputed


def test_inference_unanno_retriever_model(
    tmp_path: Path, dataset_dir: str, dataset, features
) -> None:
    ds_dict, _ = dataset
    job_dir = tmp_path / "job_retriever_unanno"
    _prepare_job(
        job_dir,
        dataset_dir,
        features,
        generator_override={
            "_target_": "ralf.train.models.retrieval.retriever.Retriever",
            "_partial_": True,
            "retrieval_backbone": "saliency",
            "top_k": 1,
            "save_index": False,
            "cache_dir": str(tmp_path / "retriever_cache"),
        },
        db_dataset=ds_dict["train"],
    )

    cache_dir = tmp_path / "cache_retriever_unanno"
    cache_dir.mkdir(parents=True, exist_ok=True)
    _prepare_retrieval_cache(
        cache_dir, dataset_dir, split="with_no_annotation", top_k=32
    )

    import ralf.train.helpers.retrieval_dataset_wrapper as rdw

    old_cache_dir = rdw.CACHE_DIR
    old_precomputed = rdw.PRECOMPUTED_WEIGHT_DIR
    rdw.CACHE_DIR = str(cache_dir)
    rdw.PRECOMPUTED_WEIGHT_DIR = str(cache_dir)

    test_cfg = OmegaConf.create(
        {
            "job_dir": str(job_dir),
            "dataset_path": dataset_dir,
            "result_dir": str(tmp_path / "results_retriever_unanno"),
            "test_split": "with_no_annotation",
            "cond_type": "uncond",
            "batch_size": 1,
            "debug": True,
            "debug_num_samples": 0,
            "num_seeds": 1,
            "use_db_dataset": False,
            "ckpt_filter_substring": None,
            "sampling": {"name": "random", "temperature": 1.0},
            "generator": {"retrieval_backbone": "saliency"},
            "no_anno_dataset_name": "cgl",
            "repeat_retrieved_layouts": False,
            "save_vis": False,
            "dynamic_topk": None,
        }
    )
    try:
        inference_unanno.main.__wrapped__(test_cfg)
    finally:
        rdw.CACHE_DIR = old_cache_dir
        rdw.PRECOMPUTED_WEIGHT_DIR = old_precomputed


def test_inference_main_random_retrieval(
    tmp_path: Path, dataset_dir: str, job_dir_with_ckpt_retrieval_aug
) -> None:
    job_dir = job_dir_with_ckpt_retrieval_aug
    result_dir = tmp_path / "results_random"
    test_cfg = OmegaConf.create(
        {
            "job_dir": str(job_dir),
            "dataset_path": dataset_dir,
            "result_dir": str(result_dir),
            "test_split": "train",
            "cond_type": "uncond",
            "batch_size": 1,
            "debug": True,
            "debug_num_samples": 0,
            "num_seeds": 1,
            "use_db_dataset": False,
            "ckpt_filter_substring": None,
            "sampling": {"name": "random", "temperature": 1.0},
            "generator": {"retrieval_backbone": "saliency"},
            "preload_data": False,
        }
    )
    inference.main.__wrapped__(test_cfg)


def test_inference_main_copy_generator(
    tmp_path: Path, dataset_dir: str, dataset, features
) -> None:
    ds_dict, _ = dataset
    job_dir = tmp_path / "job_retriever"
    _prepare_job(
        job_dir,
        dataset_dir,
        features,
        generator_override={
            "_target_": "ralf.train.models.retrieval.retriever.Retriever",
            "_partial_": True,
            "retrieval_backbone": "merge",
            "top_k": 1,
        },
        db_dataset=ds_dict["train"],
    )
    cache_dir = tmp_path / "cache_retriever"
    cache_dir.mkdir(parents=True, exist_ok=True)
    _prepare_retrieval_cache(
        cache_dir, dataset_dir, split="train", top_k=32, retrieval_backbone="merge"
    )

    import ralf.train.helpers.retrieval_dataset_wrapper as rdw

    old_cache_dir = rdw.CACHE_DIR
    old_precomputed = rdw.PRECOMPUTED_WEIGHT_DIR
    rdw.CACHE_DIR = str(cache_dir)
    rdw.PRECOMPUTED_WEIGHT_DIR = str(cache_dir)

    test_cfg = OmegaConf.create(
        {
            "job_dir": str(job_dir),
            "dataset_path": dataset_dir,
            "result_dir": str(tmp_path / "results_copy"),
            "test_split": "train",
            "cond_type": "uncond",
            "batch_size": 1,
            "debug": True,
            "debug_num_samples": 0,
            "num_seeds": 1,
            "use_db_dataset": False,
            "ckpt_filter_substring": None,
            "sampling": {"name": "random", "temperature": 1.0},
            "generator": {"retrieval_backbone": "merge"},
            "preload_data": False,
        }
    )
    try:
        inference.main.__wrapped__(test_cfg)
    finally:
        rdw.CACHE_DIR = old_cache_dir
        rdw.PRECOMPUTED_WEIGHT_DIR = old_precomputed


def test_inference_main_cgl_no_tokenization(
    tmp_path: Path, dataset_dir: str, features
) -> None:
    job_dir = tmp_path / "job_cgl"
    _prepare_job(
        job_dir,
        dataset_dir,
        features,
        generator_override={
            "_target_": "ralf.train.models.cgl.CGLGenerator",
            "_partial_": True,
            "d_model": 256,
            "auxilary_task": None,
        },
        tokenization=False,
    )
    result_dir = tmp_path / "results_cgl"
    test_cfg = OmegaConf.create(
        {
            "job_dir": str(job_dir),
            "dataset_path": dataset_dir,
            "result_dir": str(result_dir),
            "test_split": "train",
            "cond_type": None,
            "batch_size": 1,
            "debug": True,
            "debug_num_samples": 1,
            "num_seeds": 1,
            "use_db_dataset": False,
            "ckpt_filter_substring": None,
            "sampling": {"name": "random", "temperature": 1.0},
            "generator": {"retrieval_backbone": "saliency"},
            "preload_data": False,
        }
    )
    inference.main.__wrapped__(test_cfg)


def test_inference_main_relation_val_returns(
    tmp_path: Path, dataset_dir: str, job_dir_with_ckpt
) -> None:
    job_dir = job_dir_with_ckpt
    test_cfg = OmegaConf.create(
        {
            "job_dir": str(job_dir),
            "dataset_path": dataset_dir,
            "result_dir": str(tmp_path / "results_relation_val"),
            "test_split": "val",
            "cond_type": "relation",
            "batch_size": 1,
            "debug": True,
            "debug_num_samples": 0,
            "num_seeds": 1,
            "use_db_dataset": False,
            "ckpt_filter_substring": None,
            "sampling": {"name": "random", "temperature": 1.0},
            "generator": {"retrieval_backbone": "saliency"},
            "preload_data": False,
        }
    )
    inference.main.__wrapped__(test_cfg)


def test_inference_main_aux_task_mismatch(
    tmp_path: Path, dataset_dir: str, features
) -> None:
    job_dir = tmp_path / "job_main_aux_mismatch"
    _prepare_job(
        job_dir,
        dataset_dir,
        features,
        generator_target="ralf.train.models.autoreg.ConcateAuxilaryTaskAutoreg",
        generator_extra={
            "auxilary_task": "c",
            "use_multitask": False,
            "use_flag_embedding": True,
            "global_task_embedding": False,
        },
    )
    test_cfg = OmegaConf.create(
        {
            "job_dir": str(job_dir),
            "dataset_path": dataset_dir,
            "result_dir": str(tmp_path / "results_main_aux_mismatch"),
            "test_split": "train",
            "cond_type": "uncond",
            "batch_size": 1,
            "debug": True,
            "debug_num_samples": 0,
            "num_seeds": 1,
            "use_db_dataset": False,
            "ckpt_filter_substring": None,
            "sampling": {"name": "random", "temperature": 1.0},
            "generator": {"retrieval_backbone": "saliency"},
            "preload_data": False,
        }
    )
    inference.main.__wrapped__(test_cfg)
