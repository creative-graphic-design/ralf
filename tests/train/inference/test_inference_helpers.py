from pathlib import Path

import pytest
import torch
from omegaconf import OmegaConf

from ralf.train.inference import (
    _validate_outputs,
    build_network,
    find_checkpoints,
    load_train_cfg,
    render_batch,
    render_input_output,
)
from ralf.train.inference_single_data import (
    _enumerate_meta,
    read_image,
)
from ralf.train.inference_single_data import (
    _validate_outputs as validate_outputs_single,
)
from ralf.train.inference_single_data import (
    build_network as build_network_single,
)
from ralf.train.inference_single_data import (
    find_checkpoints as find_checkpoints_single,
)
from ralf.train.inference_single_data import (
    load_train_cfg as load_train_cfg_single,
)
from ralf.train.inference_single_data import (
    render_batch as render_batch_single,
)
from ralf.train.models.retrieval.retriever import Retriever


def _write_config(
    job_dir: Path,
    dataset_dir: str,
    target: str = "ralf.train.models.autoreg.Autoreg",
) -> None:
    cfg = OmegaConf.create(
        {
            "dataset": {
                "name": "cgl",
                "max_seq_length": 10,
                "data_dir": dataset_dir,
                "data_type": "parquet",
                "path": None,
            },
            "data": {"transforms": ["image", "shuffle"], "tokenization": True},
            "tokenizer": {"num_bin": 128, "geo_quantization": "linear"},
            "generator": {
                "_target_": target,
                "_partial_": True,
                "d_model": 256,
            },
            "sampling": {"name": "random", "temperature": 1.0},
        }
    )
    with (job_dir / "config.yaml").open("w") as f:
        f.write(OmegaConf.to_yaml(cfg))


def test_inference_helpers(tmp_path, features, dataset_dir, small_batch) -> None:
    job_dir = tmp_path / "job"
    job_dir.mkdir(parents=True, exist_ok=True)
    _write_config(job_dir, dataset_dir)
    ckpt_path = job_dir / "model.pt"
    torch.save({"state_dict": {}}, ckpt_path)

    fs, train_cfg, ckpt_dirs = load_train_cfg(str(job_dir))
    assert ckpt_dirs == [str(job_dir)]
    ckpts = find_checkpoints(str(job_dir))
    assert any(str(ckpt_path) in ckpt for ckpt in ckpts)

    model, _ = build_network(
        train_cfg=train_cfg,
        features=features,
        max_seq_length=10,
    )
    assert model is not None

    outputs = {
        "label": small_batch["label"],
        "mask": small_batch["mask"],
        "center_x": small_batch["center_x"],
        "center_y": small_batch["center_y"],
        "width": small_batch["width"],
        "height": small_batch["height"],
        "id": small_batch["id"],
    }
    validated = _validate_outputs(outputs)
    assert isinstance(validated, list)

    render_inputs = {**outputs, "image": small_batch["image"]}
    rendered = render_batch(render_inputs, features)
    assert rendered.ndim == 4

    rendered_io = render_input_output(
        render_inputs,
        outputs,
        features,
        small_batch["image"],
    )
    assert rendered_io.ndim == 4


def test_inference_single_data_helpers(tmp_path, dataset_dir) -> None:
    job_dir = tmp_path / "job_single"
    job_dir.mkdir(parents=True, exist_ok=True)
    _write_config(job_dir, dataset_dir)
    ckpt_path = job_dir / "best.pt"
    torch.save({"state_dict": {}}, ckpt_path)

    fs, train_cfg, ckpt_dirs = load_train_cfg_single(str(job_dir))
    assert ckpt_dirs == [str(job_dir)]
    ckpts = find_checkpoints_single(str(job_dir))
    assert any(str(ckpt_path) in ckpt for ckpt in ckpts)

    # enumerate meta (single seed)
    seed_dir = tmp_path / "0"
    seed_dir.mkdir(parents=True, exist_ok=True)
    _write_config(seed_dir, dataset_dir)
    test_cfg = OmegaConf.create({"job_dir": str(tmp_path)})
    train_cfg2, dirs = _enumerate_meta(test_cfg)
    assert isinstance(dirs, list)

    # smoke read image
    image_path = tmp_path / "sample.png"
    torch.zeros(3, 10, 10).mul(255).byte().numpy()
    from PIL import Image

    Image.new("RGB", (10, 10), color=(0, 0, 0)).save(image_path)
    img = read_image(str(image_path))
    assert img.ndim == 4


def test_load_train_cfg_does_not_rewrite_targets(tmp_path, dataset_dir) -> None:
    job_dir = tmp_path / "job_rewrite"
    job_dir.mkdir(parents=True, exist_ok=True)
    target = "image2layout.train.models.autoreg.Autoreg"
    _write_config(job_dir, dataset_dir, target=target)

    _, train_cfg, _ = load_train_cfg(str(job_dir))
    assert train_cfg.generator._target_ == target

    _, train_cfg_single, _ = load_train_cfg_single(str(job_dir))
    assert train_cfg_single.generator._target_ == target


def test_inference_find_checkpoints_filter(tmp_path, dataset_dir) -> None:
    job_dir = tmp_path / "job_filter"
    job_dir.mkdir(parents=True, exist_ok=True)
    _write_config(job_dir, dataset_dir)
    ckpt_path = job_dir / "model_epoch1.pt"
    torch.save({"state_dict": {}}, ckpt_path)
    ckpts = find_checkpoints(str(job_dir), filter_substring="epoch1")
    assert any("epoch1" in ckpt for ckpt in ckpts)

    ckpt_path_single = job_dir / "single_epoch1.pt"
    torch.save({"state_dict": {}}, ckpt_path_single)
    ckpts_single = find_checkpoints_single(str(job_dir), filter_substring="epoch1")
    assert any("epoch1" in ckpt for ckpt in ckpts_single)


def test_build_network_retriever(features, dataset, dataset_dir) -> None:
    ds_dict, _ = dataset
    train_cfg = OmegaConf.create(
        {
            "dataset": {
                "name": "cgl",
                "max_seq_length": 10,
                "data_dir": dataset_dir,
                "data_type": "parquet",
                "path": None,
            },
            "data": {"transforms": ["image", "shuffle"], "tokenization": True},
            "tokenizer": {"num_bin": 128, "geo_quantization": "linear"},
            "generator": {
                "_target_": "ralf.train.models.retrieval.retriever.Retriever",
                "_partial_": True,
                "retrieval_backbone": "merge",
                "top_k": 1,
            },
        }
    )
    model, _ = build_network(
        train_cfg=train_cfg,
        features=features,
        max_seq_length=10,
        db_dataset=ds_dict["train"],
    )
    assert isinstance(model, Retriever)

    model_single, _ = build_network_single(
        train_cfg=train_cfg,
        features=features,
        max_seq_length=10,
        db_dataset=ds_dict["train"],
    )
    assert isinstance(model_single, Retriever)


def test_inference_helper_error_paths(tmp_path) -> None:
    with pytest.raises(FileNotFoundError):
        load_train_cfg(str(tmp_path / "missing_job"))
    with pytest.raises(FileNotFoundError):
        load_train_cfg_single(str(tmp_path / "missing_job_single"))


def test_inference_single_outputs_and_render(small_batch, features) -> None:
    outputs = {
        "label": small_batch["label"],
        "mask": small_batch["mask"],
        "center_x": small_batch["center_x"],
        "center_y": small_batch["center_y"],
        "width": small_batch["width"],
        "height": small_batch["height"],
        "id": small_batch["id"],
    }
    validated = validate_outputs_single(outputs)
    assert isinstance(validated, list)

    render_inputs = {**outputs, "image": small_batch["image"]}
    rendered = render_batch_single(render_inputs, features)
    assert rendered.ndim == 4
