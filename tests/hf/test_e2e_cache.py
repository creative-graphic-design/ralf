import os

import pytest
import torch
from omegaconf import OmegaConf, open_dict

from ralf.train.data import collate_fn, get_dataset
from ralf.train.global_variables import PRECOMPUTED_WEIGHT_DIR
from ralf.train.helpers.layout_tokenizer import init_layout_tokenizer
from ralf.train.models.autoreg import ConcateAuxilaryTaskAutoreg
from ralf.transformers.autoreg import RalfAutoregConfig, RalfAutoregModel

CACHE_PATH = os.environ.get(
    "RALF_CACHE_DIR", "/cvl-gen-vsfs2/public_dataset/layout/RALF_cache"
)


def _find_autoreg_job_dir(cache_path: str) -> tuple[str, dict] | None:
    training_logs = os.path.join(cache_path, "training_logs")
    if not os.path.isdir(training_logs):
        return None
    for entry in sorted(os.listdir(training_logs)):
        job_dir = os.path.join(training_logs, entry)
        config_path = os.path.join(job_dir, "config.yaml")
        ckpt_path = os.path.join(job_dir, "gen_final_model.pt")
        if not (
            os.path.isdir(job_dir)
            and os.path.exists(config_path)
            and os.path.exists(ckpt_path)
        ):
            continue
        train_cfg = OmegaConf.load(config_path)
        generator_target = str(train_cfg.generator._target_)
        if (
            "Autoreg" in generator_target
            and "RetrievalAugmented" not in generator_target
        ):
            return job_dir, train_cfg
    return None


def _ensure_weights(cache_path: str) -> bool:
    weight_path = os.path.join(
        os.environ.get("RALF_PRECOMPUTED_WEIGHT_DIR", PRECOMPUTED_WEIGHT_DIR),
        "resnet50_a1_0-14fe96d1.pth",
    )
    return os.path.exists(weight_path)


def _make_autoreg_config(train_cfg, tokenizer) -> RalfAutoregConfig:
    gen_cfg = train_cfg.generator
    return RalfAutoregConfig(
        d_model=getattr(gen_cfg, "d_model", 256),
        encoder_pos_emb=getattr(gen_cfg, "encoder_pos_emb", "sine"),
        decoder_pos_emb=getattr(gen_cfg, "decoder_pos_emb", "layout"),
        weight_init=getattr(gen_cfg, "weight_init", False),
        shared_embedding=getattr(gen_cfg, "shared_embedding", False),
        decoder_num_layers=getattr(gen_cfg, "decoder_num_layers", 6),
        decoder_d_model=getattr(gen_cfg, "decoder_d_model", 256),
        auxilary_task=getattr(gen_cfg, "auxilary_task", "uncond"),
        use_flag_embedding=getattr(gen_cfg, "use_flag_embedding", True),
        use_multitask=getattr(gen_cfg, "use_multitask", False),
        relation_size=getattr(gen_cfg, "RELATION_SIZE", 10),
        global_task_embedding=getattr(gen_cfg, "global_task_embedding", False),
        max_position_embeddings=tokenizer.max_token_length,
    )


def test_e2e_autoreg_cache_parity(tmp_path) -> None:
    if not os.path.exists(CACHE_PATH):
        pytest.skip("RALF cache not available")
    job = _find_autoreg_job_dir(CACHE_PATH)
    if job is None:
        pytest.skip("No autoreg job dir found")
    if not _ensure_weights(CACHE_PATH):
        pytest.skip("ResNet50 weights not found")

    job_dir, train_cfg = job

    os.environ["RALF_CACHE_DIR"] = CACHE_PATH
    os.environ["RALF_DATASET_DIR"] = os.path.join(CACHE_PATH, "dataset")
    os.environ["RALF_PRECOMPUTED_WEIGHT_DIR"] = os.path.join(
        CACHE_PATH, "PRECOMPUTED_WEIGHT_DIR"
    )

    with open_dict(train_cfg):
        data_dir = train_cfg.dataset.data_dir
        dataset_root = os.environ["RALF_DATASET_DIR"]
        dataset_name = getattr(train_cfg.dataset, "name", None)
        candidate = (
            os.path.join(dataset_root, dataset_name) if dataset_name else dataset_root
        )
        if not data_dir or not os.path.exists(data_dir):
            train_cfg.dataset.data_dir = candidate

    dataset, features = get_dataset(
        dataset_cfg=train_cfg.dataset,
        transforms=list(train_cfg.data.transforms),
    )

    tokenizer = init_layout_tokenizer(
        tokenizer_cfg=train_cfg.tokenizer,
        dataset_cfg=train_cfg.dataset,
        label_feature=features["label"].feature,
    )

    gen_cfg = train_cfg.generator
    original = ConcateAuxilaryTaskAutoreg(
        features=features,
        tokenizer=tokenizer,
        d_model=getattr(gen_cfg, "d_model", 256),
        encoder_pos_emb=getattr(gen_cfg, "encoder_pos_emb", "sine"),
        decoder_pos_emb=getattr(gen_cfg, "decoder_pos_emb", "layout"),
        weight_init=getattr(gen_cfg, "weight_init", False),
        shared_embedding=getattr(gen_cfg, "shared_embedding", False),
        decoder_num_layers=getattr(gen_cfg, "decoder_num_layers", 6),
        decoder_d_model=getattr(gen_cfg, "decoder_d_model", 256),
        auxilary_task=getattr(gen_cfg, "auxilary_task", "uncond"),
        use_flag_embedding=getattr(gen_cfg, "use_flag_embedding", True),
        use_multitask=getattr(gen_cfg, "use_multitask", False),
        RELATION_SIZE=getattr(gen_cfg, "RELATION_SIZE", 10),
        global_task_embedding=getattr(gen_cfg, "global_task_embedding", False),
    )
    ckpt_path = os.path.join(job_dir, "gen_final_model.pt")
    original.load_state_dict(torch.load(ckpt_path, map_location="cpu"))
    original.eval()

    config = _make_autoreg_config(train_cfg, tokenizer)
    hf_model = RalfAutoregModel.from_pretrained(
        job_dir, config=config, tokenizer=tokenizer
    )
    hf_model.eval()

    sample = dataset["test"][0]
    batch = collate_fn([sample], max_seq_length=train_cfg.dataset.max_seq_length)
    inputs, _ = original.preprocess(batch)

    with torch.no_grad():
        out_orig = original(inputs)
        out_hf = hf_model(**inputs)

    assert torch.allclose(out_orig["logits"], out_hf["logits"])

    hf_model.save_pretrained(tmp_path)
    reloaded = RalfAutoregModel.from_pretrained(tmp_path, tokenizer=tokenizer)
    reloaded.eval()
    with torch.no_grad():
        out_reload = reloaded(**inputs)
    assert torch.allclose(out_orig["logits"], out_reload["logits"])
