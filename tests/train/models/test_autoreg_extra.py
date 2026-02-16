import torch
from omegaconf import OmegaConf

from ralf.train.config import TokenizerConfig
from ralf.train.helpers.layout_tokenizer import LayoutSequenceTokenizer
from ralf.train.helpers.task import get_condition
from ralf.train.models.autoreg import (
    ConcateAuxilaryTaskAutoreg,
    SoftTokenAuxilaryTaskAutoreg,
)


def _make_sampling_cfg():
    return OmegaConf.create({"name": "random", "temperature": 1.0})


def test_autoreg_aux_preprocess_and_encode(
    small_batch, features, layout_tokenizer
) -> None:
    model = SoftTokenAuxilaryTaskAutoreg(
        features=features,
        tokenizer=layout_tokenizer,
        auxilary_task="relation",
        use_multitask=False,
    )
    model.eval()
    inputs, targets = model.preprocess(small_batch)
    encoded = model._encode_into_memory(inputs)
    assert "memory" in encoded
    assert "soft_token" in encoded
    assert targets["seq"].shape[0] == small_batch["image"].shape[0]


def test_autoreg_sample_partial_and_relation(small_batch, features) -> None:
    cfg = TokenizerConfig()
    cfg_dict = cfg.__dict__.copy()
    cfg_dict["special_tokens"] = ["pad", "bos", "eos", "mask"]
    tokenizer = LayoutSequenceTokenizer(
        label_feature=features["label"].feature, max_seq_length=10, **cfg_dict
    )
    model = SoftTokenAuxilaryTaskAutoreg(
        features=features,
        tokenizer=tokenizer,
        auxilary_task="relation",
        use_multitask=False,
    )
    model.eval()

    cond_partial, _ = get_condition(
        batch=small_batch, cond_type="partial", tokenizer=tokenizer
    )
    out_partial = model.sample(
        cond=cond_partial,
        sampling_cfg=_make_sampling_cfg(),
        cond_type="partial",
        use_backtrack=False,
    )
    assert "label" in out_partial

    cond_relation, _ = get_condition(
        batch={
            k: (v[:1] if torch.is_tensor(v) else [v[0]])  # type: ignore[index]
            for k, v in small_batch.items()
        },
        cond_type="relation",
        tokenizer=tokenizer,
    )
    out_relation = model.sample(
        cond=cond_relation,
        sampling_cfg=_make_sampling_cfg(),
        cond_type="relation",
        use_backtrack=True,
    )
    assert "label" in out_relation


def test_autoreg_concate_encode(small_batch, features, layout_tokenizer) -> None:
    model = ConcateAuxilaryTaskAutoreg(
        features=features,
        tokenizer=layout_tokenizer,
        auxilary_task="c",
        use_multitask=False,
        global_task_embedding=True,
    )
    model.eval()
    inputs, _ = model.preprocess(small_batch)
    encoded = model._encode_into_memory(inputs)
    assert "memory" in encoded
