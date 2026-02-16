import os
import runpy

import pytest
import torch
from omegaconf import OmegaConf

from ralf.train.config import TokenizerConfig
from ralf.train.helpers.layout_tokenizer import LayoutSequenceTokenizer
from ralf.train.helpers.task import get_condition
from ralf.train.models.common.base_model import ConditionalInputsForDiscreteLayout
from ralf.train.models.layoutdm import LayoutDM, RetrievalAugmentedLayoutDM
from ralf.train.models.maskgit import MaskGIT, _mask_schedule_func


def test_mask_schedule_func() -> None:
    ratio = torch.tensor([0.2, 0.8])
    out = _mask_schedule_func(ratio, schedule="linear")
    out_cos = _mask_schedule_func(ratio, schedule="cosine")
    out_sq = _mask_schedule_func(ratio, schedule="square")
    assert out.shape == ratio.shape
    assert out_cos.shape == ratio.shape
    assert out_sq.shape == ratio.shape

    with pytest.raises(NotImplementedError):
        _mask_schedule_func(ratio, schedule="unknown")


def _mask_tokenizer(features) -> LayoutSequenceTokenizer:
    cfg = TokenizerConfig()
    cfg_dict = cfg.__dict__.copy()
    cfg_dict.pop("special_tokens", None)
    return LayoutSequenceTokenizer(
        label_feature=features["label"].feature,
        max_seq_length=10,
        special_tokens=["pad", "mask"],
        **cfg_dict,
    )


def test_maskgit_train_loss(small_batch, features) -> None:
    tokenizer = _mask_tokenizer(features)
    model = MaskGIT(features=features, tokenizer=tokenizer)
    model.eval()
    inputs, targets = model.preprocess(small_batch)
    with torch.no_grad():
        outputs, losses = model.train_loss(inputs, targets)
    assert "nll_loss" in losses
    assert outputs["logits"].shape[0] == small_batch["image"].shape[0]


def test_layoutdm_train_loss(small_batch, features) -> None:
    tokenizer = _mask_tokenizer(features)
    model = LayoutDM(features=features, tokenizer=tokenizer, q_type="default")
    model.eval()
    inputs, targets = model.preprocess(small_batch)
    with torch.no_grad():
        outputs, losses = model.train_loss(inputs, targets)
    assert "kl_loss" in losses
    assert outputs["logits"].shape[0] == small_batch["image"].shape[0]


def test_maskgit_and_layoutdm_sample(small_batch, features) -> None:
    tokenizer = _mask_tokenizer(features)
    maskgit = MaskGIT(features=features, tokenizer=tokenizer)
    maskgit.eval()
    cond, _ = get_condition(small_batch, "partial", tokenizer)
    sampling_cfg = OmegaConf.create(
        {"name": "random", "temperature": 1.0, "num_timesteps": 2}
    )
    output = maskgit.sample(cond=cond, sampling_cfg=sampling_cfg)
    assert "label" in output

    layoutdm = LayoutDM(features=features, tokenizer=tokenizer, q_type="default")
    layoutdm.eval()
    cond_uncond, _ = get_condition(small_batch, "uncond", tokenizer)
    sampling_cfg_dm = OmegaConf.create(
        {"name": "random", "temperature": 1.0, "num_timesteps": 2}
    )
    output_dm = layoutdm.sample(cond=cond_uncond, sampling_cfg=sampling_cfg_dm)
    assert "label" in output_dm


def test_maskgit_sample_without_padding(
    small_batch, features, layout_tokenizer
) -> None:
    tokenizer = _mask_tokenizer(features)
    model = MaskGIT(
        features=features,
        tokenizer=tokenizer,
        mask_schedule="cosine",
        use_padding_as_vocab=False,
        use_gumbel_noise=False,
    )
    model.eval()
    image = torch.cat([small_batch["image"], small_batch["saliency"]], dim=1)
    cond = ConditionalInputsForDiscreteLayout(
        image=image, id=None, seq=None, mask=None, task="c"
    )
    sampling_cfg = OmegaConf.create(
        {"name": "random", "temperature": 1.0, "num_timesteps": 2}
    )
    output = model.sample(cond=cond, sampling_cfg=sampling_cfg)
    assert "label" in output


def test_maskgit_sample_with_seq_and_pad_weight(small_batch, features) -> None:
    tokenizer = _mask_tokenizer(features)
    model = MaskGIT(
        features=features,
        tokenizer=tokenizer,
        use_padding_as_vocab=True,
        pad_weight=0.5,
        use_gumbel_noise=True,
    )
    model.eval()
    cond, _ = get_condition(small_batch, "c", tokenizer)
    sampling_cfg = OmegaConf.create(
        {"name": "random", "temperature": 1.0, "num_timesteps": 1}
    )
    output = model.sample(cond=cond, sampling_cfg=sampling_cfg)
    assert "label" in output


def test_maskgit_sample_with_seq_no_padding(small_batch, features) -> None:
    tokenizer = _mask_tokenizer(features)
    model = MaskGIT(
        features=features,
        tokenizer=tokenizer,
        use_padding_as_vocab=False,
        use_gumbel_noise=False,
    )
    model.eval()
    cond, _ = get_condition(small_batch, "c", tokenizer)
    sampling_cfg = OmegaConf.create(
        {"name": "random", "temperature": 1.0, "num_timesteps": 1}
    )
    output = model.sample(cond=cond, sampling_cfg=sampling_cfg)
    assert "label" in output


def test_maskgit_main_plot(tmp_path) -> None:
    cwd = os.getcwd()
    os.environ["MPLBACKEND"] = "Agg"
    os.chdir(tmp_path)
    try:
        runpy.run_module("ralf.train.models.maskgit", run_name="__main__")
        assert (tmp_path / "mask_schedule.pdf").exists()
    finally:
        os.chdir(cwd)


def test_layoutdm_refinement_sample(small_batch, features) -> None:
    tokenizer = _mask_tokenizer(features)
    model = LayoutDM(features=features, tokenizer=tokenizer, q_type="default")
    model.eval()
    cond, _ = get_condition(small_batch, "refinement", tokenizer)
    cond.seq_observed = cond.seq
    sampling_cfg = OmegaConf.create(
        {
            "name": "random",
            "temperature": 1.0,
            "num_timesteps": 2,
            "refine_lambda": 1.0,
            "refine_mode": "uniform",
            "refine_offset_ratio": 0.1,
        }
    )
    output = model.sample(cond=cond, sampling_cfg=sampling_cfg)
    assert "label" in output


def test_layoutdm_retrieval_augmented_preprocess(
    small_batch, features, layout_tokenizer, dataset
) -> None:
    ds_dict, _ = dataset
    tokenizer = _mask_tokenizer(features)
    model = RetrievalAugmentedLayoutDM(
        features=features,
        tokenizer=tokenizer,
        q_type="default",
        db_dataset=ds_dict["train"],
        top_k=1,
        dataset_name="cgl",
        retrieval_backbone="saliency",
        random_retrieval=True,
        saliency_k=1,
        use_reference_image=False,
        max_seq_length=10,
    )
    model.eval()
    retrieved = {
        "image": small_batch["image"].unsqueeze(1),
        "saliency": small_batch["saliency"].unsqueeze(1),
        "center_x": small_batch["center_x"].unsqueeze(1),
        "center_y": small_batch["center_y"].unsqueeze(1),
        "width": small_batch["width"].unsqueeze(1),
        "height": small_batch["height"].unsqueeze(1),
        "label": small_batch["label"].unsqueeze(1),
        "mask": small_batch["mask"].unsqueeze(1),
    }
    inputs, targets = model.preprocess({**small_batch, "retrieved": retrieved})
    assert "retrieved" in inputs
    _ = model.train_loss(inputs, targets)
