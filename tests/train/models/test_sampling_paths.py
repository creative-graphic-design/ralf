import torch
from omegaconf import OmegaConf

from ralf.train.config import TokenizerConfig
from ralf.train.helpers.layout_tokenizer import LayoutSequenceTokenizer
from ralf.train.models.common.base_model import ConditionalInputsForDiscreteLayout
from ralf.train.models.layoutdm import LayoutDM
from ralf.train.models.maskgit import MaskGIT


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


def test_maskgit_sample_path(small_batch, features) -> None:
    tokenizer = _mask_tokenizer(features)
    model = MaskGIT(features=features, tokenizer=tokenizer)
    model.eval()
    encoded = tokenizer.encode(small_batch)
    image = torch.cat([small_batch["image"], small_batch["saliency"]], dim=1)
    cond = ConditionalInputsForDiscreteLayout(
        image=image, id=None, seq=encoded["seq"], mask=encoded["mask"]
    )
    sampling_cfg = OmegaConf.create(
        {"num_timesteps": 2, "name": "random", "temperature": 1.0}
    )
    out = model.sample(cond=cond, sampling_cfg=sampling_cfg)
    assert "label" in out and "mask" in out


def test_layoutdm_sample_path(small_batch, features) -> None:
    tokenizer = _mask_tokenizer(features)
    model = LayoutDM(features=features, tokenizer=tokenizer, q_type="default")
    model.eval()
    image = torch.cat([small_batch["image"], small_batch["saliency"]], dim=1)
    cond = ConditionalInputsForDiscreteLayout(image=image, id=None)
    sampling_cfg = OmegaConf.create(
        {"num_timesteps": 2, "name": "random", "temperature": 1.0}
    )
    out = model.sample(cond=cond, sampling_cfg=sampling_cfg, cond_type="uncond")
    assert "label" in out and "mask" in out
