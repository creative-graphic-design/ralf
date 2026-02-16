import os

import pytest
import torch
from omegaconf import OmegaConf

from ralf.train.models.common.base_model import (
    BaseModel,
    ConditionalInputsForDiscreteLayout,
    RetrievalAugmentedConditionalInputsForDiscreteLayout,
)


class DummyLayoutDM(BaseModel):
    def __init__(self, tokenizer):
        super().__init__()
        self.tokenizer = tokenizer
        self.linear = torch.nn.Linear(1, 1)

    def forward(self, inputs: dict) -> dict:  # type: ignore
        return inputs

    def sample(self, *args, **kwargs):  # type: ignore
        return {}

    def generator_loss(self, *args, **kwargs):  # type: ignore
        return None

    def preprocess(self, inputs: dict):  # type: ignore
        return inputs, inputs


class DummyLayoutDMWithParams(BaseModel):
    def __init__(self, tokenizer):
        super().__init__()
        self.tokenizer = tokenizer
        self.linear = torch.nn.Linear(4, 4)
        self.embed = torch.nn.Embedding(8, 4)
        self.lstm = torch.nn.LSTM(4, 4, batch_first=True)

    def forward(self, inputs: dict) -> dict:  # type: ignore
        return inputs

    def sample(self, *args, **kwargs):  # type: ignore
        return {}

    def generator_loss(self, *args, **kwargs):  # type: ignore
        return None

    def preprocess(self, inputs: dict):  # type: ignore
        return inputs, inputs


def test_conditional_inputs_duplicate_and_retrieval_post_init() -> None:
    image = torch.zeros(1, 4, 8, 8)
    cond = ConditionalInputsForDiscreteLayout(image=image, id=None)
    cond.duplicate(2)
    assert cond.image.shape[0] == 2

    retrieved = {
        "image": torch.zeros(1, 1, 3, 8, 8),
        "saliency": torch.zeros(1, 1, 1, 8, 8),
    }
    cond_retrieval = RetrievalAugmentedConditionalInputsForDiscreteLayout(
        image=image, id=None, retrieved=retrieved
    )
    assert cond_retrieval.retrieved["image"].shape[2] == 4


def test_aggregate_sampling_config_and_postprocess(layout_tokenizer) -> None:
    model = DummyLayoutDM(tokenizer=layout_tokenizer)

    sampling_cfg = OmegaConf.create({"name": "random", "temperature": 1.0})
    test_cfg = OmegaConf.create(
        {
            "cond_type": "refinement",
            "refine_lambda": 1.0,
            "refine_mode": "add",
            "refine_offset_ratio": 0.5,
            "lambda": 0.2,
            "relation_lambda": 1.0,
            "relation_mode": "add",
            "relation_tau": 1.0,
            "relation_num_update": 1,
        }
    )
    aggregated = model.aggregate_sampling_config(
        sampling_cfg=sampling_cfg, test_cfg=test_cfg
    )
    assert aggregated.num_timesteps == layout_tokenizer.max_token_length
    assert aggregated.refine_lambda == test_cfg.refine_lambda

    logits = torch.zeros(1, layout_tokenizer.max_token_length, layout_tokenizer.N_total)
    outputs = model.postprocess({"logits": logits})
    assert "label" in outputs


def test_optim_groups_and_postprocess_rearrange(layout_tokenizer) -> None:
    model = DummyLayoutDMWithParams(tokenizer=layout_tokenizer)

    groups = model.optim_groups(base_lr=1e-3, weight_decay=0.1)
    assert len(groups) == 2

    groups_custom = model.optim_groups(
        base_lr=1e-3,
        weight_decay=0.1,
        custom_lr={"linear": 1e-4},
        forced_no_weight_decay=["embed.weight"],
    )
    assert len(groups_custom) >= 2

    logits = torch.zeros(1, layout_tokenizer.N_total, layout_tokenizer.max_token_length)
    outputs = model.postprocess({"logits": logits})
    assert "label" in outputs


def test_base_model_main_creates_plot(tmp_path) -> None:
    pytest.importorskip("matplotlib", exc_type=ImportError)
    import ralf.train.models.common.base_model as base_model

    cwd = os.getcwd()
    os.chdir(tmp_path)
    try:
        base_model.main()
    finally:
        os.chdir(cwd)
    assert (tmp_path / "dummy.pdf").exists()
