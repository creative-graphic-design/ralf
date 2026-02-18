import torch.nn as nn

from ralf.train.models.common.base_model import BaseModel


class DummyModel(BaseModel):
    def __init__(self) -> None:
        super().__init__()
        self.linear = nn.Linear(4, 2)
        self.norm = nn.LayerNorm(2)
        self.conv = nn.Conv2d(1, 1, 1)

    def forward(self, inputs: dict) -> dict:
        return inputs

    def sample(self, *args, **kwargs):  # type: ignore
        return {}

    def generator_loss(self, *args, **kwargs):  # type: ignore
        return None

    def preprocess(self, inputs: dict):  # type: ignore
        return inputs, inputs


def test_base_model_optim_groups() -> None:
    model = DummyModel()
    groups = model.optim_groups(
        base_lr=1e-3,
        weight_decay=0.1,
        custom_lr={"linear": 1e-4},
    )
    assert isinstance(groups, list)
    assert sum(len(g["params"]) for g in groups) > 0
