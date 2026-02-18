import torch

from ralf.train.helpers.distrubuted import DDPWrapper


def test_ddpwrapper_getattr() -> None:
    model = torch.nn.Linear(2, 2)
    ddp = DDPWrapper.__new__(DDPWrapper)
    torch.nn.Module.__init__(ddp)
    ddp.module = model
    ddp._modules = {"module": model}
    ddp._parameters = {}
    ddp._buffers = {}
    assert ddp.weight is model.weight
