import torch

from ralf.train.schedulers import (
    DSGANScheduler,
    MultiStepLRScheduler,
    ReduceLROnPlateauScheduler,
    VoidScheduler,
)


def _make_optimizer() -> torch.optim.Optimizer:
    return torch.optim.Adam([torch.nn.Parameter(torch.randn(1))], lr=1e-3)


def test_void_scheduler() -> None:
    sched = VoidScheduler(_make_optimizer())
    assert sched is not None


def test_reduce_lr_on_plateau_scheduler() -> None:
    sched = ReduceLROnPlateauScheduler(_make_optimizer())
    sched.step(0.5)


def test_multi_step_lr_scheduler() -> None:
    sched = MultiStepLRScheduler(_make_optimizer(), epochs=10, milestones=[1, 2])
    sched.step()


def test_dsgan_scheduler() -> None:
    sched = DSGANScheduler(_make_optimizer(), epochs=300, network="generator")
    sched.step()
