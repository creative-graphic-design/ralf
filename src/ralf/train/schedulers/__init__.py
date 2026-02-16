from typing import Any

from .dsgan import DSGANScheduler  # type: ignore
from .multi_step_lr import MultiStepLRScheduler  # type: ignore
from .reduce_lr_on_plateau import ReduceLROnPlateauScheduler  # type: ignore
from .void import VoidScheduler  # type: ignore

__all__ = [
    "DSGANScheduler",
    "MultiStepLRScheduler",
    "ReduceLROnPlateauScheduler",
    "VoidScheduler",
    "requires_metrics",
]


def requires_metrics(scheduler: Any) -> bool:
    return isinstance(scheduler, ReduceLROnPlateauScheduler)
