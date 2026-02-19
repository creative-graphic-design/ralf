import os
from typing import Any, Optional, Type

import datasets as ds
import torch
from torch import nn
from transformers import PreTrainedModel


def build_features_from_tokenizer(tokenizer: Any) -> ds.Features:
    if hasattr(tokenizer, "label_names"):
        label_feature = ds.ClassLabel(
            num_classes=len(tokenizer.label_names), names=tokenizer.label_names
        )
    elif hasattr(tokenizer, "_label_feature"):
        label_feature = tokenizer._label_feature
    else:
        raise ValueError("tokenizer must expose label_names or _label_feature")
    return ds.Features({"label": ds.Sequence(label_feature)})


def build_features_from_labels(label_names: list[str]) -> ds.Features:
    label_feature = ds.ClassLabel(num_classes=len(label_names), names=label_names)
    return ds.Features({"label": ds.Sequence(label_feature)})


def normalize_state_dict(
    state_dict: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    if not state_dict:
        return state_dict
    if all(k.startswith("module.") for k in state_dict):
        return {k[len("module.") :]: v for k, v in state_dict.items()}
    return state_dict


def add_prefix(
    state_dict: dict[str, torch.Tensor], prefix: str
) -> dict[str, torch.Tensor]:
    return {f"{prefix}{k}": v for k, v in state_dict.items()}


def remove_prefix(
    state_dict: dict[str, torch.Tensor], prefix: str
) -> dict[str, torch.Tensor]:
    if not all(k.startswith(prefix) for k in state_dict):
        return state_dict
    return {k[len(prefix) :]: v for k, v in state_dict.items()}


def load_state_dict_with_fallback(
    model: nn.Module, state_dict: dict[str, torch.Tensor]
) -> None:
    normalized = normalize_state_dict(state_dict)
    candidates = [
        normalized,
        add_prefix(normalized, "model."),
        remove_prefix(normalized, "model."),
    ]
    for candidate in candidates:
        try:
            model.load_state_dict(candidate, strict=True)
            return
        except RuntimeError:
            continue
    model.load_state_dict(normalized, strict=False)


def resolve_checkpoint_path(path: str) -> str:
    if os.path.isdir(path):
        pt = os.path.join(path, "pytorch_model.bin")
        if os.path.exists(pt):
            return pt
        for name in ["gen_final_model.pt", "model.pt"]:
            candidate = os.path.join(path, name)
            if os.path.exists(candidate):
                return candidate
        raise FileNotFoundError(f"No checkpoint found in {path}")
    return path


class RalfPreTrainedModel(PreTrainedModel):
    tokenizer_class: Optional[Type] = None

    def _init_weights(self, module: nn.Module) -> None:
        del module
        return

    @classmethod
    def _load_tokenizer(cls, path: str, tokenizer: Optional[Any]) -> Optional[Any]:
        if tokenizer is not None:
            return tokenizer
        if cls.tokenizer_class is None:
            return None
        if os.path.isdir(path):
            try:
                return cls.tokenizer_class.from_pretrained(path)
            except Exception:
                return None
        return None

    @classmethod
    def from_pretrained(  # type: ignore[override]
        cls,
        pretrained_model_name_or_path: str,
        *model_args: Any,
        tokenizer: Optional[Any] = None,
        **kwargs: Any,
    ):
        tokenizer = cls._load_tokenizer(pretrained_model_name_or_path, tokenizer)
        config = kwargs.pop("config", None)
        path = pretrained_model_name_or_path

        if os.path.isdir(path) and os.path.exists(os.path.join(path, "config.json")):
            return super().from_pretrained(
                pretrained_model_name_or_path,
                *model_args,
                tokenizer=tokenizer,
                **kwargs,
            )

        if os.path.isdir(path):
            pt_path = resolve_checkpoint_path(path)
        else:
            pt_path = path

        if not os.path.exists(pt_path):
            raise FileNotFoundError(f"Checkpoint not found: {pt_path}")

        if config is None:
            raise ValueError("config is required to load legacy checkpoints")
        model = cls(config, *model_args, tokenizer=tokenizer, **kwargs)
        state_dict = torch.load(pt_path, map_location="cpu")
        load_state_dict_with_fallback(model, state_dict)
        return model
