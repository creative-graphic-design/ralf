from typing import Any, Optional

import torch
from transformers.processing_utils import ProcessorMixin

from ..icvt.tokenization_icvt import ICVTTokenizer
from .tokenization_ralf import RalfTokenizer


class RalfProcessor(ProcessorMixin):
    """Tokenizer-focused processor for RALF models."""

    attributes = ["tokenizer", "icvt_tokenizer"]
    tokenizer_class = "RalfTokenizer"

    def __init__(
        self,
        tokenizer: Optional[RalfTokenizer] = None,
        icvt_tokenizer: Optional[ICVTTokenizer] = None,
    ) -> None:
        self.tokenizer = tokenizer
        self.icvt_tokenizer = icvt_tokenizer

    def __call__(self, inputs: dict[str, Any], model_type: str) -> tuple[dict, dict]:
        if model_type in {"autoreg", "ralf"}:
            if self.tokenizer is None:
                raise ValueError("tokenizer is required")
            data = self.tokenizer.encode_layout(inputs)
            image = torch.cat([inputs["image"], inputs["saliency"]], dim=1)
            model_inputs = {
                "seq": data["seq"][:, :-1],
                "tgt_key_padding_mask": ~data["mask"][:, :-1],
                "image": image,
            }
            if "retrieved" in inputs:
                model_inputs["retrieved"] = inputs["retrieved"]
            targets = {"seq": data["seq"][:, 1:]}
            return model_inputs, targets

        if model_type == "maskgit":
            if self.tokenizer is None:
                raise ValueError("tokenizer is required")
            data = self.tokenizer.encode_layout(inputs)
            image = torch.cat([inputs["image"], inputs["saliency"]], dim=1)
            model_inputs = {
                "seq": data["seq"],
                "image": image,
            }
            targets = {"seq": data["seq"], "loss_mask": data["mask"]}
            return model_inputs, targets

        if model_type == "layoutdm":
            if self.tokenizer is None:
                raise ValueError("tokenizer is required")
            data = self.tokenizer.encode_layout(inputs)
            image = torch.cat([inputs["image"], inputs["saliency"]], dim=1)
            model_inputs = {"image": image}
            targets = {**data, "image": image}
            return model_inputs, targets

        if model_type == "icvt":
            if self.icvt_tokenizer is None:
                raise ValueError("icvt_tokenizer is required")
            tokenized = self.icvt_tokenizer.encode_layout(inputs)
            model_inputs = {
                "image": torch.cat([inputs["image"], inputs["saliency"]], dim=1),
                **tokenized,
            }
            targets = tokenized
            return model_inputs, targets

        raise ValueError(f"Unknown model_type: {model_type}")
