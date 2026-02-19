import json
import os
from typing import Any, Optional

import datasets as ds
import torch
from torch import Tensor
from transformers import BatchEncoding, PreTrainedTokenizer

from ralf.transformers.internal.global_variables import GEO_KEYS
from ralf.transformers.internal.helpers.bucketizer import (
    BaseBucketizer,
    bucketizer_factory,
)


class ICVTTokenizer(PreTrainedTokenizer):
    """HF-compatible tokenizer mirroring ICVT's internal tokenizer."""

    vocab_files_names = {"vocab_file": "vocab.json"}
    model_input_names = ["label", "center_x", "center_y", "width", "height", "mask"]
    padding_side = "right"
    truncation_side = "right"

    def __init__(
        self,
        num_classes: Optional[int] = None,
        label_names: Optional[list[str]] = None,
        n_boundaries: int = 128,
        max_seq_length: Optional[int] = None,
        model_max_length: Optional[int] = None,
        vocab_file: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        if vocab_file:
            with open(vocab_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            num_classes = num_classes or data.get("num_classes")
            label_names = label_names or data.get("label_names")
            n_boundaries = data.get("n_boundaries", n_boundaries)
            max_seq_length = max_seq_length or data.get("max_seq_length")
            model_max_length = data.get("model_max_length", model_max_length)

        if num_classes is None and label_names is None:
            raise ValueError("num_classes or label_names is required")
        if label_names is None:
            label_names = [str(i) for i in range(num_classes)]
        if num_classes is None:
            num_classes = len(label_names)

        self._label_names = label_names
        self._label_feature = ds.ClassLabel(
            num_classes=len(label_names), names=label_names
        )
        self._num_classes = num_classes
        self._bg_idx = num_classes
        self._n_boundaries = n_boundaries
        self._max_seq_length = max_seq_length or 10

        self._bucketizers: dict[str, BaseBucketizer] = {}
        for key in GEO_KEYS:
            self._bucketizers[key] = bucketizer_factory("linear")(
                n_boundaries=self._n_boundaries
            )

        if model_max_length is None:
            model_max_length = self._max_seq_length

        if "pad_token" not in kwargs:
            kwargs["pad_token"] = "pad"

        super().__init__(model_max_length=model_max_length, **kwargs)

    @property
    def bg_idx(self) -> int:
        return self._bg_idx

    @property
    def label_names(self) -> list[str]:
        return self._label_names

    @property
    def max_seq_length(self) -> int:
        return self._max_seq_length

    def get_vocab(self) -> dict[str, int]:
        vocab = {name: idx for idx, name in enumerate(self._label_names)}
        vocab["bg"] = self._bg_idx
        offset = len(vocab)
        for key in GEO_KEYS:
            for i in range(self._n_boundaries):
                vocab[f"{key}_{i}"] = offset
                offset += 1
        return vocab

    @property
    def vocab_size(self) -> int:
        return len(self.get_vocab())

    def _tokenize(self, text: str, **kwargs: Any) -> list[str]:
        raise NotImplementedError("ICVTTokenizer does not support text tokenization")

    def _convert_token_to_id(self, token: str) -> int:
        vocab = self.get_vocab()
        if token not in vocab:
            raise KeyError(f"Unknown token: {token}")
        return vocab[token]

    def _convert_id_to_token(self, index: int) -> str:
        inv = {v: k for k, v in self.get_vocab().items()}
        return inv[index]

    def save_vocabulary(
        self, save_directory: str, filename_prefix: Optional[str] = None
    ) -> tuple[str, ...]:
        os.makedirs(save_directory, exist_ok=True)
        filename = (
            "vocab.json" if filename_prefix is None else f"{filename_prefix}-vocab.json"
        )
        path = os.path.join(save_directory, filename)
        data = {
            "num_classes": self._num_classes,
            "label_names": self._label_names,
            "n_boundaries": self._n_boundaries,
            "max_seq_length": self._max_seq_length,
            "model_max_length": self.model_max_length,
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=True, indent=2)
        return (path,)

    def encode_layout(self, inputs: dict[str, Tensor]) -> dict[str, Tensor]:
        padding_mask = ~inputs["mask"]
        outputs = {"mask": inputs["mask"]}

        for key in GEO_KEYS:
            outputs[key] = self._bucketizers[key].encode(inputs[key])
            outputs[key][padding_mask] = 0

        if "label" in inputs:
            outputs["label"] = inputs["label"].clone()
            outputs["label"][padding_mask] = self._bg_idx
        return outputs

    def decode_layout(self, inputs: dict[str, Tensor]) -> dict[str, Tensor]:
        outputs = {"label": inputs["label"]}
        for key in GEO_KEYS:
            outputs[key] = self._bucketizers[key].decode(inputs[key])
        outputs["mask"] = inputs["label"] != self._bg_idx
        return outputs

    def encode(self, inputs: dict[str, Tensor], **kwargs: Any) -> dict[str, Tensor]:  # type: ignore
        if not isinstance(inputs, dict):
            raise NotImplementedError("ICVTTokenizer only supports layout dict inputs")
        return self.encode_layout(inputs)

    def __call__(
        self,
        text: Any = None,
        return_tensors: Optional[str] = None,
        **kwargs: Any,
    ) -> BatchEncoding:
        if text is None:
            raise ValueError("text must be a layout dict or list of layout dicts")
        if isinstance(text, dict):
            batch = [text]
        elif isinstance(text, list) and text and isinstance(text[0], dict):
            batch = text
        else:
            raise NotImplementedError("ICVTTokenizer only supports layout dict inputs")

        encoded = [self.encode_layout(item) for item in batch]
        data = {key: torch.cat([e[key] for e in encoded], dim=0) for key in encoded[0]}
        encoding = BatchEncoding(data, tensor_type=None)
        if return_tensors:
            encoding = encoding.convert_to_tensors(return_tensors)
        return encoding

    def decode(self, token_ids: Any, **kwargs: Any) -> dict[str, Tensor]:  # type: ignore
        if isinstance(token_ids, dict):
            return self.decode_layout(token_ids)
        raise NotImplementedError("ICVTTokenizer expects dict inputs for decode")
