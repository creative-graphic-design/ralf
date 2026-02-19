import json
import logging
import os
from copy import deepcopy
from typing import Any, Optional

import datasets as ds
import torch
from einops import rearrange, reduce, repeat
from torch import BoolTensor, Tensor
from transformers import BatchEncoding, PreTrainedTokenizer

from ralf.transformers.internal.global_variables import GEO_KEYS
from ralf.transformers.internal.helpers.bucketizer import (
    BaseBucketizer,
    bucketizer_factory,
)

logger = logging.getLogger(__name__)

SPECIAL_TOKEN_VOCABULARIES = ["pad", "bos", "eos", "mask"]


_TORCH_PADDING_VALUE_FACTORY = {
    torch.int64: 0,
    torch.float32: 0.0,
    torch.bool: False,
}


def padding_value_factory(dtype: Any) -> Any:
    return _TORCH_PADDING_VALUE_FACTORY[dtype]


def _pad_sequence(seq: Tensor, max_seq_length: int) -> Tensor:
    dim = -1
    new_shape = list(seq.shape)
    s = max_seq_length - new_shape[dim]
    if s > 0:
        new_shape[dim] = s
        dtype = seq.dtype
        value = padding_value_factory(dtype)
        pad = torch.full(new_shape, value, dtype=dtype)
        new_seq = torch.cat([seq, pad], dim=dim)
    else:
        new_seq = seq
    return new_seq


class RalfTokenizer(PreTrainedTokenizer):
    """
    Hugging Face compatible tokenizer for layout sequences.

    This tokenizer mirrors LayoutSequenceTokenizer behavior while providing
    a PreTrainedTokenizer interface.
    """

    vocab_files_names = {"vocab_file": "vocab.json"}
    model_input_names = ["input_ids", "attention_mask"]
    padding_side = "right"
    truncation_side = "right"

    def __init__(
        self,
        label_names: Optional[list[str]] = None,
        max_seq_length: Optional[int] = None,
        num_bin: int = 128,
        var_order: Optional[list[str]] = None,
        pad_until_max: bool = False,
        special_tokens: Optional[list[str]] = None,
        is_loc_vocab_shared: bool = False,
        geo_quantization: str = "linear",
        kmeans_cluster_centers: Optional[dict[str, list[float]]] = None,
        model_max_length: Optional[int] = None,
        vocab_file: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        if vocab_file:
            with open(vocab_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            label_names = label_names or data["label_names"]
            max_seq_length = max_seq_length or data["max_seq_length"]
            num_bin = data.get("num_bin", num_bin)
            var_order = var_order or data.get("var_order")
            pad_until_max = data.get("pad_until_max", pad_until_max)
            special_tokens = special_tokens or data.get("special_tokens")
            is_loc_vocab_shared = data.get("is_loc_vocab_shared", is_loc_vocab_shared)
            geo_quantization = data.get("geo_quantization", geo_quantization)
            kmeans_cluster_centers = data.get(
                "kmeans_cluster_centers", kmeans_cluster_centers
            )
            model_max_length = data.get("model_max_length", model_max_length)

        if label_names is None or max_seq_length is None:
            raise ValueError("label_names and max_seq_length are required")

        self._label_names = label_names
        self._label_feature = ds.ClassLabel(
            num_classes=len(label_names), names=label_names
        )
        self._max_seq_length = max_seq_length
        self._num_bin = num_bin
        self._var_order = var_order or [
            "label",
            "width",
            "height",
            "center_x",
            "center_y",
        ]
        self._pad_until_max = pad_until_max
        self._special_tokens = special_tokens or ["pad", "bos", "eos"]
        self._is_loc_vocab_shared = is_loc_vocab_shared
        self._geo_quantization = geo_quantization

        for token in self._special_tokens:
            if token not in SPECIAL_TOKEN_VOCABULARIES:
                raise ValueError(f"Unknown special token: {token}")
        if "pad" not in self._special_tokens:
            raise ValueError("pad must be included in special_tokens")
        if "mask" in self._special_tokens:
            if self._special_tokens.index("mask") != self.N_sp_token - 1:
                raise ValueError("mask must be the last special token")

        self._bucketizers: dict[str, BaseBucketizer] = {}
        if self._geo_quantization == "kmeans" and not kmeans_cluster_centers:
            raise ValueError("kmeans_cluster_centers is required for kmeans")

        for key in self._var_order:
            if key == "label":
                continue
            bucketizer_cls = bucketizer_factory(self._geo_quantization)
            if self._geo_quantization == "kmeans":
                centers = torch.tensor(kmeans_cluster_centers[key])  # type: ignore
                bucketizer = bucketizer_cls(
                    cluster_centers=centers, n_boundaries=self._num_bin
                )
            else:
                bucketizer = bucketizer_cls(n_boundaries=self._num_bin)
            self._bucketizers[key] = bucketizer

        self._special_token_name_to_id = {
            token: self.N_label + self.N_bbox + idx
            for idx, token in enumerate(self._special_tokens)
        }
        self._special_token_id_to_name = {
            v: k for (k, v) in self._special_token_name_to_id.items()
        }

        self._use_bos_eos = (
            "bos" in self._special_tokens and "eos" in self._special_tokens
        )
        if model_max_length is None:
            model_max_length = self.max_token_length + (1 if self._use_bos_eos else 0)

        if "pad_token" not in kwargs:
            kwargs["pad_token"] = self._get_special_token("pad")
        if "bos_token" not in kwargs:
            kwargs["bos_token"] = self._get_special_token("bos")
        if "eos_token" not in kwargs:
            kwargs["eos_token"] = self._get_special_token("eos")
        if "mask_token" not in kwargs:
            kwargs["mask_token"] = self._get_special_token("mask")

        super().__init__(model_max_length=model_max_length, **kwargs)

    def _get_special_token(self, name: str) -> Optional[str]:
        return name if name in self._special_tokens else None

    @property
    def label_names(self) -> list[str]:
        return self._label_names

    @property
    def bucketizers(self) -> dict[str, BaseBucketizer]:
        return self._bucketizers

    @property
    def geo_quantization(self) -> str:
        return self._geo_quantization

    @property
    def is_loc_vocab_shared(self) -> bool:
        return self._is_loc_vocab_shared

    @property
    def max_seq_length(self) -> int:
        return self._max_seq_length

    @property
    def max_token_length(self) -> int:
        return self.max_seq_length * self.N_var_per_element

    @property
    def N_bbox(self) -> int:
        if self.is_loc_vocab_shared:
            return self.N_bbox_per_var
        return self.N_bbox_per_var * 4

    @property
    def N_bbox_per_var(self) -> int:
        return self._num_bin

    @property
    def N_label(self) -> int:
        return int(self._label_feature.num_classes)

    @property
    def N_sp_token(self) -> int:
        return len(self._special_tokens)

    @property
    def N_total(self) -> int:
        return self.N_label + self.N_bbox + self.N_sp_token

    @property
    def N_var_per_element(self) -> int:
        return len(self._var_order)

    @property
    def pad_until_max(self) -> bool:
        return self._pad_until_max

    @property
    def special_tokens(self) -> list[str]:
        return self._special_tokens

    @property
    def var_order(self) -> list[str]:
        return self._var_order

    def name_to_id(self, name: str) -> int:
        return self._special_token_name_to_id[name]

    def id_to_name(self, id_: int) -> str:
        return self._special_token_id_to_name[id_]

    def _fill_until_max_seq_length(
        self, inputs: dict[str, Tensor]
    ) -> dict[str, Tensor]:
        if self._pad_until_max:
            for key, value in inputs.items():
                inputs[key] = _pad_sequence(value, self.max_seq_length)
        return inputs

    def _insert_padding_token(self, inputs: dict[str, Tensor]) -> dict[str, Tensor]:
        if "pad" in self._special_tokens:
            pad_mask = ~inputs["mask"]
            pad_id = self.name_to_id("pad")
            inputs["label"][pad_mask] = pad_id
            for key in GEO_KEYS:
                inputs[key][pad_mask] = pad_id
        return inputs

    def _detect_oov(self, inputs: dict[str, Tensor]) -> Tensor:
        label_valid = (0 <= inputs["label"]) & (inputs["label"] < self.N_label)
        geo_valid = torch.full(label_valid.size(), fill_value=True)
        for key in GEO_KEYS:
            valid = (0 <= inputs[key]) & (inputs[key] < self.N_bbox)
            geo_valid &= valid
        invalid = torch.logical_not(label_valid & geo_valid)
        return invalid

    def _detect_eos(self, label: Tensor) -> Tensor:
        if self._use_bos_eos:
            invalid = torch.cumsum(label == self.name_to_id("eos"), dim=1) > 0
        else:
            invalid = torch.full(label.size(), fill_value=False)
        return invalid

    def _tokenize(self, text: str, **kwargs: Any) -> list[str]:
        raise NotImplementedError("RalfTokenizer does not support text tokenization")

    def get_vocab(self) -> dict[str, int]:
        vocab = {token: idx for idx, token in enumerate(self._build_vocab())}
        return vocab

    def _build_vocab(self) -> list[str]:
        vocab: list[str] = []
        vocab.extend(self._label_names)
        if self.is_loc_vocab_shared:
            vocab.extend([f"loc_{i}" for i in range(self.N_bbox_per_var)])
        else:
            for key in GEO_KEYS:
                vocab.extend([f"{key}_{i}" for i in range(self.N_bbox_per_var)])
        vocab.extend(self._special_tokens)
        return vocab

    @property
    def vocab_size(self) -> int:
        return self.N_total

    def _convert_token_to_id(self, token: str) -> int:
        if token in self._label_names:
            return self._label_names.index(token)
        if token in self._special_token_name_to_id:
            return self._special_token_name_to_id[token]
        if token.startswith("loc_") and self.is_loc_vocab_shared:
            return self.N_label + int(token.split("_")[-1])
        for key in GEO_KEYS:
            prefix = f"{key}_"
            if token.startswith(prefix) and not self.is_loc_vocab_shared:
                value = int(token.split("_")[-1])
                offset = GEO_KEYS.index(key) * self.N_bbox_per_var
                return self.N_label + offset + value
        try:
            value = int(token)
        except ValueError as exc:
            raise KeyError(f"Unknown token: {token}") from exc
        if self.is_loc_vocab_shared:
            return self.N_label + value
        first_geo = [k for k in self._var_order if k != "label"][0]
        offset = GEO_KEYS.index(first_geo) * self.N_bbox_per_var
        return self.N_label + offset + value

    def _convert_id_to_token(self, index: int) -> str:
        if index < self.N_label:
            return self._label_names[index]
        if index >= self.N_label + self.N_bbox:
            return self._special_token_id_to_name[index]
        bbox_index = index - self.N_label
        if self.is_loc_vocab_shared:
            return f"loc_{bbox_index}"
        var_idx = bbox_index // self.N_bbox_per_var
        local = bbox_index % self.N_bbox_per_var
        return f"{GEO_KEYS[var_idx]}_{local}"

    def convert_tokens_to_ids(self, tokens: list[str]) -> list[int]:
        return [self._convert_token_to_id(token) for token in tokens]

    def build_inputs_with_special_tokens(
        self, token_ids_0: list[int], token_ids_1: Optional[list[int]] = None
    ) -> list[int]:
        if token_ids_1 is not None:
            raise ValueError("RalfTokenizer does not support sequence pairs")
        if not self._use_bos_eos:
            return token_ids_0
        return [self.name_to_id("bos")] + token_ids_0 + [self.name_to_id("eos")]

    def get_special_tokens_mask(
        self,
        token_ids_0: list[int],
        token_ids_1: Optional[list[int]] = None,
        already_has_special_tokens: bool = False,
    ) -> list[int]:
        if token_ids_1 is not None:
            raise ValueError("RalfTokenizer does not support sequence pairs")
        special_ids = set(self._special_token_name_to_id.values())
        if already_has_special_tokens:
            return [1 if tok in special_ids else 0 for tok in token_ids_0]
        if not self._use_bos_eos:
            return [0 for _ in token_ids_0]
        return [1] + [0 for _ in token_ids_0] + [1]

    def save_vocabulary(
        self, save_directory: str, filename_prefix: Optional[str] = None
    ) -> tuple[str, ...]:
        os.makedirs(save_directory, exist_ok=True)
        filename = (
            "vocab.json" if filename_prefix is None else f"{filename_prefix}-vocab.json"
        )
        path = os.path.join(save_directory, filename)
        data = {
            "label_names": self._label_names,
            "max_seq_length": self._max_seq_length,
            "num_bin": self._num_bin,
            "var_order": self._var_order,
            "pad_until_max": self._pad_until_max,
            "special_tokens": self._special_tokens,
            "is_loc_vocab_shared": self._is_loc_vocab_shared,
            "geo_quantization": self._geo_quantization,
            "model_max_length": self.model_max_length,
            "kmeans_cluster_centers": self._serialize_kmeans_centers(),
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=True, indent=2)
        return (path,)

    def _serialize_kmeans_centers(self) -> Optional[dict[str, list[float]]]:
        if self._geo_quantization != "kmeans":
            return None
        centers = {}
        for key, bucketizer in self._bucketizers.items():
            centers[key] = bucketizer.centers[:, 0].tolist()
        return centers

    def encode_layout(
        self, inputs: dict[str, Tensor], add_special_tokens: bool = True
    ) -> dict[str, Tensor]:
        data = {}
        data["label"] = deepcopy(inputs["label"])
        for i, key in enumerate(GEO_KEYS):
            data[key] = self._bucketizers[key].encode(inputs[key])
            data[key] += self.N_label
            if not self._is_loc_vocab_shared:
                data[key] += i * self.N_bbox_per_var

        data["mask"] = deepcopy(inputs["mask"])
        data = self._fill_until_max_seq_length(data)
        data = self._insert_padding_token(data)

        B, S = data["label"].size()[:2]
        C = self.N_var_per_element
        seq_len = reduce(data["mask"].int(), "b s -> b 1", reduction="sum")
        indices = rearrange(torch.arange(0, S), "s -> 1 s")
        assert torch.all(torch.logical_not(data["mask"]) == (seq_len <= indices)).item()

        seq = torch.stack([data[key] for key in self._var_order], dim=-1)
        seq = rearrange(seq, "b s x -> b (s x)")
        mask = repeat(data["mask"], "b s -> b (s c)", c=C).clone()

        if add_special_tokens and self._use_bos_eos:
            indices = rearrange(torch.arange(0, S * C), "s -> 1 s")
            eos_mask = seq_len * C == indices
            seq[eos_mask] = self.name_to_id("eos")
            mask[eos_mask] = True
            bos = torch.full((B, 1), self.name_to_id("bos"))
            seq = torch.cat([bos, seq], axis=-1)
            mask = torch.cat([torch.full((B, 1), fill_value=True), mask], axis=-1)

        return {"seq": seq, "mask": mask}

    def encode(self, inputs: dict[str, Tensor], **kwargs: Any) -> list[int]:  # type: ignore
        if not isinstance(inputs, dict):
            raise NotImplementedError("RalfTokenizer only supports layout dict inputs")
        return self.encode_layout(
            inputs, add_special_tokens=kwargs.get("add_special_tokens", True)
        )["seq"][0].tolist()  # type: ignore

    def decode_layout(self, seq: Tensor) -> dict[str, Tensor]:
        seq = deepcopy(seq)
        if self._use_bos_eos and seq.size(-1) % self.N_var_per_element != 0:
            seq = seq[:, 1:]
        seq = rearrange(seq, "b (s c) -> b s c", c=self.N_var_per_element)
        outputs = {}
        for i, key in enumerate(self._var_order):
            outputs[key] = seq[..., i]
            if key in GEO_KEYS:
                outputs[key] = outputs[key] - self.N_label
                if not self._is_loc_vocab_shared:
                    mult = GEO_KEYS.index(key)
                    outputs[key] = outputs[key] - mult * self.N_bbox_per_var

        invalid = self._detect_eos(outputs["label"])
        invalid = invalid | self._detect_oov(outputs)

        for key in GEO_KEYS:
            outputs[key][invalid] = 0
            outputs[key] = self._bucketizers[key].decode(outputs[key])

        for key in self._var_order:
            padding_value = padding_value_factory(outputs[key].dtype)
            outputs[key][invalid] = padding_value

        outputs["mask"] = torch.logical_not(invalid)
        return outputs

    def decode(self, token_ids: Any, **kwargs: Any) -> dict[str, Tensor]:  # type: ignore
        if isinstance(token_ids, list):
            seq = torch.tensor(token_ids).unsqueeze(0)
        elif torch.is_tensor(token_ids):
            seq = token_ids
        else:
            raise ValueError("token_ids must be list or torch.Tensor")
        return self.decode_layout(seq)

    def __call__(
        self,
        text: Any = None,
        add_special_tokens: bool = True,
        return_tensors: Optional[str] = None,
        return_attention_mask: Optional[bool] = True,
        return_special_tokens_mask: bool = False,
        **kwargs: Any,
    ) -> BatchEncoding:
        if text is None:
            raise ValueError("text must be a layout dict or list of layout dicts")
        if isinstance(text, dict):
            batch = [text]
        elif isinstance(text, list) and text and isinstance(text[0], dict):
            batch = text
        else:
            raise NotImplementedError("RalfTokenizer only supports layout dict inputs")

        encoded = [
            self.encode_layout(item, add_special_tokens=add_special_tokens)
            for item in batch
        ]
        seq = torch.cat([e["seq"] for e in encoded], dim=0)
        mask = torch.cat([e["mask"] for e in encoded], dim=0)

        data = {
            "input_ids": seq,
            "seq": seq,
            "mask": mask,
        }
        if return_attention_mask is not False:
            data["attention_mask"] = mask
        if return_special_tokens_mask:
            data["special_tokens_mask"] = torch.tensor(
                [
                    self.get_special_tokens_mask(
                        row.tolist(), already_has_special_tokens=True
                    )
                    for row in seq
                ]
            )

        encoding = BatchEncoding(data, tensor_type=None)
        if return_tensors:
            encoding = encoding.convert_to_tensors(return_tensors)
        return encoding

    @property
    def token_mask(self) -> Tensor:
        ng_tokens = ["bos", "mask"]
        last = BoolTensor(
            [False if x in ng_tokens else True for x in self._special_tokens]
        )

        masks = {}
        if self.is_loc_vocab_shared:
            for key in GEO_KEYS:
                masks[key] = torch.cat(
                    [
                        torch.full((self.N_label,), False),
                        torch.full((self.N_bbox_per_var,), True),
                        last,
                    ]
                )
        else:
            false_tensor = torch.full((self.N_bbox,), False)
            for key in self._var_order:
                if key == "label":
                    continue
                tensor = deepcopy(false_tensor)
                mult = GEO_KEYS.index(key)
                start, stop = (
                    mult * self.N_bbox_per_var,
                    (mult + 1) * self.N_bbox_per_var,
                )
                tensor[start:stop] = True
                masks[key] = torch.cat(
                    [torch.full((self.N_label,), False), tensor, last]
                )

        masks["label"] = torch.cat(
            [
                torch.full((self.N_label,), True),
                torch.full((self.N_bbox,), False),
                last,
            ]
        )

        mask = torch.stack([masks[k] for k in self._var_order], dim=0)
        mask = repeat(mask, "x c -> (s x) c", s=self.max_seq_length)
        return mask
