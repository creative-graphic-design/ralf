import os
import random
import string
from typing import Any

import datasets as ds
import torch
from einops import rearrange, reduce

from ralf.train.global_variables import GEO_KEYS, PRECOMPUTED_WEIGHT_DIR
from ralf.train.helpers.layout_tokenizer import CHOICES, LayoutSequenceTokenizer
from ralf.transformers.ralf import RalfTokenizer
from tests.util import repeat_func


def _setup_dummy_cfg() -> dict[str, Any]:
    special_tokens = ["pad"]
    if random.random() > 0.5:
        special_tokens.extend(["bos", "eos"])
    if random.random() > 0.5:
        special_tokens.extend(["mask"])
    pad_until_max = True if random.random() > 0.5 and "pad" in special_tokens else False

    data = {
        "num_bin": 2 ** random.randint(4, 6),
        "special_tokens": special_tokens,
        "pad_until_max": pad_until_max,
    }
    for key, seq in CHOICES.items():
        data[key] = random.choice(seq)  # type: ignore
    return data


def _setup_dummy_inputs(
    batch_size: int, max_seq_length: int, num_labels: int
) -> dict[str, torch.Tensor]:
    seq_len = torch.randint(1, max_seq_length, (batch_size, 1))
    n = int(seq_len.max().item())
    inputs = {
        "label": torch.randint(num_labels, (batch_size, n)),
        "center_x": torch.rand((batch_size, n)),
        "center_y": torch.rand((batch_size, n)),
        "width": torch.rand((batch_size, n)),
        "height": torch.rand((batch_size, n)),
        "mask": seq_len > torch.arange(0, n).view(1, n),
    }
    inputs["label"][~inputs["mask"]] = 0
    for key in GEO_KEYS:
        inputs[key][~inputs["mask"]] = 0.0
    return inputs


@repeat_func(50)
def test_ralf_tokenizer_matches_layout_tokenizer() -> None:
    batch_size = random.randint(1, 4)
    max_seq_length = random.randint(2, 16)
    num_labels = random.randint(2, 8)
    names = list(string.ascii_lowercase)[:num_labels]

    label_feature = ds.ClassLabel(num_classes=num_labels, names=names)
    kwargs = _setup_dummy_cfg()

    weight_path = None
    if kwargs["geo_quantization"] == "kmeans":
        weight_path = os.path.join(
            os.environ.get("RALF_PRECOMPUTED_WEIGHT_DIR", PRECOMPUTED_WEIGHT_DIR),
            "clustering",
            "cache",
            "pku10_kmeans_train_clusters.pkl",
        )
        if not os.path.exists(weight_path):
            import pytest

            pytest.skip("kmeans clusters not found")

    legacy = LayoutSequenceTokenizer(
        label_feature=label_feature,
        max_seq_length=max_seq_length,
        weight_path=weight_path,
        **kwargs,
    )

    centers = None
    if kwargs["geo_quantization"] == "kmeans":
        centers = {
            key: legacy.bucketizers[key].centers[:, 0].tolist() for key in GEO_KEYS
        }

    tokenizer = RalfTokenizer(
        label_names=names,
        max_seq_length=max_seq_length,
        num_bin=kwargs["num_bin"],
        var_order=kwargs["var_order"],
        pad_until_max=kwargs["pad_until_max"],
        special_tokens=kwargs["special_tokens"],
        is_loc_vocab_shared=kwargs["is_loc_vocab_shared"],
        geo_quantization=kwargs["geo_quantization"],
        kmeans_cluster_centers=centers,
    )

    inputs = _setup_dummy_inputs(batch_size, max_seq_length, num_labels)
    seq_len = reduce(inputs["mask"], "b s -> b 1", reduction="sum")

    legacy_seq = legacy.encode(inputs)["seq"]
    ralf_seq_full = tokenizer.encode_layout(inputs)["seq"]
    ralf_seq = ralf_seq_full
    if "bos" in tokenizer.special_tokens and "eos" in tokenizer.special_tokens:
        legacy_seq = legacy_seq[:, 1:]
        ralf_seq = ralf_seq_full[:, 1:]
    assert torch.all(legacy_seq == ralf_seq).item()

    legacy_dec = legacy.decode(legacy_seq)
    ralf_dec = tokenizer.decode_layout(ralf_seq)
    if tokenizer.pad_until_max:
        indexes = torch.arange(0, max_seq_length)
        legacy_dec["mask"] = seq_len > rearrange(indexes, "s -> 1 s")
        ralf_dec["mask"] = seq_len > rearrange(indexes, "s -> 1 s")

    for k in legacy_dec:
        if legacy_dec[k].dtype == torch.float32:
            assert torch.allclose(legacy_dec[k], ralf_dec[k])
        else:
            assert torch.all(legacy_dec[k] == ralf_dec[k]).item()

    encoded = tokenizer(inputs, return_tensors="pt")
    assert "input_ids" in encoded
    assert torch.all(encoded["input_ids"] == ralf_seq_full).item()
