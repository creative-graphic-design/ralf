import random

import torch

from ralf.train.global_variables import GEO_KEYS
from ralf.train.models.icvt import Tokenizer as LegacyTokenizer
from ralf.transformers.icvt import ICVTTokenizer


def _dummy_inputs(batch_size: int, max_seq_length: int, num_labels: int):
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


def test_icvt_tokenizer_matches_legacy() -> None:
    batch_size = random.randint(1, 4)
    max_seq_length = random.randint(2, 16)
    num_labels = random.randint(2, 8)

    legacy = LegacyTokenizer(num_classes=num_labels, n_boundaries=128)
    tokenizer = ICVTTokenizer(
        num_classes=num_labels, n_boundaries=128, max_seq_length=max_seq_length
    )

    inputs = _dummy_inputs(batch_size, max_seq_length, num_labels)
    legacy_enc = legacy.encode(inputs)
    new_enc = tokenizer.encode_layout(inputs)
    for key in legacy_enc:
        assert torch.all(legacy_enc[key] == new_enc[key]).item()

    legacy_dec = legacy.decode(legacy_enc)
    new_dec = tokenizer.decode_layout(new_enc)
    for key in legacy_dec:
        if legacy_dec[key].dtype == torch.float32:
            assert torch.allclose(legacy_dec[key], new_dec[key])
        else:
            assert torch.all(legacy_dec[key] == new_dec[key]).item()
