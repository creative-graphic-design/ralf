import os

import datasets as ds
import torch

from ralf.train.global_variables import PRECOMPUTED_WEIGHT_DIR
from ralf.train.models.autoreg import ConcateAuxilaryTaskAutoreg
from ralf.train.models.common.base_model import ConditionalInputsForDiscreteLayout
from ralf.transformers.autoreg import RalfAutoregConfig, RalfAutoregModel
from ralf.transformers.ralf import RalfTokenizer


def _dummy_layout(batch_size: int, seq_length: int, num_labels: int):
    seq_len = torch.randint(1, seq_length, (batch_size, 1))
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
    inputs["center_x"][~inputs["mask"]] = 0.0
    inputs["center_y"][~inputs["mask"]] = 0.0
    inputs["width"][~inputs["mask"]] = 0.0
    inputs["height"][~inputs["mask"]] = 0.0
    return inputs


def test_autoreg_wrapper_matches_original() -> None:
    weight_path = os.path.join(
        os.environ.get("RALF_PRECOMPUTED_WEIGHT_DIR", PRECOMPUTED_WEIGHT_DIR),
        "resnet50_a1_0-14fe96d1.pth",
    )
    if not os.path.exists(weight_path):
        import pytest

        pytest.skip("ResNet50 weights not found")
    torch.manual_seed(0)
    batch_size = 2
    num_labels = 5
    max_seq_length = 8

    tokenizer = RalfTokenizer(
        label_names=[str(i) for i in range(num_labels)],
        max_seq_length=max_seq_length,
        num_bin=32,
        special_tokens=["pad", "bos", "eos"],
        pad_until_max=False,
    )

    config = RalfAutoregConfig(
        d_model=256,
        decoder_d_model=256,
        decoder_num_layers=2,
        auxilary_task="uncond",
        max_position_embeddings=tokenizer.max_token_length,
    )

    features = ds.Features(
        {"label": ds.ClassLabel(num_classes=num_labels, names=tokenizer.label_names)}
    )
    original = ConcateAuxilaryTaskAutoreg(
        features=features,
        tokenizer=tokenizer,
        d_model=config.d_model,
        decoder_d_model=config.decoder_d_model,
        decoder_num_layers=config.decoder_num_layers,
        auxilary_task=config.auxilary_task,
    )
    original.eval()

    wrapper = RalfAutoregModel(config=config, tokenizer=tokenizer)
    wrapper.model.load_state_dict(original.state_dict())
    wrapper.eval()

    layout = _dummy_layout(batch_size, max_seq_length, num_labels)
    data = tokenizer.encode_layout(layout)
    image = torch.rand((batch_size, 4, 64, 64))
    cond_inputs = ConditionalInputsForDiscreteLayout(image=image, id=None)
    seq_constraints = original.preprocessor(cond_inputs)
    inputs = {
        "seq": data["seq"][:, :-1],
        "tgt_key_padding_mask": ~data["mask"][:, :-1],
        "image": image,
        "seq_layout_const": seq_constraints["seq"],
        "seq_layout_const_pad_mask": seq_constraints["pad_mask"],
    }

    out_orig = original(inputs)
    out_wrap = wrapper(
        seq=inputs["seq"],
        tgt_key_padding_mask=inputs["tgt_key_padding_mask"],
        image=inputs["image"],
        seq_layout_const=inputs["seq_layout_const"],
        seq_layout_const_pad_mask=inputs["seq_layout_const_pad_mask"],
    )

    assert torch.allclose(out_orig["logits"], out_wrap["logits"])


def test_autoreg_save_and_load(tmp_path) -> None:
    weight_path = os.path.join(
        os.environ.get("RALF_PRECOMPUTED_WEIGHT_DIR", PRECOMPUTED_WEIGHT_DIR),
        "resnet50_a1_0-14fe96d1.pth",
    )
    if not os.path.exists(weight_path):
        import pytest

        pytest.skip("ResNet50 weights not found")
    torch.manual_seed(0)
    num_labels = 4
    max_seq_length = 6

    tokenizer = RalfTokenizer(
        label_names=[str(i) for i in range(num_labels)],
        max_seq_length=max_seq_length,
        num_bin=32,
        special_tokens=["pad", "bos", "eos"],
        pad_until_max=False,
    )
    config = RalfAutoregConfig(
        d_model=256,
        decoder_d_model=256,
        decoder_num_layers=2,
        auxilary_task="uncond",
        max_position_embeddings=tokenizer.max_token_length,
    )
    model = RalfAutoregModel(config=config, tokenizer=tokenizer)

    tokenizer.save_pretrained(tmp_path)
    model.save_pretrained(tmp_path)

    loaded = RalfAutoregModel.from_pretrained(tmp_path)
    assert loaded is not None
