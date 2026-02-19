import os

import pytest
import torch

from ralf.transformers.autoreg import RalfAutoregConfig
from ralf.transformers.cglgan import RalfCGLGANConfig
from ralf.transformers.dsgan import RalfDSGANConfig
from ralf.transformers.icvt import ICVTTokenizer, RalfICVTConfig
from ralf.transformers.layoutdm import RalfLayoutDMConfig
from ralf.transformers.maskgit import RalfMaskGITConfig
from ralf.transformers.modeling_utils import (
    add_prefix,
    build_features_from_labels,
    build_features_from_tokenizer,
    load_state_dict_with_fallback,
    normalize_state_dict,
    remove_prefix,
    resolve_checkpoint_path,
)
from ralf.transformers.ralf import (
    RalfProcessor,
    RalfRetrievalAugmentedAutoregConfig,
    RalfTokenizer,
)
from ralf.transformers.ralf.tokenization_ralf import (
    _pad_sequence,
    padding_value_factory,
)


def _dummy_layout(
    batch_size: int, seq_length: int, num_labels: int
) -> dict[str, torch.Tensor]:
    torch.manual_seed(0)
    mask = torch.zeros((batch_size, seq_length), dtype=torch.bool)
    mask[:, : max(seq_length - 1, 1)] = True
    label = torch.randint(num_labels, (batch_size, seq_length))
    center_x = torch.rand((batch_size, seq_length))
    center_y = torch.rand((batch_size, seq_length))
    width = torch.rand((batch_size, seq_length))
    height = torch.rand((batch_size, seq_length))
    for key in [label, center_x, center_y, width, height]:
        key[~mask] = 0
    return {
        "label": label,
        "center_x": center_x,
        "center_y": center_y,
        "width": width,
        "height": height,
        "mask": mask,
    }


def test_config_defaults() -> None:
    _ = RalfAutoregConfig()
    _ = RalfMaskGITConfig()
    _ = RalfLayoutDMConfig()
    _ = RalfRetrievalAugmentedAutoregConfig()
    _ = RalfCGLGANConfig()
    _ = RalfDSGANConfig()
    _ = RalfICVTConfig()


def test_modeling_utils_prefix_and_resolve(tmp_path) -> None:
    state = {"weight": torch.ones(2, 2)}
    assert normalize_state_dict({"module.weight": state["weight"]}) == state
    prefixed = add_prefix(state, "model.")
    assert remove_prefix(prefixed, "model.") == state
    assert remove_prefix(state, "model.") == state

    model_path = tmp_path / "model.pt"
    model_path.write_bytes(b"test")
    assert resolve_checkpoint_path(str(model_path)) == str(model_path)

    ckpt_dir = tmp_path / "ckpt"
    ckpt_dir.mkdir()
    (ckpt_dir / "gen_final_model.pt").write_bytes(b"test")
    assert resolve_checkpoint_path(str(ckpt_dir)) == str(
        ckpt_dir / "gen_final_model.pt"
    )


def test_modeling_utils_load_state_dict_with_fallback() -> None:
    model = torch.nn.Linear(2, 2, bias=False)
    state = {k: v.clone() for k, v in model.state_dict().items()}
    prefixed = {f"model.{k}": v.clone() for k, v in state.items()}
    load_state_dict_with_fallback(model, prefixed)


def test_modeling_utils_build_features() -> None:
    tokenizer = RalfTokenizer(
        label_names=["a", "b"],
        max_seq_length=4,
        num_bin=8,
        special_tokens=["pad", "bos", "eos"],
        pad_until_max=False,
    )
    features = build_features_from_tokenizer(tokenizer)
    assert features["label"].feature.names == ["a", "b"]
    features = build_features_from_labels(["x", "y", "z"])
    assert features["label"].feature.names == ["x", "y", "z"]


def test_ralf_tokenizer_branches(tmp_path) -> None:
    tokenizer = RalfTokenizer(
        label_names=["0", "1"],
        max_seq_length=3,
        num_bin=4,
        special_tokens=["pad", "bos", "eos", "mask"],
        pad_until_max=True,
    )
    layout = _dummy_layout(1, 3, 2)
    encoded = tokenizer(layout, return_special_tokens_mask=True, return_tensors="pt")
    assert "special_tokens_mask" in encoded
    decoded = tokenizer.decode(encoded["seq"])
    assert decoded["mask"].shape[-1] == layout["mask"].shape[-1]
    assert tokenizer.bucketizers
    assert tokenizer.geo_quantization == "linear"
    assert tokenizer.var_order

    assert tokenizer.get_special_tokens_mask(
        encoded["seq"][0].tolist(), already_has_special_tokens=True
    )
    token_ids = tokenizer.build_inputs_with_special_tokens([0, 1, 2])
    assert token_ids[0] == tokenizer.name_to_id("bos")
    with pytest.raises(ValueError):
        tokenizer.build_inputs_with_special_tokens([0], [1])
    with pytest.raises(ValueError):
        tokenizer.get_special_tokens_mask([0], token_ids_1=[1])

    vocab = tokenizer.get_vocab()
    assert vocab[
        tokenizer.id_to_name(tokenizer.name_to_id("pad"))
    ] == tokenizer.name_to_id("pad")
    assert tokenizer.convert_tokens_to_ids(["0", "1"]) == [0, 1]
    assert tokenizer._convert_token_to_id("pad") == tokenizer.name_to_id("pad")
    assert tokenizer._convert_token_to_id("center_x_1") >= tokenizer.N_label
    assert tokenizer._convert_token_to_id("5") >= tokenizer.N_label
    assert tokenizer._convert_id_to_token(tokenizer.name_to_id("pad")) == "pad"

    seq = torch.tensor(
        [[tokenizer.name_to_id("bos"), 0, 1, 2, tokenizer.name_to_id("eos")]]
    )
    tokenizer.decode(seq)
    tokenizer.decode(encoded["seq"][0].tolist())

    encoded_list = tokenizer([layout, layout], return_attention_mask=False)
    assert "attention_mask" not in encoded_list

    tokenizer.save_pretrained(tmp_path)
    loaded = RalfTokenizer.from_pretrained(tmp_path)
    assert loaded.label_names == tokenizer.label_names

    shared = RalfTokenizer(
        label_names=["0", "1"],
        max_seq_length=2,
        num_bin=3,
        special_tokens=["pad"],
        pad_until_max=False,
        is_loc_vocab_shared=True,
    )
    assert shared.build_inputs_with_special_tokens([1, 2]) == [1, 2]
    assert shared.get_special_tokens_mask(
        [shared.name_to_id("pad")], already_has_special_tokens=True
    )
    assert shared.get_special_tokens_mask([0, 1], already_has_special_tokens=False) == [
        0,
        0,
    ]
    assert shared._convert_token_to_id("loc_1") == shared.N_label + 1
    assert shared._convert_id_to_token(shared.N_label + 1) == "loc_1"

    assert (
        tokenizer.token_mask.shape[0]
        == tokenizer.max_seq_length * tokenizer.N_var_per_element
    )
    assert (
        shared.token_mask.shape[0] == shared.max_seq_length * shared.N_var_per_element
    )


def test_ralf_tokenizer_errors_and_kmeans() -> None:
    with pytest.raises(ValueError):
        RalfTokenizer(label_names=None, max_seq_length=None)

    with pytest.raises(ValueError):
        RalfTokenizer(
            label_names=["0"],
            max_seq_length=2,
            special_tokens=["pad", "bos", "eos", "unknown"],
        )

    with pytest.raises(ValueError):
        RalfTokenizer(
            label_names=["0"], max_seq_length=2, special_tokens=["bos", "eos"]
        )

    with pytest.raises(ValueError):
        RalfTokenizer(
            label_names=["0"],
            max_seq_length=2,
            special_tokens=["pad", "mask", "bos"],
        )

    with pytest.raises(ValueError):
        RalfTokenizer(
            label_names=["0"],
            max_seq_length=2,
            geo_quantization="kmeans",
        )

    centers = {
        key: [0.1, 0.5, 0.9] for key in ["center_x", "center_y", "width", "height"]
    }
    kmeans_tok = RalfTokenizer(
        label_names=["0"],
        max_seq_length=2,
        num_bin=3,
        geo_quantization="kmeans",
        kmeans_cluster_centers=centers,
    )
    serialized = kmeans_tok._serialize_kmeans_centers()
    for key, values in centers.items():
        assert serialized[key] == pytest.approx(values)

    with pytest.raises(NotImplementedError):
        kmeans_tok._tokenize("text")
    with pytest.raises(NotImplementedError):
        kmeans_tok.encode("text")
    with pytest.raises(ValueError):
        kmeans_tok.decode("bad")
    with pytest.raises(ValueError):
        kmeans_tok()
    with pytest.raises(NotImplementedError):
        kmeans_tok(text="bad")


def test_tokenizer_helpers() -> None:
    tensor = torch.ones(1, 3, dtype=torch.float32)
    assert padding_value_factory(tensor.dtype) == 0.0
    assert _pad_sequence(tensor, 2).shape == tensor.shape


def test_icvt_tokenizer_branches(tmp_path) -> None:
    tokenizer = ICVTTokenizer(num_classes=2, max_seq_length=3, n_boundaries=4)
    assert tokenizer.bg_idx == 2
    assert tokenizer.label_names == ["0", "1"]
    assert tokenizer.max_seq_length == 3
    layout = _dummy_layout(1, 3, 2)
    encoded = tokenizer(layout)
    assert "label" in encoded
    decoded = tokenizer.decode(dict(encoded))
    assert decoded["mask"].shape == layout["mask"].shape

    vocab = tokenizer.get_vocab()
    assert vocab["bg"] == tokenizer.bg_idx
    assert tokenizer._convert_token_to_id("bg") == tokenizer.bg_idx
    assert tokenizer._convert_id_to_token(tokenizer.bg_idx) == "bg"

    tokenizer.save_pretrained(tmp_path)
    loaded = ICVTTokenizer.from_pretrained(tmp_path)
    assert loaded.max_seq_length == tokenizer.max_seq_length
    vocab_file = os.path.join(tmp_path, "vocab.json")
    loaded_from_vocab = ICVTTokenizer(vocab_file=vocab_file)
    assert loaded_from_vocab.max_seq_length == tokenizer.max_seq_length

    with pytest.raises(NotImplementedError):
        tokenizer._tokenize("text")
    with pytest.raises(KeyError):
        tokenizer._convert_token_to_id("unknown")
    with pytest.raises(ValueError):
        ICVTTokenizer()
    with pytest.raises(ValueError):
        tokenizer()
    with pytest.raises(NotImplementedError):
        tokenizer.decode("bad")
    with pytest.raises(NotImplementedError):
        tokenizer(text="bad")


def test_ralf_processor_branches() -> None:
    ralf_tokenizer = RalfTokenizer(
        label_names=["0", "1"],
        max_seq_length=3,
        num_bin=4,
        special_tokens=["pad", "bos", "eos"],
        pad_until_max=False,
    )
    icvt_tokenizer = ICVTTokenizer(num_classes=2, max_seq_length=3, n_boundaries=4)
    processor = RalfProcessor(tokenizer=ralf_tokenizer, icvt_tokenizer=icvt_tokenizer)

    layout = _dummy_layout(1, 3, 2)
    layout["image"] = torch.rand((1, 3, 8, 8))
    layout["saliency"] = torch.rand((1, 1, 8, 8))

    inputs, targets = processor(layout, model_type="autoreg")
    assert "seq" in inputs and "seq" in targets

    inputs, targets = processor(layout, model_type="maskgit")
    assert "loss_mask" in targets

    inputs, targets = processor(layout, model_type="layoutdm")
    assert "image" in targets

    inputs, targets = processor(layout, model_type="icvt")
    assert "label" in inputs and "label" in targets

    with pytest.raises(ValueError):
        processor(layout, model_type="unknown")

    processor_missing = RalfProcessor()
    with pytest.raises(ValueError):
        processor_missing(layout, model_type="autoreg")
    with pytest.raises(ValueError):
        processor_missing(layout, model_type="maskgit")
    with pytest.raises(ValueError):
        processor_missing(layout, model_type="layoutdm")
    with pytest.raises(ValueError):
        processor_missing(layout, model_type="icvt")
