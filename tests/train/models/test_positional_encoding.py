import pytest
import torch

from ralf.train.models.common.positional_encoding import (
    ElemAttrPositionalEncoding1d,
    PositionalEncoding1d,
    PositionEmbeddingLearned,
    PositionEmbeddingSine,
    build_position_encoding_1d,
    build_position_encoding_2d,
)


def test_position_encoding_1d() -> None:
    pe = build_position_encoding_1d("layout", 16)
    x = torch.zeros(1, 4, 16)
    out = pe(x)
    assert out.shape == x.shape

    pe2 = build_position_encoding_1d("elem_attr", 16, n_attr_per_elem=5)
    x_attr = torch.zeros(1, 5, 16)
    out2 = pe2(x_attr)
    assert out2.shape == x_attr.shape


def test_position_encoding_2d() -> None:
    pe = build_position_encoding_2d("sine", 16)
    x = torch.zeros(1, 16, 4, 4)
    out = pe(x)
    assert out.shape[0] == x.shape[0]

    pe2 = build_position_encoding_2d("reshape", 16)
    out2 = pe2(x)
    assert out2.shape == (1, 16, 16)

    pe3 = build_position_encoding_2d("learnable", 16)
    out3 = pe3(x)
    assert out3.shape == (1, 16, 16)


def test_position_encoding_extra_variants() -> None:
    pe_none = build_position_encoding_1d("none", 8)
    x = torch.zeros(1, 3, 8)
    out = pe_none(x)
    assert out.shape == x.shape

    pe = PositionalEncoding1d(d_model=8, batch_first=False, scale_input=False)
    x_seq = torch.zeros(4, 1, 8)
    out_seq = pe(x_seq)
    assert out_seq.shape == x_seq.shape

    pe_attr = ElemAttrPositionalEncoding1d(d_model=8, n_attr_per_elem=2)
    pe_attr.reset_parameters()
    x_attr = torch.zeros(1, 4, 8)
    out_attr = pe_attr(x_attr)
    assert out_attr.shape == x_attr.shape

    learned = PositionEmbeddingLearned(d_model=8)
    learned.reset_parameters()
    img = torch.zeros(1, 8, 2, 2)
    out_img = learned(img)
    assert out_img.shape == (1, 4, 8)

    with pytest.raises(ValueError):
        build_position_encoding_1d("unknown", 8)
    with pytest.raises(ValueError):
        build_position_encoding_2d("unknown", 8)
    with pytest.raises(ValueError):
        PositionEmbeddingSine(d_model=8, normalize=False, scale=1.0)
