import torch

from ralf.train.models.diffusion.common import (
    AdaInsNorm,
    AdaLayerNorm,
    CustomTransformerDecoder,
    CustomTransformerDecoderLayer,
    DiscreteDiffusionDecoder,
    SinusoidalPosEmb,
)


def test_sinusoidal_pos_emb() -> None:
    emb = SinusoidalPosEmb(num_steps=10, dim=6)
    x = torch.tensor([1.0, 2.0])
    out = emb(x)
    assert out.shape == (2, 6)


def test_adaptive_norm_layers() -> None:
    x = torch.randn(2, 3, 8)
    timestep = torch.tensor([1, 2])
    layer = AdaLayerNorm(n_embd=8, max_timestep=10, emb_type="adalayernorm_abs")
    out = layer(x, timestep)
    assert out.shape == x.shape

    ins = AdaInsNorm(n_embd=8, max_timestep=10, emb_type="adainnorm_mlp")
    out_ins = ins(x, timestep.float())
    assert out_ins.shape == x.shape


def test_custom_decoder_layer_branches() -> None:
    tgt = torch.randn(2, 4, 8)
    memory = torch.randn(2, 3, 8)
    timestep = torch.tensor([1, 2])

    layer_pre = CustomTransformerDecoderLayer(
        d_model=8,
        nhead=2,
        batch_first=True,
        norm_first=True,
        timestep_type="adalayernorm_abs",
    )
    out_pre = layer_pre(tgt, memory, timestep=timestep)
    assert out_pre.shape == tgt.shape

    layer_post = CustomTransformerDecoderLayer(
        d_model=8, nhead=2, batch_first=True, norm_first=False, timestep_type=None
    )
    out_post = layer_post(tgt, memory)
    assert out_post.shape == tgt.shape


def test_custom_decoder_and_discrete_decoder() -> None:
    tgt = torch.randn(2, 4, 8)
    memory = torch.randn(2, 3, 8)
    timestep = torch.tensor([1, 2])
    layer = CustomTransformerDecoderLayer(
        d_model=8, nhead=2, batch_first=True, timestep_type="adalayernorm"
    )
    decoder = CustomTransformerDecoder(decoder_layer=layer, num_layers=2)
    out = decoder(tgt, memory, timestep=timestep)
    assert out.shape == tgt.shape

    disc = DiscreteDiffusionDecoder(
        d_label=5, d_model=8, num_layers=2, nhead=2, timestep_type="adalayernorm"
    )
    seq = torch.randint(0, 5, (2, 4))
    out_disc = disc(seq, memory, timestep=timestep)
    assert out_disc.shape == (2, 4, 5)
