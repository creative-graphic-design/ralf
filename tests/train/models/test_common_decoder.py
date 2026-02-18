import torch

from ralf.train.models.common.common import (
    BaseDecoder,
    SeqLengthDistribution,
    UserConstraintTransformerEncoder,
)


def test_base_decoder_forward_and_reset() -> None:
    decoder = BaseDecoder(d_label=10, d_model=128, num_layers=1, nhead=2)
    decoder.init_weight()
    decoder.reset_embedding_layer(12)
    tgt = torch.zeros(1, 5, dtype=torch.long)
    memory = torch.zeros(1, 5, 256)
    out = decoder(tgt=tgt, memory=memory, tgt_key_padding_mask=None, is_causal=False)
    assert out.shape[:2] == (1, 5)


def test_seq_length_distribution() -> None:
    dist = SeqLengthDistribution(max_seq_length=5)
    mask = torch.tensor([[True, True, False, False, False]])
    dist.update(mask)
    sampled = dist.sample(batch_size=2)
    assert sampled.min() >= 1


def test_user_constraint_transformer_encoder() -> None:
    encoder = UserConstraintTransformerEncoder(
        d_model=16, nhead=2, num_layers=1, d_label=8, embedding_layer=None
    )
    encoder.init_weight()
    src = torch.zeros(1, 4, dtype=torch.long)
    mask = torch.zeros(1, 4, dtype=torch.bool)
    out = encoder(src=src, src_key_padding_mask=mask, task_token=None)
    assert out.shape == (1, 4, 16)
