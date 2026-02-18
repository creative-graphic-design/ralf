import torch

from ralf.train.helpers.mask import batch_topk_mask, sample_mask, sequence_mask
from ralf.train.helpers.sampling import sample


def test_batch_topk_mask_shape() -> None:
    scores = torch.rand(2, 10)
    topk_mask, _ = batch_topk_mask(scores, topk=torch.tensor([3, 3]))
    assert topk_mask.shape == scores.shape
    assert topk_mask.sum(dim=1).tolist() == [3, 3]


def test_sample_mask() -> None:
    mask = torch.ones((2, 5), dtype=torch.bool)
    ratio = torch.full((2,), fill_value=0.4)
    sampled = sample_mask(mask, ratio=ratio)
    assert sampled.shape == mask.shape


def test_batch_topk_mask_with_mask() -> None:
    scores = torch.tensor([[1.0, 0.1, 0.2]])
    mask = torch.tensor([[True, False, True]])
    topk_mask, kth = batch_topk_mask(scores, topk=torch.tensor([1]), mask=mask)
    assert topk_mask.shape == scores.shape
    assert topk_mask[0, 1].item() is False
    assert topk_mask.sum().item() == 1
    assert kth.shape == (1, 1)


def test_sequence_mask_with_maxlen() -> None:
    lengths = torch.tensor([1, 3])
    mask = sequence_mask(lengths, maxlen=4)
    assert mask.shape == (2, 4)
    assert mask[0].tolist() == [True, False, False, False]
    assert mask[1].tolist() == [True, True, True, False]


def test_sampling_output_shape() -> None:
    logits = torch.randn(2, 6, 4)
    cfg = type("Cfg", (), {"name": "random", "temperature": 1.0})
    sampled = sample(logits, sampling_cfg=cfg)
    assert sampled.shape == torch.Size([2, 1, 4])


def test_sampling_modes() -> None:
    logits = torch.tensor([[2.0, 0.5, -1.0]])
    cfg_det = type("Cfg", (), {"name": "deterministic", "temperature": 1.0})
    out_det = sample(logits, sampling_cfg=cfg_det)
    assert out_det.shape == torch.Size([1, 1])

    cfg_topk = type("Cfg", (), {"name": "top_k", "temperature": 1.0, "top_k": 1})
    out_topk = sample(logits, sampling_cfg=cfg_topk)
    assert out_topk.shape == torch.Size([1, 1])

    cfg_topp = type("Cfg", (), {"name": "top_p", "temperature": 1.0, "top_p": 0.9})
    out_topp = sample(logits, sampling_cfg=cfg_topp)
    assert out_topp.shape == torch.Size([1, 1])

    cfg_gumbel = type("Cfg", (), {"name": "gumbel", "temperature": 1.0})
    out_gumbel = sample(logits, sampling_cfg=cfg_gumbel)
    assert out_gumbel.shape == torch.Size([1, 1])
