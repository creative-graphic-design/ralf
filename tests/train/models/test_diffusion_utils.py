import torch
from omegaconf import OmegaConf

from ralf.train.config import TokenizerConfig
from ralf.train.helpers.layout_tokenizer import LayoutSequenceTokenizer
from ralf.train.helpers.relationships import RelLoc, RelSize
from ralf.train.models.common.base_model import ConditionalInputsForDiscreteLayout
from ralf.train.models.diffusion.discrete.clg_lo import (
    Graph,
    relation_loc_canvas_t,
    relation_size_sm,
)
from ralf.train.models.diffusion.discrete.constrained import (
    ConstrainedMaskAndReplaceDiffusion,
)
from ralf.train.models.diffusion.discrete.logit_adjustment import (
    _index_to_smoothed_log_onehot,
    set_weak_logits_for_refinement,
    update_logits_for_relation,
)
from ralf.train.models.diffusion.discrete.pf_converter import Converter
from ralf.train.models.diffusion.discrete.util import index_to_log_onehot


def _mask_tokenizer(features) -> LayoutSequenceTokenizer:
    cfg = TokenizerConfig()
    cfg_dict = cfg.__dict__.copy()
    cfg_dict.pop("special_tokens", None)
    return LayoutSequenceTokenizer(
        label_feature=features["label"].feature,
        max_seq_length=2,
        special_tokens=["pad", "mask"],
        **cfg_dict,
    )


def test_logit_adjustment_and_converter(features) -> None:
    tokenizer = _mask_tokenizer(features)
    seq = torch.zeros(1, tokenizer.max_token_length, dtype=torch.long)
    logits = _index_to_smoothed_log_onehot(seq, tokenizer, mode="uniform")
    assert logits.shape[1] == tokenizer.N_total

    mask = torch.ones_like(seq, dtype=torch.bool)
    cond = ConditionalInputsForDiscreteLayout(
        image=torch.zeros(1, 4, 2, 2),
        id=None,
        seq_observed=seq,
        mask=mask,
        seq=seq,
    )
    sampling_cfg = OmegaConf.create(
        {
            "refine_lambda": 0.1,
            "refine_mode": "uniform",
            "refine_offset_ratio": 0.1,
        }
    )
    cond = set_weak_logits_for_refinement(cond, tokenizer, sampling_cfg)
    assert cond.weak_logits is not None

    cond.edge_indexes = torch.tensor([[[0, 1]]])
    cond.edge_attributes = torch.tensor([[1 << RelSize.SMALLER]])
    model_log_prob = torch.randn(1, tokenizer.N_total, tokenizer.max_token_length)
    sampling_cfg = OmegaConf.create({"relation_lambda": 0.01, "relation_num_update": 1})
    updated = update_logits_for_relation(
        t=11,
        cond=cond,
        model_log_prob=model_log_prob,
        tokenizer=tokenizer,
        sampling_cfg=sampling_cfg,
    )
    assert updated.shape == model_log_prob.shape

    graph = Graph(
        edge_indexes=torch.tensor([[[0, 1]]]),
        edge_attributes=torch.tensor([[1 << RelLoc.TOP]]),
    )
    bbox = torch.rand(1, 2, 4)
    _ = relation_loc_canvas_t(bbox, graph)
    graph2 = Graph(
        edge_indexes=torch.tensor([[[0, 1]]]),
        edge_attributes=torch.tensor([[1 << RelSize.SMALLER]]),
    )
    _ = relation_size_sm(bbox, graph2)

    converter = Converter(tokenizer)
    ids_p = torch.zeros(1, tokenizer.max_seq_length, tokenizer.N_var_per_element).long()
    ids_f = converter.p_to_f_id_all(ids_p)
    ids_p_back = converter.f_to_p_id_all(ids_f)
    assert ids_p_back.shape == ids_p.shape


def test_constrained_diffusion_q_pred(features) -> None:
    tokenizer = _mask_tokenizer(features)
    model = ConstrainedMaskAndReplaceDiffusion(
        d_model=8,
        num_layers=1,
        nhead=1,
        tokenizer=tokenizer,
        num_timesteps=2,
        pos_emb="elem_attr",
        auxiliary_loss_weight=0.0,
    )
    seq = torch.zeros(1, tokenizer.max_token_length, dtype=torch.long)
    log_x_start = index_to_log_onehot(seq, tokenizer.N_total)
    t = torch.zeros(1, dtype=torch.long)
    out = model.q_pred(log_x_start, t, key="label")
    assert out.shape[0] == 1

    out_one = model.q_pred_one_timestep(log_x_start, t, key="label")
    assert out_one.shape[0] == 1

    log_sample = model.log_sample_categorical(log_x_start, key="label")
    assert log_sample.shape[0] == 1

    converter = Converter(tokenizer)
    log_prob = index_to_log_onehot(seq, tokenizer.N_total)
    log_p = converter.f_to_p_log(log_prob, "label")
    log_f = converter.p_to_f_log(log_p, "label")
    assert log_f.shape[1] == tokenizer.N_total

    log_x_start_label = index_to_log_onehot(
        torch.zeros(1, tokenizer.max_token_length, dtype=torch.long),
        model.mat_size["label"],
    )
    log_x_t_label = model.q_sample(log_x_start_label, t, key="label")
    log_x_t_full = converter.p_to_f_log(log_x_t_label, "label")
    posterior = model.q_posterior(log_x_start, log_x_t_full, t)
    assert posterior.shape[0] == 1
