import os
import random

import numpy as np
import pytest
import torch

from ralf.train.global_variables import PRECOMPUTED_WEIGHT_DIR
from ralf.train.models.cgl import CGLDiscriminator, CGLGenerator
from ralf.train.models.dsgan import DSDiscriminator, DSGenerator
from ralf.train.models.icvt import ICVTGenerator
from ralf.train.models.layoutdm import LayoutDM
from ralf.train.models.maskgit import MaskGIT
from ralf.transformers.cglgan import (
    RalfCGLGANConfig,
    RalfCGLGANDiscriminatorModel,
    RalfCGLGANGeneratorModel,
)
from ralf.transformers.dsgan import (
    RalfDSGANConfig,
    RalfDSGANDiscriminatorModel,
    RalfDSGANGeneratorModel,
)
from ralf.transformers.icvt import ICVTTokenizer, RalfICVTConfig, RalfICVTModel
from ralf.transformers.layoutdm import RalfLayoutDMConfig, RalfLayoutDMModel
from ralf.transformers.maskgit import RalfMaskGITConfig, RalfMaskGITModel
from ralf.transformers.modeling_utils import (
    build_features_from_labels,
    build_features_from_tokenizer,
)
from ralf.transformers.ralf import RalfTokenizer

_TORCHVISION_RESNET50 = "resnet50-11ad3fa6.pth"


def _ensure_torchvision_weights() -> bool:
    checkpoints = os.path.join(torch.hub.get_dir(), "checkpoints")
    return os.path.exists(os.path.join(checkpoints, _TORCHVISION_RESNET50))


def _ensure_precomputed_weights() -> bool:
    weight_dir = os.environ.get("RALF_PRECOMPUTED_WEIGHT_DIR", PRECOMPUTED_WEIGHT_DIR)
    weight_path = os.path.join(weight_dir, "resnet50_a1_0-14fe96d1.pth")
    return os.path.exists(weight_path)


def _dummy_layout(
    batch_size: int, seq_length: int, num_labels: int
) -> dict[str, torch.Tensor]:
    torch.manual_seed(0)
    seq_len = torch.randint(1, seq_length + 1, (batch_size, 1))
    n = int(seq_len.max().item())
    inputs = {
        "label": torch.randint(num_labels, (batch_size, n)),
        "center_x": torch.rand((batch_size, n)),
        "center_y": torch.rand((batch_size, n)),
        "width": torch.rand((batch_size, n)),
        "height": torch.rand((batch_size, n)),
        "mask": seq_len > torch.arange(0, n).view(1, n),
    }
    for key in ["label", "center_x", "center_y", "width", "height"]:
        inputs[key][~inputs["mask"]] = 0
    return inputs


def _add_images(
    batch: dict[str, torch.Tensor], height: int, width: int
) -> dict[str, torch.Tensor]:
    batch = dict(batch)
    batch["image"] = torch.rand((batch["label"].size(0), 3, height, width))
    batch["saliency"] = torch.rand((batch["label"].size(0), 1, height, width))
    return batch


def test_maskgit_wrapper_matches_original() -> None:
    if not _ensure_torchvision_weights():
        pytest.skip("Torchvision resnet50 weights not found")
    torch.manual_seed(0)
    batch_size = 1
    num_labels = 4
    max_seq_length = 4

    tokenizer = RalfTokenizer(
        label_names=[str(i) for i in range(num_labels)],
        max_seq_length=max_seq_length,
        num_bin=16,
        special_tokens=["pad", "mask"],
        pad_until_max=True,
    )
    config = RalfMaskGITConfig(
        d_model=256,
        mask_schedule="linear",
        use_padding_as_vocab=True,
        use_gumbel_noise=True,
        pad_weight=1.0,
        num_timesteps=8,
        max_position_embeddings=tokenizer.max_token_length,
    )
    features = build_features_from_tokenizer(tokenizer)

    original = MaskGIT(
        features=features,
        tokenizer=tokenizer,
        d_model=config.d_model,
        mask_schedule=config.mask_schedule,
        use_padding_as_vocab=config.use_padding_as_vocab,
        use_gumbel_noise=config.use_gumbel_noise,
        pad_weight=config.pad_weight,
    )
    original.num_timesteps = config.num_timesteps
    original.eval()

    wrapper = RalfMaskGITModel(config=config, tokenizer=tokenizer)
    wrapper.model.load_state_dict(original.state_dict())
    wrapper.eval()

    layout = _dummy_layout(batch_size, max_seq_length, num_labels)
    data = tokenizer.encode_layout(layout)
    image = torch.rand((batch_size, 4, 64, 64))
    inputs = {
        "seq": data["seq"],
        "tgt_key_padding_mask": ~data["mask"],
        "image": image,
    }

    out_orig = original(inputs)
    out_wrap = wrapper(
        seq=inputs["seq"],
        tgt_key_padding_mask=inputs["tgt_key_padding_mask"],
        image=inputs["image"],
    )

    assert torch.allclose(out_orig["logits"], out_wrap["logits"])


def test_layoutdm_wrapper_matches_original() -> None:
    if not _ensure_torchvision_weights():
        pytest.skip("Torchvision resnet50 weights not found")
    torch.manual_seed(0)
    batch_size = 1
    num_labels = 4
    max_seq_length = 4

    tokenizer = RalfTokenizer(
        label_names=[str(i) for i in range(num_labels)],
        max_seq_length=max_seq_length,
        num_bin=16,
        special_tokens=["pad", "mask"],
        pad_until_max=True,
    )
    config = RalfLayoutDMConfig(
        d_model=256,
        num_timesteps=10,
        pos_emb="elem_attr",
        auxiliary_loss_weight=1e-1,
        q_type="constrained",
        retrieval_augmentation=False,
        max_position_embeddings=tokenizer.max_token_length,
    )
    features = build_features_from_tokenizer(tokenizer)

    original = LayoutDM(
        features=features,
        tokenizer=tokenizer,
        d_model=config.d_model,
        num_timesteps=config.num_timesteps,
        pos_emb=config.pos_emb,
        auxiliary_loss_weight=config.auxiliary_loss_weight,
        q_type=config.q_type,
        retrieval_augmentation=config.retrieval_augmentation,
    )
    original.eval()

    wrapper = RalfLayoutDMModel(config=config, tokenizer=tokenizer)
    wrapper.model.load_state_dict(original.state_dict())
    wrapper.eval()

    layout = _dummy_layout(batch_size, max_seq_length, num_labels)
    data = tokenizer.encode_layout(layout)
    image = torch.rand((batch_size, 4, 64, 64))
    inputs = {"image": image}
    targets = {"seq": data["seq"]}

    torch.manual_seed(0)
    out_orig, loss_orig = original.train_loss(inputs, targets)
    torch.manual_seed(0)
    out_wrap, loss_wrap = wrapper.train_loss(inputs, targets)

    assert torch.allclose(out_orig["logits"], out_wrap["logits"])
    for key in loss_orig:
        assert torch.allclose(loss_orig[key], loss_wrap[key])


def test_icvt_wrapper_matches_original() -> None:
    if not _ensure_torchvision_weights():
        pytest.skip("Torchvision resnet50 weights not found")
    torch.manual_seed(0)
    batch_size = 1
    num_labels = 3
    max_seq_length = 10

    tokenizer = ICVTTokenizer(
        label_names=[str(i) for i in range(num_labels)],
        max_seq_length=max_seq_length,
        n_boundaries=128,
    )
    config = RalfICVTConfig(
        d_model=320,
        backbone="resnet50",
        ga_type=None,
        kl_mult=1.0,
        decoder_only=False,
        ignore_bg_bbox_loss=True,
        n_boundaries=128,
        max_seq_length=max_seq_length,
        max_position_embeddings=max_seq_length,
    )
    features = build_features_from_labels(tokenizer.label_names)

    original = ICVTGenerator(
        features=features,
        d_model=config.d_model,
        backbone=config.backbone,
        ga_type=config.ga_type,
        kl_mult=config.kl_mult,
        decoder_only=config.decoder_only,
        ignore_bg_bbox_loss=config.ignore_bg_bbox_loss,
    )
    original.tokenizer = tokenizer
    original.eval()

    wrapper = RalfICVTModel(config=config, tokenizer=tokenizer)
    wrapper.model.load_state_dict(original.state_dict())
    wrapper.eval()

    layout = _dummy_layout(batch_size, max_seq_length, num_labels)
    encoded = tokenizer.encode_layout(layout)
    image = torch.rand((batch_size, 4, 350, 240))
    inputs = {**encoded, "image": image}
    targets = encoded

    torch.manual_seed(0)
    out_orig, loss_orig = original.train_loss(inputs, targets)
    torch.manual_seed(0)
    out_wrap, loss_wrap = wrapper.train_loss(inputs, targets)

    for key in out_orig:
        assert torch.allclose(out_orig[key], out_wrap[key])
    for key in loss_orig:
        assert torch.allclose(loss_orig[key], loss_wrap[key])


def test_cglgan_wrapper_matches_original() -> None:
    if not _ensure_precomputed_weights():
        pytest.skip("Precomputed resnet50 weights not found")
    torch.manual_seed(0)
    np.random.seed(0)
    random.seed(0)
    batch_size = 1
    num_labels = 3
    max_seq_length = 10

    config = RalfCGLGANConfig(
        backbone="resnet50",
        in_channels=8,
        out_channels=128,
        num_layers=2,
        max_seq_length=max_seq_length,
        apply_weight=True,
        d_model=128,
        use_reorder=False,
        use_reorder_for_random=False,
        auxilary_task="uncond",
        dis_backbone="resnet50",
        dis_in_channels=8,
        dis_out_channels=128,
        dis_num_layers=2,
        dis_d_model=128,
    )
    label_names = [str(i) for i in range(num_labels)]
    features = build_features_from_labels(label_names)

    original_gen = CGLGenerator(
        features=features,
        backbone=config.backbone,
        in_channels=config.in_channels,
        out_channels=config.out_channels,
        num_layers=config.num_layers,
        max_seq_length=config.max_seq_length,
        apply_weight=config.apply_weight,
        d_model=config.d_model,
        use_reorder=config.use_reorder,
        use_reorder_for_random=config.use_reorder_for_random,
        auxilary_task=config.auxilary_task,
    )
    original_gen.eval()

    wrapper_gen = RalfCGLGANGeneratorModel(
        config=config, label_names=label_names, features=features
    )
    wrapper_gen.model.load_state_dict(original_gen.state_dict())
    wrapper_gen.eval()

    batch = _add_images(
        _dummy_layout(batch_size, max_seq_length, num_labels), height=350, width=240
    )
    inputs, _ = original_gen.preprocess(batch)

    out_orig = original_gen(inputs)
    out_wrap = wrapper_gen(inputs)

    assert torch.allclose(out_orig["pred_logits"], out_wrap["pred_logits"])
    assert torch.allclose(out_orig["pred_boxes"], out_wrap["pred_boxes"])

    original_dis = CGLDiscriminator(
        features=features,
        backbone=config.dis_backbone,
        in_channels=config.dis_in_channels,
        out_channels=config.dis_out_channels,
        num_layers=config.dis_num_layers,
        d_model=config.dis_d_model,
        max_seq_length=config.max_seq_length,
    )
    original_dis.set_argmax(config.use_reorder)
    original_dis.eval()

    wrapper_dis = RalfCGLGANDiscriminatorModel(
        config=config, label_names=label_names, features=features
    )
    wrapper_dis.model.load_state_dict(original_dis.state_dict())
    wrapper_dis.model.set_argmax(config.use_reorder)
    wrapper_dis.eval()

    out_orig_dis = original_dis(inputs["image"], inputs["layout"])
    out_wrap_dis = wrapper_dis(inputs["image"], inputs["layout"])

    assert torch.allclose(out_orig_dis, out_wrap_dis)


def test_dsgan_wrapper_matches_original() -> None:
    if not _ensure_precomputed_weights():
        pytest.skip("Precomputed resnet50 weights not found")
    torch.manual_seed(0)
    np.random.seed(0)
    random.seed(0)
    batch_size = 1
    num_labels = 3
    max_seq_length = 10

    config = RalfDSGANConfig(
        backbone="resnet50",
        in_channels=8,
        out_channels=32,
        num_lstm_layers=4,
        max_seq_length=max_seq_length,
        apply_weight=False,
        use_reorder=False,
        use_reorder_for_random=False,
        dis_backbone="resnet50",
        dis_in_channels=8,
        dis_out_channels=32,
        dis_num_lstm_layers=2,
        dis_d_model=128,
    )
    label_names = [str(i) for i in range(num_labels)]
    features = build_features_from_labels(label_names)

    original_gen = DSGenerator(
        features=features,
        d_model=config.d_model,
        backbone=config.backbone,
        in_channels=config.in_channels,
        out_channels=config.out_channels,
        num_lstm_layers=config.num_lstm_layers,
        max_seq_length=config.max_seq_length,
        apply_weight=config.apply_weight,
        use_reorder=config.use_reorder,
        use_reorder_for_random=config.use_reorder_for_random,
    )
    original_gen.eval()

    wrapper_gen = RalfDSGANGeneratorModel(
        config=config, label_names=label_names, features=features
    )
    wrapper_gen.model.load_state_dict(original_gen.state_dict())
    wrapper_gen.eval()

    batch = _add_images(
        _dummy_layout(batch_size, max_seq_length, num_labels), height=350, width=240
    )
    inputs, _ = original_gen.preprocess(batch)

    out_orig = original_gen(inputs)
    out_wrap = wrapper_gen(inputs)

    assert torch.allclose(out_orig["pred_logits"], out_wrap["pred_logits"])
    assert torch.allclose(out_orig["pred_boxes"], out_wrap["pred_boxes"])

    original_dis = DSDiscriminator(
        features=features,
        backbone=config.dis_backbone,
        in_channels=config.dis_in_channels,
        out_channels=config.dis_out_channels,
        num_lstm_layers=config.dis_num_lstm_layers,
        d_model=config.dis_d_model,
    )
    original_dis.set_argmax(config.use_reorder)
    original_dis.eval()

    wrapper_dis = RalfDSGANDiscriminatorModel(
        config=config, label_names=label_names, features=features
    )
    wrapper_dis.model.load_state_dict(original_dis.state_dict())
    wrapper_dis.model.set_argmax(config.use_reorder)
    wrapper_dis.eval()

    out_orig_dis = original_dis(inputs["image"], inputs["layout"])
    out_wrap_dis = wrapper_dis(inputs["image"], inputs["layout"])

    assert torch.allclose(out_orig_dis, out_wrap_dis)
