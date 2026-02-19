from typing import Any, Optional

import torch
from torch import Tensor

from ralf.transformers.internal.models.common.base_model import (
    ConditionalInputsForDiscreteLayout,
)
from ralf.transformers.internal.models.maskgit import MaskGIT

from ..modeling_utils import RalfPreTrainedModel, build_features_from_tokenizer
from ..ralf.tokenization_ralf import RalfTokenizer
from .configuration_maskgit import RalfMaskGITConfig


class RalfMaskGITPreTrainedModel(RalfPreTrainedModel):
    config_class = RalfMaskGITConfig
    base_model_prefix = "model"
    tokenizer_class = RalfTokenizer


class RalfMaskGITModel(RalfMaskGITPreTrainedModel):
    def __init__(
        self,
        config: RalfMaskGITConfig,
        tokenizer: Optional[RalfTokenizer] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(config)
        if tokenizer is None:
            raise ValueError("tokenizer is required")
        self.tokenizer = tokenizer
        features = build_features_from_tokenizer(tokenizer)
        self.model = MaskGIT(
            features=features,
            tokenizer=tokenizer,
            d_model=config.d_model,
            mask_schedule=config.mask_schedule,
            use_padding_as_vocab=config.use_padding_as_vocab,
            use_gumbel_noise=config.use_gumbel_noise,
            pad_weight=config.pad_weight,
        )
        self.model.num_timesteps = config.num_timesteps
        self.post_init()

    def forward(
        self,
        input_ids: Optional[Tensor] = None,
        attention_mask: Optional[Tensor] = None,
        image: Optional[Tensor] = None,
        seq: Optional[Tensor] = None,
        tgt_key_padding_mask: Optional[Tensor] = None,
        **kwargs: Any,
    ) -> dict[str, Tensor]:
        if seq is None:
            seq = input_ids
        if seq is None:
            raise ValueError("seq or input_ids must be provided")
        if tgt_key_padding_mask is None and attention_mask is not None:
            tgt_key_padding_mask = ~attention_mask.bool()
        inputs = {
            "seq": seq,
            "tgt_key_padding_mask": tgt_key_padding_mask,
            "image": image,
        }
        inputs.update({k: v for k, v in kwargs.items() if v is not None})
        return self.model(inputs)

    @torch.no_grad()
    def sample(
        self,
        cond: ConditionalInputsForDiscreteLayout,
        batch_size: Optional[int] = None,
        sampling_cfg: Any = None,
        **kwargs: Any,
    ) -> dict[str, Tensor]:
        return self.model.sample(
            cond=cond, batch_size=batch_size, sampling_cfg=sampling_cfg, **kwargs
        )

    def train_loss(self, *args: Any, **kwargs: Any):
        return self.model.train_loss(*args, **kwargs)
