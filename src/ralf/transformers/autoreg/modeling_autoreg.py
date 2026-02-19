from typing import Any, Optional

import torch
from torch import Tensor

from ralf.transformers.internal.models.autoreg import ConcateAuxilaryTaskAutoreg
from ralf.transformers.internal.models.common.base_model import (
    ConditionalInputsForDiscreteLayout,
)

from ..modeling_utils import RalfPreTrainedModel, build_features_from_tokenizer
from ..ralf.tokenization_ralf import RalfTokenizer
from .configuration_autoreg import RalfAutoregConfig


class RalfAutoregPreTrainedModel(RalfPreTrainedModel):
    config_class = RalfAutoregConfig
    base_model_prefix = "model"
    tokenizer_class = RalfTokenizer


class RalfAutoregModel(RalfAutoregPreTrainedModel):
    def __init__(
        self,
        config: RalfAutoregConfig,
        tokenizer: Optional[RalfTokenizer] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(config)
        if tokenizer is None:
            raise ValueError("tokenizer is required")
        self.tokenizer = tokenizer
        features = build_features_from_tokenizer(tokenizer)
        self.model = ConcateAuxilaryTaskAutoreg(
            features=features,
            tokenizer=tokenizer,
            d_model=config.d_model,
            encoder_pos_emb=config.encoder_pos_emb,
            decoder_pos_emb=config.decoder_pos_emb,
            weight_init=config.weight_init,
            shared_embedding=config.shared_embedding,
            decoder_num_layers=config.decoder_num_layers,
            decoder_d_model=config.decoder_d_model,
            auxilary_task=config.auxilary_task,
            use_flag_embedding=config.use_flag_embedding,
            use_multitask=config.use_multitask,
            RELATION_SIZE=config.relation_size,
            global_task_embedding=config.global_task_embedding,
        )
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
