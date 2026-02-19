from typing import Any, Optional

import torch
from torch import Tensor

from ralf.transformers.internal.models.common.base_model import (
    RetrievalAugmentedConditionalInputsForDiscreteLayout,
)
from ralf.transformers.internal.models.retrieval_augmented_autoreg import (
    ConcateAuxilaryTaskConcateCrossAttnRetrievalAugmentedAutoreg,
)

from ..modeling_utils import RalfPreTrainedModel, build_features_from_tokenizer
from .configuration_ralf import RalfRetrievalAugmentedAutoregConfig
from .tokenization_ralf import RalfTokenizer


class RalfRetrievalAugmentedAutoregPreTrainedModel(RalfPreTrainedModel):
    config_class = RalfRetrievalAugmentedAutoregConfig
    base_model_prefix = "model"
    tokenizer_class = RalfTokenizer


class RalfRetrievalAugmentedAutoregModel(RalfRetrievalAugmentedAutoregPreTrainedModel):
    def __init__(
        self,
        config: RalfRetrievalAugmentedAutoregConfig,
        tokenizer: Optional[RalfTokenizer] = None,
        db_dataset: Any = None,
        dataset_name: Optional[str] = None,
        max_seq_length: Optional[int] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(config)
        if tokenizer is None:
            raise ValueError("tokenizer is required")
        if db_dataset is None or dataset_name is None or max_seq_length is None:
            raise ValueError("db_dataset, dataset_name, max_seq_length are required")
        self.tokenizer = tokenizer
        features = build_features_from_tokenizer(tokenizer)
        self.model = ConcateAuxilaryTaskConcateCrossAttnRetrievalAugmentedAutoreg(
            features=features,
            tokenizer=tokenizer,
            dataset_name=dataset_name,
            max_seq_length=max_seq_length,
            db_dataset=db_dataset,
            d_model=config.d_model,
            encoder_pos_emb=config.encoder_pos_emb,
            decoder_pos_emb=config.decoder_pos_emb,
            weight_init=config.weight_init,
            top_k=config.top_k,
            layout_backbone=config.layout_backbone,
            use_reference_image=config.use_reference_image,
            freeze_layout_encoder=config.freeze_layout_encoder,
            retrieval_backbone=config.retrieval_backbone,
            random_retrieval=config.random_retrieval,
            saliency_k=config.saliency_k,
            decoder_d_model=config.decoder_d_model,
            auxilary_task=config.auxilary_task,
            use_flag_embedding=config.use_flag_embedding,
            use_multitask=config.use_multitask,
            RELATION_SIZE=config.relation_size,
            shared_embedding=config.shared_embedding,
            global_task_embedding=config.global_task_embedding,
        )
        self.post_init()

    def forward(
        self,
        input_ids: Optional[Tensor] = None,
        attention_mask: Optional[Tensor] = None,
        image: Optional[Tensor] = None,
        retrieved: Optional[dict[str, Tensor]] = None,
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
            "retrieved": retrieved,
        }
        inputs.update({k: v for k, v in kwargs.items() if v is not None})
        return self.model(inputs)

    @torch.no_grad()
    def sample(
        self,
        cond: RetrievalAugmentedConditionalInputsForDiscreteLayout,
        batch_size: Optional[int] = None,
        sampling_cfg: Any = None,
        **kwargs: Any,
    ) -> dict[str, Tensor]:
        return self.model.sample(
            cond=cond, batch_size=batch_size, sampling_cfg=sampling_cfg, **kwargs
        )

    def train_loss(self, *args: Any, **kwargs: Any):
        return self.model.train_loss(*args, **kwargs)
