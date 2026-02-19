from typing import Any, Optional

import torch
from torch import Tensor

from ralf.transformers.internal.models.common.base_model import (
    ConditionalInputsForDiscreteLayout,
)
from ralf.transformers.internal.models.layoutdm import (
    LayoutDM,
    RetrievalAugmentedLayoutDM,
)

from ..modeling_utils import RalfPreTrainedModel, build_features_from_tokenizer
from ..ralf.tokenization_ralf import RalfTokenizer
from .configuration_layoutdm import RalfLayoutDMConfig


class RalfLayoutDMPreTrainedModel(RalfPreTrainedModel):
    config_class = RalfLayoutDMConfig
    base_model_prefix = "model"
    tokenizer_class = RalfTokenizer


class RalfLayoutDMModel(RalfLayoutDMPreTrainedModel):
    def __init__(
        self,
        config: RalfLayoutDMConfig,
        tokenizer: Optional[RalfTokenizer] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(config)
        if tokenizer is None:
            raise ValueError("tokenizer is required")
        self.tokenizer = tokenizer
        features = build_features_from_tokenizer(tokenizer)
        self.model = LayoutDM(
            features=features,
            tokenizer=tokenizer,
            d_model=config.d_model,
            num_timesteps=config.num_timesteps,
            pos_emb=config.pos_emb,
            auxiliary_loss_weight=config.auxiliary_loss_weight,
            q_type=config.q_type,
            retrieval_augmentation=config.retrieval_augmentation,
        )
        self.post_init()

    def train_loss(self, *args: Any, **kwargs: Any):
        return self.model.train_loss(*args, **kwargs)

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


class RalfRetrievalAugmentedLayoutDMModel(RalfLayoutDMPreTrainedModel):
    def __init__(
        self,
        config: RalfLayoutDMConfig,
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
        self.model = RetrievalAugmentedLayoutDM(
            features=features,
            tokenizer=tokenizer,
            d_model=config.d_model,
            num_timesteps=config.num_timesteps,
            pos_emb=config.pos_emb,
            auxiliary_loss_weight=config.auxiliary_loss_weight,
            q_type=config.q_type,
            retrieval_augmentation=True,
            db_dataset=db_dataset,
            top_k=config.top_k,
            dataset_name=dataset_name,
            retrieval_backbone=config.retrieval_backbone,
            random_retrieval=config.random_retrieval,
            saliency_k=config.saliency_k,
            use_reference_image=config.use_reference_image,
            max_seq_length=max_seq_length,
        )
        self.post_init()

    def train_loss(self, *args: Any, **kwargs: Any):
        return self.model.train_loss(*args, **kwargs)

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
