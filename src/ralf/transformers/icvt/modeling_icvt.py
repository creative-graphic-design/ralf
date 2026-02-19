from typing import Any, Optional

import torch

from ralf.transformers.internal.models.icvt import ICVTGenerator

from ..modeling_utils import RalfPreTrainedModel, build_features_from_labels
from .configuration_icvt import RalfICVTConfig
from .tokenization_icvt import ICVTTokenizer


class RalfICVTPreTrainedModel(RalfPreTrainedModel):
    config_class = RalfICVTConfig
    base_model_prefix = "model"
    tokenizer_class = ICVTTokenizer


class RalfICVTModel(RalfICVTPreTrainedModel):
    def __init__(
        self,
        config: RalfICVTConfig,
        tokenizer: Optional[ICVTTokenizer] = None,
        label_names: Optional[list[str]] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(config)
        if tokenizer is None:
            if label_names is None:
                raise ValueError("tokenizer or label_names is required")
            tokenizer = ICVTTokenizer(
                label_names=label_names,
                n_boundaries=config.n_boundaries,
                max_seq_length=config.max_seq_length,
            )
        self.tokenizer = tokenizer
        features = build_features_from_labels(tokenizer.label_names)
        self.model = ICVTGenerator(
            features=features,
            d_model=config.d_model,
            backbone=config.backbone,
            ga_type=config.ga_type,
            kl_mult=config.kl_mult,
            decoder_only=config.decoder_only,
            ignore_bg_bbox_loss=config.ignore_bg_bbox_loss,
        )
        self.model.tokenizer = tokenizer
        self.post_init()

    def train_loss(self, *args: Any, **kwargs: Any):
        return self.model.train_loss(*args, **kwargs)

    @torch.no_grad()
    def sample(self, *args: Any, **kwargs: Any):
        return self.model.sample(*args, **kwargs)
