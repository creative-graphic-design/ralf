from typing import Any, Optional

import torch
from torch import Tensor

from ralf.transformers.internal.models.dsgan import (
    DSDiscriminator,
    DSGenerator,
    RetrievalAugmentedDSGenerator,
)

from ..modeling_utils import (
    RalfPreTrainedModel,
    build_features_from_labels,
    build_features_from_tokenizer,
)
from ..ralf.tokenization_ralf import RalfTokenizer
from .configuration_dsgan import RalfDSGANConfig


def _resolve_features(
    tokenizer: Optional[RalfTokenizer],
    label_names: Optional[list[str]],
    features: Optional[Any],
):
    if features is not None:
        return features
    if tokenizer is not None:
        return build_features_from_tokenizer(tokenizer)
    if label_names is not None:
        return build_features_from_labels(label_names)
    raise ValueError("features, tokenizer, or label_names must be provided")


class RalfDSGANPreTrainedModel(RalfPreTrainedModel):
    config_class = RalfDSGANConfig
    base_model_prefix = "model"
    tokenizer_class = RalfTokenizer


class RalfDSGANGeneratorModel(RalfDSGANPreTrainedModel):
    def __init__(
        self,
        config: RalfDSGANConfig,
        tokenizer: Optional[RalfTokenizer] = None,
        label_names: Optional[list[str]] = None,
        features: Optional[Any] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(config)
        features = _resolve_features(tokenizer, label_names, features)
        self.model = DSGenerator(
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
        self.post_init()

    def forward(self, inputs: dict[str, Tensor]) -> dict[str, Tensor]:
        return self.model(inputs)

    @torch.no_grad()
    def sample(self, *args: Any, **kwargs: Any):
        return self.model.sample(*args, **kwargs)


class RalfRetrievalAugmentedDSGANGeneratorModel(RalfDSGANPreTrainedModel):
    def __init__(
        self,
        config: RalfDSGANConfig,
        tokenizer: Optional[RalfTokenizer] = None,
        label_names: Optional[list[str]] = None,
        features: Optional[Any] = None,
        db_dataset: Any = None,
        dataset_name: Optional[str] = None,
        max_seq_length: Optional[int] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(config)
        if db_dataset is None or dataset_name is None:
            raise ValueError("db_dataset and dataset_name are required")
        features = _resolve_features(tokenizer, label_names, features)
        self.model = RetrievalAugmentedDSGenerator(
            features=features,
            d_model=config.d_model,
            backbone=config.backbone,
            in_channels=config.in_channels,
            out_channels=config.out_channels,
            num_lstm_layers=config.num_lstm_layers,
            max_seq_length=max_seq_length or config.max_seq_length,
            apply_weight=config.apply_weight,
            use_reorder=config.use_reorder,
            use_reorder_for_random=config.use_reorder_for_random,
            db_dataset=db_dataset,
            top_k=config.top_k,
            dataset_name=dataset_name,
            retrieval_backbone=config.retrieval_backbone,
            random_retrieval=config.random_retrieval,
            saliency_k=config.saliency_k,
            use_reference_image=config.use_reference_image,
        )
        self.post_init()

    def forward(self, inputs: dict[str, Tensor]) -> dict[str, Tensor]:
        return self.model(inputs)

    @torch.no_grad()
    def sample(self, *args: Any, **kwargs: Any):
        return self.model.sample(*args, **kwargs)


class RalfDSGANDiscriminatorModel(RalfDSGANPreTrainedModel):
    def __init__(
        self,
        config: RalfDSGANConfig,
        tokenizer: Optional[RalfTokenizer] = None,
        label_names: Optional[list[str]] = None,
        features: Optional[Any] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(config)
        features = _resolve_features(tokenizer, label_names, features)
        self.model = DSDiscriminator(
            features=features,
            backbone=config.dis_backbone,
            in_channels=config.dis_in_channels,
            out_channels=config.dis_out_channels,
            num_lstm_layers=config.dis_num_lstm_layers,
            d_model=config.dis_d_model,
        )
        self.post_init()

    def forward(self, img: Tensor, layout: Tensor) -> Tensor:  # type: ignore
        return self.model(img, layout)
