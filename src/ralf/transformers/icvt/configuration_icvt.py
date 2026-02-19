from transformers import PretrainedConfig


class RalfICVTConfig(PretrainedConfig):
    model_type = "ralf_icvt"

    def __init__(
        self,
        d_model: int = 256,
        backbone: str = "resnet50",
        ga_type: str | None = None,
        kl_mult: float = 1.0,
        decoder_only: bool = False,
        ignore_bg_bbox_loss: bool = False,
        num_layers: int = 6,
        n_boundaries: int = 128,
        max_seq_length: int = 10,
        max_position_embeddings: int = 256,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.d_model = d_model
        self.backbone = backbone
        self.ga_type = ga_type
        self.kl_mult = kl_mult
        self.decoder_only = decoder_only
        self.ignore_bg_bbox_loss = ignore_bg_bbox_loss
        self.num_layers = num_layers
        self.n_boundaries = n_boundaries
        self.max_seq_length = max_seq_length
        self.max_position_embeddings = max_position_embeddings
