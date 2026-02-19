from transformers import PretrainedConfig


class RalfMaskGITConfig(PretrainedConfig):
    model_type = "ralf_maskgit"

    def __init__(
        self,
        d_model: int = 256,
        mask_schedule: str = "linear",
        use_padding_as_vocab: bool = True,
        use_gumbel_noise: bool = True,
        pad_weight: float = 1.0,
        num_timesteps: int = 50,
        max_position_embeddings: int = 256,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.d_model = d_model
        self.mask_schedule = mask_schedule
        self.use_padding_as_vocab = use_padding_as_vocab
        self.use_gumbel_noise = use_gumbel_noise
        self.pad_weight = pad_weight
        self.num_timesteps = num_timesteps
        self.max_position_embeddings = max_position_embeddings
