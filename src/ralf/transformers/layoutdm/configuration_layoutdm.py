from transformers import PretrainedConfig


class RalfLayoutDMConfig(PretrainedConfig):
    model_type = "ralf_layoutdm"

    def __init__(
        self,
        d_model: int = 256,
        num_timesteps: int = 50,
        pos_emb: str = "elem_attr",
        auxiliary_loss_weight: float = 1e-1,
        q_type: str = "constrained",
        retrieval_augmentation: bool = False,
        top_k: int = 16,
        random_retrieval: bool = False,
        saliency_k: int | str = 8,
        use_reference_image: bool = False,
        retrieval_backbone: str = "dreamsim",
        max_position_embeddings: int = 256,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.d_model = d_model
        self.num_timesteps = num_timesteps
        self.pos_emb = pos_emb
        self.auxiliary_loss_weight = auxiliary_loss_weight
        self.q_type = q_type
        self.retrieval_augmentation = retrieval_augmentation
        self.top_k = top_k
        self.random_retrieval = random_retrieval
        self.saliency_k = saliency_k
        self.use_reference_image = use_reference_image
        self.retrieval_backbone = retrieval_backbone
        self.max_position_embeddings = max_position_embeddings
