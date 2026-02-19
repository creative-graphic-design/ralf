from transformers import PretrainedConfig


class RalfRetrievalAugmentedAutoregConfig(PretrainedConfig):
    model_type = "ralf_retrieval_autoreg"

    def __init__(
        self,
        d_model: int = 256,
        encoder_pos_emb: str = "sine",
        decoder_pos_emb: str = "layout",
        weight_init: bool = True,
        top_k: int = 16,
        layout_backbone: str = "feature_extractor",
        use_reference_image: bool = False,
        freeze_layout_encoder: bool = True,
        retrieval_backbone: str = "saliency",
        random_retrieval: bool = False,
        saliency_k: int | str = 8,
        decoder_d_model: int = 256,
        auxilary_task: str = "uncond",
        use_flag_embedding: bool = True,
        use_multitask: bool = False,
        relation_size: int = 10,
        shared_embedding: bool = False,
        global_task_embedding: bool = False,
        max_position_embeddings: int = 256,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.d_model = d_model
        self.encoder_pos_emb = encoder_pos_emb
        self.decoder_pos_emb = decoder_pos_emb
        self.weight_init = weight_init
        self.top_k = top_k
        self.layout_backbone = layout_backbone
        self.use_reference_image = use_reference_image
        self.freeze_layout_encoder = freeze_layout_encoder
        self.retrieval_backbone = retrieval_backbone
        self.random_retrieval = random_retrieval
        self.saliency_k = saliency_k
        self.decoder_d_model = decoder_d_model
        self.auxilary_task = auxilary_task
        self.use_flag_embedding = use_flag_embedding
        self.use_multitask = use_multitask
        self.relation_size = relation_size
        self.shared_embedding = shared_embedding
        self.global_task_embedding = global_task_embedding
        self.max_position_embeddings = max_position_embeddings
