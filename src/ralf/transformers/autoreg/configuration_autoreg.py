from transformers import PretrainedConfig


class RalfAutoregConfig(PretrainedConfig):
    model_type = "ralf_autoreg"

    def __init__(
        self,
        d_model: int = 256,
        encoder_pos_emb: str = "sine",
        decoder_pos_emb: str = "layout",
        weight_init: bool = False,
        shared_embedding: bool = False,
        decoder_num_layers: int = 6,
        decoder_d_model: int = 256,
        auxilary_task: str = "uncond",
        use_flag_embedding: bool = True,
        use_multitask: bool = False,
        relation_size: int = 10,
        global_task_embedding: bool = False,
        max_position_embeddings: int = 256,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.d_model = d_model
        self.encoder_pos_emb = encoder_pos_emb
        self.decoder_pos_emb = decoder_pos_emb
        self.weight_init = weight_init
        self.shared_embedding = shared_embedding
        self.decoder_num_layers = decoder_num_layers
        self.decoder_d_model = decoder_d_model
        self.auxilary_task = auxilary_task
        self.use_flag_embedding = use_flag_embedding
        self.use_multitask = use_multitask
        self.relation_size = relation_size
        self.global_task_embedding = global_task_embedding
        self.max_position_embeddings = max_position_embeddings
