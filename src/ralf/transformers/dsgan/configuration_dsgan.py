from transformers import PretrainedConfig


class RalfDSGANConfig(PretrainedConfig):
    model_type = "ralf_dsgan"

    def __init__(
        self,
        d_model: int = 256,
        backbone: str = "resnet50",
        in_channels: int = 8,
        out_channels: int = 32,
        num_lstm_layers: int = 4,
        max_seq_length: int = 10,
        apply_weight: bool = False,
        use_reorder: bool = True,
        use_reorder_for_random: bool = False,
        top_k: int = 16,
        random_retrieval: bool = False,
        saliency_k: int | str = 8,
        use_reference_image: bool = False,
        retrieval_backbone: str = "dreamsim",
        dis_backbone: str = "resnet18",
        dis_in_channels: int = 8,
        dis_out_channels: int = 32,
        dis_num_lstm_layers: int = 2,
        dis_d_model: int = 256,
        max_position_embeddings: int = 256,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.d_model = d_model
        self.backbone = backbone
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_lstm_layers = num_lstm_layers
        self.max_seq_length = max_seq_length
        self.apply_weight = apply_weight
        self.use_reorder = use_reorder
        self.use_reorder_for_random = use_reorder_for_random
        self.top_k = top_k
        self.random_retrieval = random_retrieval
        self.saliency_k = saliency_k
        self.use_reference_image = use_reference_image
        self.retrieval_backbone = retrieval_backbone
        self.dis_backbone = dis_backbone
        self.dis_in_channels = dis_in_channels
        self.dis_out_channels = dis_out_channels
        self.dis_num_lstm_layers = dis_num_lstm_layers
        self.dis_d_model = dis_d_model
        self.max_position_embeddings = max_position_embeddings
