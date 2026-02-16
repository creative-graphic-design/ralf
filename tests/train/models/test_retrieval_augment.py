import torch

from ralf.train.models.common.retrieval_augment import RetrievalAugmentation


def test_retrieval_augmentation_forward(small_batch, features, dataset) -> None:
    ds_dict, _ = dataset
    retrieval = RetrievalAugmentation(
        d_model=16,
        dataset_name="cgl",
        top_k=1,
        num_classes=features["label"].feature.num_classes,
        max_seq_length=10,
        use_reference_image=False,
    )
    retrieved = {
        "image": small_batch["image"].unsqueeze(1),
        "saliency": small_batch["saliency"].unsqueeze(1),
        "center_x": small_batch["center_x"].unsqueeze(1),
        "center_y": small_batch["center_y"].unsqueeze(1),
        "width": small_batch["width"].unsqueeze(1),
        "height": small_batch["height"].unsqueeze(1),
        "label": small_batch["label"].unsqueeze(1),
        "mask": small_batch["mask"].unsqueeze(1),
    }
    retrieved = retrieval.preprocess_retrieved_samples(retrieved)
    img_feature = torch.zeros(small_batch["image"].size(0), 4, 16)
    memory = retrieval(
        image_backbone=None, img_feature=img_feature, retrieved_layouts=retrieved
    )
    assert memory.shape[0] == small_batch["image"].size(0)
