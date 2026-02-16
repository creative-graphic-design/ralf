import torch

from ralf.train.models.retrieval_augmented_autoreg import (
    RetrievalAugmentedAutoregAdapter,
)


def test_retrieval_augmented_autoreg_train_loss(
    small_batch, features, layout_tokenizer, dataset
) -> None:
    ds_dict, _ = dataset
    image = torch.cat([small_batch["image"], small_batch["saliency"]], dim=1)
    retrieved = {
        "image": image.unsqueeze(1),
        "saliency": small_batch["saliency"].unsqueeze(1),
        "center_x": small_batch["center_x"].unsqueeze(1),
        "center_y": small_batch["center_y"].unsqueeze(1),
        "width": small_batch["width"].unsqueeze(1),
        "height": small_batch["height"].unsqueeze(1),
        "label": small_batch["label"].unsqueeze(1),
        "mask": small_batch["mask"].unsqueeze(1),
    }
    inputs = {**small_batch, "retrieved": retrieved}

    model = RetrievalAugmentedAutoregAdapter(
        features=features,
        tokenizer=layout_tokenizer,
        dataset_name="cgl",
        max_seq_length=10,
        db_dataset=ds_dict["train"],
        top_k=1,
        retrieval_backbone="saliency",
        random_retrieval=False,
        saliency_k=1,
        use_reference_image=False,
    )
    model.eval()
    model_inputs, targets = model.preprocess(inputs)
    with torch.no_grad():
        outputs, losses = model.train_loss(model_inputs, targets)
    assert "nll_loss" in losses
    assert outputs["logits"].shape[0] == small_batch["image"].shape[0]


def test_retrieval_augmented_autoreg_sample(
    small_batch, features, layout_tokenizer, dataset
) -> None:
    from omegaconf import OmegaConf

    from ralf.train.models.common.base_model import (
        RetrievalAugmentedConditionalInputsForDiscreteLayout,
    )

    ds_dict, _ = dataset
    image = torch.cat([small_batch["image"], small_batch["saliency"]], dim=1)
    retrieved = {
        "image": image.unsqueeze(1),
        "saliency": small_batch["saliency"].unsqueeze(1),
        "center_x": small_batch["center_x"].unsqueeze(1),
        "center_y": small_batch["center_y"].unsqueeze(1),
        "width": small_batch["width"].unsqueeze(1),
        "height": small_batch["height"].unsqueeze(1),
        "label": small_batch["label"].unsqueeze(1),
        "mask": small_batch["mask"].unsqueeze(1),
    }
    encoded = layout_tokenizer.encode(small_batch)
    cond = RetrievalAugmentedConditionalInputsForDiscreteLayout(
        image=image,
        id=None,
        seq=encoded["seq"],
        mask=encoded["mask"],
        retrieved=retrieved,
    )
    model = RetrievalAugmentedAutoregAdapter(
        features=features,
        tokenizer=layout_tokenizer,
        dataset_name="cgl",
        max_seq_length=10,
        db_dataset=ds_dict["train"],
        top_k=1,
        retrieval_backbone="saliency",
        random_retrieval=False,
        saliency_k=1,
        use_reference_image=False,
    )
    sampling_cfg = OmegaConf.create({"name": "random", "temperature": 1.0})
    output = model.sample(
        cond=cond,
        sampling_cfg=sampling_cfg,
        cond_type="partial",
        use_backtrack=False,
    )
    assert "label" in output
