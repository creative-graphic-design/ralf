import torch

from ralf.train.models.cgl import (
    CGLDiscriminator,
    CGLGenerator,
    RetrievalAugmentedCGLGenerator,
)


def test_cgl_generator_preprocess_and_decode(small_batch, features) -> None:
    model = CGLGenerator(
        features=features,
        backbone="resnet18",
        d_model=64,
        out_channels=64,
        max_seq_length=10,
        auxilary_task="uncond",
    )
    model.eval()

    for task in [None, "uncond", "c", "cwh", "partial", "refinement"]:
        model.auxilary_task = task
        inputs, targets = model.preprocess(small_batch)
        assert "layout" in inputs
        assert "labels" in targets

    img_feature, layout_feature = model._encode_into_memory(inputs)
    outputs = model.decode(img_feature, layout_feature)
    assert "pred_logits" in outputs
    assert "pred_boxes" in outputs

    model.update_per_epoch(epoch=0, warmup_dis_epoch=1, max_epoch=2)
    model.update_per_epoch(epoch=2, warmup_dis_epoch=1, max_epoch=2)


def test_cgl_discriminator_forward(small_batch, features) -> None:
    d_label = features["label"].feature.num_classes + 1
    model = CGLDiscriminator(
        features=features,
        backbone="resnet18",
        d_model=64,
        out_channels=64,
        max_seq_length=10,
    )
    model.set_argmax(use_reorder=False)
    model.eval()

    img = torch.cat([small_batch["image"], small_batch["saliency"]], dim=1)
    layout = torch.randn(img.size(0), 10, 2, d_label)
    out = model(img, layout)
    assert out.shape[0] == img.size(0)


def test_retrieval_augmented_cgl_generator(small_batch, features, dataset) -> None:
    ds_dict, _ = dataset
    label_max = features["label"].feature.num_classes - 1
    retrieved = {
        "image": small_batch["image"].unsqueeze(1),
        "saliency": small_batch["saliency"].unsqueeze(1),
        "center_x": small_batch["center_x"].unsqueeze(1),
        "center_y": small_batch["center_y"].unsqueeze(1),
        "width": small_batch["width"].unsqueeze(1),
        "height": small_batch["height"].unsqueeze(1),
        "label": torch.clamp(small_batch["label"], max=label_max).unsqueeze(1),
        "mask": small_batch["mask"].unsqueeze(1),
    }
    inputs = {**small_batch, "retrieved": retrieved}

    model = RetrievalAugmentedCGLGenerator(
        features=features,
        backbone="resnet18",
        d_model=64,
        out_channels=64,
        max_seq_length=10,
        auxilary_task="uncond",
        db_dataset=ds_dict["train"],
        top_k=1,
        dataset_name="cgl",
        retrieval_backbone="saliency",
        random_retrieval=False,
        saliency_k=1,
        use_reference_image=False,
    )
    model.eval()
    model_inputs, targets = model.preprocess(inputs)
    assert "retrieved" in model_inputs
    img_feature, layout_feature = model._encode_into_memory(model_inputs)
    outputs = model.decode(img_feature, layout_feature)
    assert "pred_logits" in outputs
