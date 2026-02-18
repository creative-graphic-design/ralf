import torch

from ralf.train.helpers.visualizer import (
    get_colors,
    mask_out_bbox_area,
    render,
    save_image,
)


def test_visualizer_render_and_mask(small_batch, features) -> None:
    num_classes = features["label"].feature.num_classes
    colors = get_colors(num_classes)
    assert len(colors) == num_classes

    prediction = {
        "image": small_batch["image"],
        "label": small_batch["label"],
        "mask": small_batch["mask"],
        "center_x": small_batch["center_x"],
        "center_y": small_batch["center_y"],
        "width": small_batch["width"],
        "height": small_batch["height"],
    }
    rendered = render(
        prediction=prediction,
        label_feature=features["label"].feature,
        use_grid=False,
    )
    assert rendered is not None
    assert rendered.shape[0] == small_batch["image"].shape[0]

    batch_bboxes = torch.stack(
        [
            small_batch["center_x"],
            small_batch["center_y"],
            small_batch["width"],
            small_batch["height"],
        ],
        dim=-1,
    )
    grid = save_image(
        batch_images=small_batch["image"],
        batch_bboxes=batch_bboxes,
        batch_labels=small_batch["label"],
        batch_masks=small_batch["mask"],
        colors=colors,
        use_grid=True,
    )
    assert grid is not None

    masked = mask_out_bbox_area(
        torch.zeros(1, 3, 8, 8),
        torch.zeros(1, 1, 4),
    )
    assert masked.shape == (1, 3, 8, 8)
