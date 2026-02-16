import numpy as np
import pytest

from ralf.train.helpers import metric
from ralf.train.inference import _validate_outputs


def _clone_batch(batch):
    keys = [
        "label",
        "mask",
        "center_x",
        "center_y",
        "width",
        "height",
        "image",
        "saliency",
    ]
    return {k: batch[k].clone() for k in keys if k in batch}


def test_metric_core_scores(small_batch, features) -> None:
    batch = _clone_batch(small_batch)
    alignment = metric.compute_alignment(batch)
    overlap = metric.compute_overlap(_clone_batch(small_batch))
    assert "alignment-LayoutGAN++" in alignment
    assert "overlap-LayoutGAN++" in overlap

    overlay = metric.compute_overlay(
        _clone_batch(small_batch), features["label"].feature
    )
    underlay = metric.compute_underlay_effectiveness(
        _clone_batch(small_batch), features["label"].feature
    )
    assert "overlay" in overlay
    assert "underlay_effectiveness_strict" in underlay

    saliency_scores = metric.compute_saliency_aware_metrics(
        _clone_batch(small_batch), features["label"].feature
    )
    assert "occlusion" in saliency_scores
    assert "unreadability" in saliency_scores

    outputs = {
        "label": small_batch["label"],
        "mask": small_batch["mask"],
        "center_x": small_batch["center_x"],
        "center_y": small_batch["center_y"],
        "width": small_batch["width"],
        "height": small_batch["height"],
        "id": small_batch["id"],
    }
    layout_list = _validate_outputs(outputs)
    filtered, validity = metric.compute_validity(layout_list)
    assert 0.0 <= validity <= 1.0
    assert isinstance(filtered, list)


def test_metric_iou_and_generative_scores() -> None:
    box_1 = np.array([[0.5, 0.5, 0.2, 0.2], [0.2, 0.2, 0.1, 0.1]])
    box_2 = np.array([[0.5, 0.5, 0.2, 0.2], [0.3, 0.3, 0.1, 0.1]])
    iou = metric.iou_func_factory("iou")(box_1, box_2)
    giou = metric.iou_func_factory("giou")(box_1, box_2)
    perceptual = metric.iou_func_factory("perceptual")(box_1, box_2)
    assert iou.shape[0] == box_1.shape[0]
    assert giou.shape[0] == box_1.shape[0]
    assert perceptual.shape[0] >= 1

    feats_real = np.random.rand(8, 4)
    feats_fake = np.random.rand(8, 4)
    scores = metric.compute_generative_model_scores(feats_real, feats_fake)
    assert "fid" in scores


def test_metric_rshm_and_sub(small_batch) -> None:
    batch = _clone_batch(small_batch)
    rshm = metric.compute_rshm(batch)
    assert "R_{shm} (vgg distance)" in rshm

    with pytest.raises(NotImplementedError):
        metric.compute_sub(batch, None)
