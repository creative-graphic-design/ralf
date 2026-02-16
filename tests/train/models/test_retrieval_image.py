import numpy as np
import torch

from ralf.train.models.retrieval.image import FeatureExtracterBackbone, coarse_saliency


def _drop_indexes(dataset):
    for name in dataset.list_indexes():
        dropped = dataset.drop_index(name)
        if dropped is not None:
            dataset = dropped
    return dataset


def test_saliency_feature_extractor_paths(dataset) -> None:
    ds_dict, _ = dataset
    small_ds = _drop_indexes(ds_dict["train"]).select(range(2))
    backbone = FeatureExtracterBackbone(
        db_dataset=small_ds, retrieval_backbone="saliency"
    )

    vectors = backbone.extract_dataset_features()
    assert isinstance(vectors, np.ndarray)
    assert vectors.shape[0] == len(small_ds)

    query = backbone.get_query(small_ds[0])
    assert query.shape == vectors[0].shape


def test_coarse_saliency_shape_and_range(dataset) -> None:
    ds_dict, _ = dataset
    saliency = ds_dict["train"][0]["saliency"].cpu()
    output = coarse_saliency(saliency, size=(8, 8))
    assert output.shape == (64,)
    assert output.min() >= -1.0 - 1e-6
    assert output.max() <= 1.0 + 1e-6


def test_coarse_saliency_invalid_size(dataset) -> None:
    ds_dict, _ = dataset
    saliency = ds_dict["train"][0]["saliency"].cpu()
    bad_saliency = torch.zeros_like(saliency[:, :-1, :])
    try:
        coarse_saliency(bad_saliency)
    except AssertionError:
        pass
    else:
        assert False, "Expected assertion for invalid saliency shape"


def test_feature_extractor_invalid_backbone(dataset) -> None:
    ds_dict, _ = dataset
    small_ds = _drop_indexes(ds_dict["train"]).select(range(1))
    try:
        FeatureExtracterBackbone(db_dataset=small_ds, retrieval_backbone="unknown")
    except ValueError as exc:
        assert "not supported" in str(exc)
