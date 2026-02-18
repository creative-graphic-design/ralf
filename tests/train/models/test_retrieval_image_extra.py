import numpy as np
import torch

from ralf.train.models.retrieval.image import FeatureExtracterBackbone, coarse_saliency


def _as_tensor(value: object) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        return value
    if isinstance(value, np.ndarray):
        return torch.from_numpy(value)
    return torch.tensor(value)


def test_coarse_saliency_and_feature_extractor(dataset) -> None:
    ds_dict, _ = dataset
    example = ds_dict["train"][0]
    saliency = _as_tensor(example["saliency"])
    if saliency.ndim == 2:
        saliency = saliency.unsqueeze(0)

    vec = coarse_saliency(saliency)
    assert isinstance(vec, np.ndarray)
    assert vec.ndim == 1

    db_dataset = ds_dict["train"].select(range(min(2, len(ds_dict["train"]))))
    backbone = FeatureExtracterBackbone(
        db_dataset=db_dataset, retrieval_backbone="saliency"
    )
    vectors = backbone.extract_dataset_features()
    assert vectors.shape[0] == len(db_dataset)
    assert vectors.shape[1] == vec.shape[0]

    query = backbone.get_query(
        {"saliency": saliency, "image": _as_tensor(example["image"])}
    )
    assert query.shape == vec.shape


def test_feature_extractor_invalid_backbone(dataset) -> None:
    ds_dict, _ = dataset
    db_dataset = ds_dict["train"].select(range(1))
    try:
        FeatureExtracterBackbone(db_dataset=db_dataset, retrieval_backbone="unknown")
    except ValueError as exc:
        assert "not supported" in str(exc)
