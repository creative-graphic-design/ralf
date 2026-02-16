import pytest
import torch

from ralf.train.models.common.image import ImageFeatureExtractor


@pytest.mark.cuda
def test_image_feature_extractor_forward() -> None:
    model = ImageFeatureExtractor(d_model=32, backbone_name="resnet18")
    model.eval()
    dummy = torch.randn(1, 4, 224, 224)
    with torch.no_grad():
        out = model(dummy)
    assert out.ndim == 4
