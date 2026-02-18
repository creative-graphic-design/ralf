import pytest
import torch


@pytest.mark.cuda
def test_inception_forward_cpu() -> None:
    from ralf.train.helpers.metric import SingletonTimmInceptionV3

    model = SingletonTimmInceptionV3()
    model.eval()
    device = next(model.parameters()).device
    x = torch.rand(1, 3, 299, 299, device=device)
    with torch.no_grad():
        out = model(x)
    assert out.shape[1] == 2048
