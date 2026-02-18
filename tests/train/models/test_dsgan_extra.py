import torch

from ralf.train.models.dsgan import CNN_LSTM, DSDiscriminator, DSGenerator


def test_cnn_lstm_forward() -> None:
    model = CNN_LSTM(in_channels=8, out_channels=4, d_model=8, num_lstm_layers=1)
    layout = torch.rand(2, 5, 2, 4)
    h0 = torch.zeros(2, 2, 8)
    out = model(layout, h0)
    assert out.shape[:2] == (2, 5)


def test_dsgenerator_decode_and_update(features) -> None:
    gen = DSGenerator(
        features=features,
        d_model=8,
        backbone="resnet50",
        in_channels=8,
        out_channels=4,
        num_lstm_layers=1,
        max_seq_length=5,
        apply_weight=False,
    )
    h0 = torch.zeros(2, 2, 8)
    layout = torch.rand(2, 5, 2, 4)
    outputs = gen.decode(h0, layout)
    assert outputs["pred_boxes"].shape[-1] == 4
    gen.update_per_epoch(epoch=1, warmup_dis_epoch=5, max_epoch=10)
    gen.update_per_epoch(epoch=6, warmup_dis_epoch=5, max_epoch=10)

    _ = DSGenerator(
        features=features,
        d_model=8,
        backbone="resnet50",
        in_channels=10,
        out_channels=4,
        num_lstm_layers=1,
        max_seq_length=5,
        apply_weight=False,
    )


def test_dsdiscriminator_forward(small_batch, features) -> None:
    dis = DSDiscriminator(
        features=features,
        backbone="resnet50",
        d_model=8,
        in_channels=8,
        out_channels=4,
        num_lstm_layers=1,
    )
    dis.set_argmax(use_reorder=False)
    image = torch.cat([small_batch["image"], small_batch["saliency"]], dim=1)
    layout = torch.rand(image.size(0), 10, 2, 4)
    out = dis(image, layout)
    assert out.shape[0] == image.size(0)
