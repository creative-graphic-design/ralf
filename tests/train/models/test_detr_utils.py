import torch

from ralf.train.models.detr.util import box_ops, misc


def test_box_ops_roundtrip() -> None:
    cxcywh = torch.tensor([[0.5, 0.5, 0.2, 0.4]])
    xyxy = box_ops.box_cxcywh_to_xyxy(cxcywh)
    back = box_ops.box_xyxy_to_cxcywh(xyxy)
    assert torch.allclose(cxcywh, back, atol=1e-6)


def test_box_ops_iou_and_masks() -> None:
    boxes1 = torch.tensor([[0.0, 0.0, 1.0, 1.0]])
    boxes2 = torch.tensor([[0.5, 0.5, 1.0, 1.0]])
    iou, union = box_ops.box_iou(boxes1, boxes2)
    assert iou.shape == (1, 1)
    assert union.shape == (1, 1)
    giou = box_ops.generalized_box_iou(boxes1, boxes2)
    assert giou.shape == (1, 1)

    mask = torch.zeros((1, 4, 4), dtype=torch.float32)
    mask[0, 1:3, 1:3] = 1.0
    boxes = box_ops.masks_to_boxes(mask)
    assert boxes.shape == (1, 4)


def test_misc_smoothed_value_and_logger() -> None:
    smoothed = misc.SmoothedValue(window_size=3)
    smoothed.update(1.0)
    smoothed.update(3.0)
    assert smoothed.median >= 1.0
    logger = misc.MetricLogger()
    logger.update(loss=1.0)
    output = list(logger.log_every([1, 2], print_freq=1, header="test"))
    assert output == [1, 2]


def test_misc_nested_tensor_and_accuracy() -> None:
    tensors = [torch.zeros(3, 5, 4), torch.zeros(3, 5, 4)]
    nested = misc.nested_tensor_from_tensor_list(tensors)
    assert nested.tensors.shape[0] == 2
    output = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    target = torch.tensor([0, 1])
    top1 = misc.accuracy(output, target, topk=(1,))
    assert len(top1) == 1


def test_misc_interpolate() -> None:
    x = torch.zeros(1, 1, 2, 2)
    out = misc.interpolate(x, size=[4, 4], mode="nearest")
    assert out.shape[-2:] == (4, 4)


def test_misc_additional_helpers() -> None:
    reduced = misc.reduce_dict({"loss": torch.tensor(1.0)})
    assert reduced["loss"].item() == 1.0
    gathered = misc.all_gather({"id": 1})
    assert gathered == [{"id": 1}]

    maxes = misc._max_by_axis([[1, 3, 2], [2, 1, 4]])
    assert maxes == [2, 3, 4]

    batch = [
        (torch.zeros(3, 2, 2), torch.tensor(0)),
        (torch.ones(3, 2, 2), torch.tensor(1)),
    ]
    nested, targets = misc.collate_fn(batch)
    assert nested.tensors.shape[0] == 2
    assert len(targets) == 2

    moved = nested.to(torch.device("cpu"))
    assert moved.tensors.device.type == "cpu"

    sha = misc.get_sha()
    assert "sha:" in sha

    import builtins

    orig_print = builtins.print
    misc.setup_for_distributed(is_master=True)
    try:
        builtins.print("ping", force=True)
    finally:
        builtins.print = orig_print

    args = type("Args", (), {"dist_url": "env://"})()
    misc.init_distributed_mode(args)
    assert args.distributed is False

    empty_out = torch.zeros(0, 2)
    empty_target = torch.zeros(0, dtype=torch.long)
    acc = misc.accuracy(empty_out, empty_target, topk=(1,))
    assert acc[0].numel() == 1
