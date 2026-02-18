import torch

from ralf.train.models.detr.util import misc as detr_misc


def test_smoothed_value_and_metric_logger() -> None:
    meter = detr_misc.SmoothedValue(window_size=2)
    meter.update(1.0)
    meter.update(3.0)
    assert meter.count == 2
    assert meter.global_avg == 2.0
    assert meter.max == 3.0
    assert "(" in str(meter)

    logger = detr_misc.MetricLogger()
    logger.update(loss=1.5, acc=2.0)
    items = list(logger.log_every(range(2), print_freq=1, header="test"))
    assert items == [0, 1]


def test_nested_tensor_and_collate() -> None:
    t1 = torch.zeros(3, 4, 5)
    t2 = torch.ones(3, 2, 3)
    nested = detr_misc.nested_tensor_from_tensor_list([t1, t2])
    tensor, mask = nested.decompose()
    assert tensor.shape[0] == 2
    assert mask.shape[0] == 2
    moved = nested.to(torch.device("cpu"))
    assert moved.tensors.device.type == "cpu"

    batch = [(t1, {"id": 1}), (t2, {"id": 2})]
    collated = detr_misc.collate_fn(batch)
    assert isinstance(collated, tuple)


def test_misc_helpers(tmp_path) -> None:
    data = {"a": 1}
    gathered = detr_misc.all_gather(data)
    assert gathered == [data]

    reduced = detr_misc.reduce_dict({"loss": torch.tensor(1.0)})
    assert "loss" in reduced

    _ = detr_misc.get_sha()

    import builtins

    original_print = builtins.print
    detr_misc.setup_for_distributed(is_master=True)
    builtins.print = original_print

    assert detr_misc.is_dist_avail_and_initialized() is False
    assert detr_misc.get_world_size() == 1
    assert detr_misc.get_rank() == 0
    assert detr_misc.is_main_process() is True

    out_path = tmp_path / "saved.pt"
    detr_misc.save_on_master({"x": 1}, str(out_path))
    assert out_path.exists()


def test_misc_onnx_nested_tensor_and_helpers(capsys) -> None:
    t1 = torch.zeros(3, 4, 5)
    t2 = torch.ones(3, 2, 3)
    nested = detr_misc._onnx_nested_tensor_from_tensor_list([t1, t2])
    tensor, mask = nested.decompose()
    assert tensor.shape == (2, 3, 4, 5)
    assert mask.dtype == torch.bool

    maxes = detr_misc._max_by_axis([[1, 5, 2], [3, 2, 4]])
    assert maxes == [3, 5, 4]

    logger = detr_misc.MetricLogger()
    try:
        _ = logger.not_a_meter
    except AttributeError as exc:
        assert "has no attribute" in str(exc)

    import builtins

    original_print = builtins.print
    detr_misc.setup_for_distributed(is_master=False)
    try:
        print("hidden")
        print("forced", force=True)
        captured = capsys.readouterr().out
        assert "forced" in captured
    finally:
        builtins.print = original_print


def test_nested_tensor_invalid_ndim() -> None:
    t1 = torch.zeros(4, 5)
    try:
        detr_misc.nested_tensor_from_tensor_list([t1])
    except ValueError as exc:
        assert "not supported" in str(exc)


def test_misc_more_branches(tmp_path) -> None:
    logger = detr_misc.MetricLogger()
    logger.update(loss=torch.tensor(1.0))
    assert isinstance(logger.loss, detr_misc.SmoothedValue)
    assert logger.__getattr__("delimiter") == "\t"
    logger.add_meter("extra", detr_misc.SmoothedValue())
    logger.synchronize_between_processes()

    nested = detr_misc.NestedTensor(torch.zeros(1, 1, 1, 1), None)
    moved = nested.to(torch.device("cpu"))
    assert moved.mask is None
    _ = repr(moved)

    import os

    old_path = os.environ.get("PATH", "")
    os.environ["PATH"] = ""
    try:
        msg = detr_misc.get_sha()
    finally:
        os.environ["PATH"] = old_path
    assert "sha:" in msg


def test_init_distributed_mode_no_env() -> None:
    class Args:
        dist_url = "env://"

    args = Args()
    detr_misc.init_distributed_mode(args)
    assert args.distributed is False


def test_accuracy_and_interpolate() -> None:
    output = torch.tensor([[0.1, 0.9], [0.8, 0.2]])
    target = torch.tensor([1, 0])
    top1 = detr_misc.accuracy(output, target, topk=(1,))[0]
    assert top1 >= 0

    empty_target = torch.tensor([], dtype=torch.long)
    empty_out = torch.zeros(0, 2)
    empty_acc = detr_misc.accuracy(empty_out, empty_target, topk=(1,))[0]
    assert empty_acc.item() == 0.0

    x = torch.randn(1, 3, 4, 4)
    y = detr_misc.interpolate(x, size=(2, 2), mode="nearest")
    assert y.shape[-2:] == (2, 2)
