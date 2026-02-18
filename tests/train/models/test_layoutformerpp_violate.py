import torch

from ralf.train.helpers.relationships import RelLoc
from ralf.train.models.common.base_model import ConditionalInputsForDiscreteLayout
from ralf.train.models.layoutformerpp.violate import (
    calculate_vio_rate_relation,
    calculate_vio_rate_reliable_label_or_size,
    calculate_violation,
    empty_vio_rate,
    remove_unncecessary_tokens,
)


def test_violate_helpers(layout_tokenizer, small_batch) -> None:
    encoded = layout_tokenizer.encode(small_batch)
    cond = ConditionalInputsForDiscreteLayout(
        image=small_batch["image"],
        id=None,
        seq=encoded["seq"],
        mask=encoded["mask"],
    )

    output_seq = encoded["seq"][:, 1:].clone()
    violation = calculate_vio_rate_reliable_label_or_size(
        "c", cond, output_seq, layout_tokenizer
    )
    assert violation["viorated"] == 0

    seq = torch.tensor([1, 2, 3, 4])
    pad_mask = torch.tensor([True, True, False, True])
    cleaned = remove_unncecessary_tokens(seq, pad_mask, pad_id=0, eos_id=3)
    assert cleaned.tolist() == [1, 2, 4]

    assert empty_vio_rate()["total"] == 1


def test_violate_relation_rate(layout_tokenizer, small_batch) -> None:
    encoded = layout_tokenizer.encode(small_batch)
    cond = ConditionalInputsForDiscreteLayout(
        image=small_batch["image"],
        id=None,
        seq=encoded["seq"],
        mask=encoded["mask"],
    )

    batch_size = encoded["seq"].size(0)
    output = {
        "center_x": torch.full((batch_size, 1), 0.2),
        "center_y": torch.full((batch_size, 1), 0.1),
        "width": torch.full((batch_size, 1), 0.2),
        "height": torch.full((batch_size, 1), 0.2),
    }
    prepared = [[[("canvas", RelLoc.TOP)]] for _ in range(batch_size)]
    violation = calculate_vio_rate_relation(cond, output, prepared)
    assert violation["total"] == batch_size

    violation2 = calculate_violation(
        "relation",
        cond,
        output,
        output,
        layout_tokenizer,
        prepared,
    )
    assert violation2["total"] == batch_size
