import torch

from ralf.train.helpers.task import get_condition
from ralf.train.models.common.base_model import ConditionalInputsForDiscreteLayout
from ralf.train.models.layoutformerpp.backtrack import generate_square_subsequent_mask
from ralf.train.models.layoutformerpp.decoding_space_restriction import (
    DECODE_SPACE_RESTRICTION,
    restrict_only_category,
    restrict_reliable_label_or_size,
)
from ralf.train.models.layoutformerpp.relation_restriction import (
    DiscretizeBoundingBox,
    RelationTypes,
    TransformerSortByDictLabelConstraint,
    TransformerSortByDictRelationConstraint,
)
from ralf.train.models.layoutformerpp.task_preprocessor import RelationshipPreprocessor
from ralf.train.models.layoutformerpp.violate import (
    calculate_violation,
    empty_vio_rate,
    remove_unncecessary_tokens,
)


def test_backtrack_mask() -> None:
    mask = generate_square_subsequent_mask(3)
    assert mask.shape == (3, 3)


def test_decoding_space_restriction() -> None:
    cond = torch.tensor([[2, 0, 0, 0, 0, 0]])
    logits = torch.zeros(1, 5)
    out = restrict_reliable_label_or_size(
        sampling_idx=0,
        cond=cond,
        logits=logits.clone(),
        pad_id=0,
        eos_id=1,
        max_length=5,
    )
    assert torch.isfinite(out).any()
    out2 = restrict_only_category(
        sampling_idx=1,
        cond=cond,
        logits=logits.clone(),
        pad_id=0,
        eos_id=1,
        max_length=5,
    )
    assert torch.isfinite(out2).any()
    assert DECODE_SPACE_RESTRICTION["none"] is not None


def test_relation_restriction_prepare_and_call(small_batch, layout_tokenizer) -> None:
    preprocessor = RelationshipPreprocessor(
        tokenizer=layout_tokenizer, global_task_embedding=False
    )
    label_constraint = TransformerSortByDictLabelConstraint(preprocessor)
    label_constraint.prepare([small_batch["label"][0].tolist()])
    _ = label_constraint(0, 0, torch.tensor([0]))

    cond_inputs, _ = get_condition(small_batch, "relation", layout_tokenizer)
    seq = preprocessor(cond_inputs)["seq"]
    rel_constraint = TransformerSortByDictRelationConstraint(preprocessor)
    rel_constraints = rel_constraint.prepare(seq[0])
    label_token = rel_constraint.label_tokens[0]
    token_ids = torch.tensor([[label_token]])
    _ = rel_constraint(token_ids, rel_constraints)

    discretizer = DiscretizeBoundingBox(num_x_grid=4, num_y_grid=4)
    bbox = torch.tensor([[0.1, 0.1, 0.2, 0.2]])
    discrete = discretizer.discretize(bbox)
    assert discrete.shape == bbox.shape
    _ = discretizer.continuize(discrete)
    _ = RelationTypes.type2index()
    _ = RelationTypes.index2type()


def test_violate_helpers(small_batch, layout_tokenizer) -> None:
    encoded = layout_tokenizer.encode(small_batch)
    seq = encoded["seq"]
    mask = encoded["mask"]
    cond = ConditionalInputsForDiscreteLayout(
        image=torch.zeros(1, 4, 2, 2), id=None, seq=seq, mask=mask
    )
    output_seq = seq[:, 1:]
    out = calculate_violation("c", cond, output_seq, output_seq, layout_tokenizer, [])
    assert isinstance(out, dict)
    removed = remove_unncecessary_tokens(
        seq=seq[0],
        pad_mask=mask[0],
        pad_id=layout_tokenizer.name_to_id("pad"),
        eos_id=layout_tokenizer.name_to_id("eos"),
    )
    assert removed.numel() > 0
    assert empty_vio_rate()["total"] == 1
