import pytest
import torch

from ralf.train.helpers.relationships import RelLoc, RelSize
from ralf.train.helpers.task import get_condition
from ralf.train.models.layoutformerpp.relation_restriction import (
    DiscretizeBoundingBox,
    RelationTypes,
    TransformerSortByDictConstraintDecodeState,
    TransformerSortByDictLabelConstraint,
    TransformerSortByDictLabelSizeConstraint,
    TransformerSortByDictRelationConstraint,
    define_canvas_restriction,
    detect_label_idx,
)
from ralf.train.models.layoutformerpp.task_preprocessor import RelationshipPreprocessor


def test_relation_types_and_discretize() -> None:
    mapping = RelationTypes.type2index()
    reverse = RelationTypes.index2type()
    assert reverse[mapping[RelationTypes.types[0]]] == RelationTypes.types[0]

    discretizer = DiscretizeBoundingBox(num_x_grid=8, num_y_grid=8)
    bbox = torch.tensor([[0.1, 0.2, 0.6, 0.7]])
    discrete = discretizer.discretize(bbox)
    cont = discretizer.continuize(discrete)
    assert cont.shape == bbox.shape

    mask = torch.zeros(256, dtype=torch.bool)
    mask[131] = True
    canvas_mask = define_canvas_restriction(mask, RelLoc.TOP, total_bin=128)
    assert canvas_mask.any()
    canvas_mask_center = define_canvas_restriction(mask, RelLoc.CENTER, total_bin=128)
    assert canvas_mask_center.any()
    canvas_mask_bottom = define_canvas_restriction(mask, RelLoc.BOTTOM, total_bin=128)
    assert canvas_mask_bottom.any()


def test_discretize_call_and_canvas_error() -> None:
    discretizer = DiscretizeBoundingBox(num_x_grid=8, num_y_grid=8)
    data = {"bboxes": torch.tensor([[0.1, 0.2, 0.6, 0.7]])}
    out = discretizer(data)
    assert "gold_bboxes" in out
    assert out["discrete_bboxes"].shape[-1] == 4

    mask = torch.zeros(256, dtype=torch.bool)
    mask[131] = True
    with pytest.raises(ValueError):
        define_canvas_restriction(mask, "unknown", 128)


def test_label_and_relation_constraints(layout_tokenizer, small_batch) -> None:
    preprocessor = RelationshipPreprocessor(
        tokenizer=layout_tokenizer, global_task_embedding=False
    )

    label_constraint = TransformerSortByDictLabelConstraint(preprocessor)
    label_constraint.prepare([[1, 2, 0]])
    tokens, _ = label_constraint(0, 0, torch.tensor([0]))
    assert len(tokens) > 0

    cond, _ = get_condition(small_batch, "relation", layout_tokenizer)
    seq_constraints = preprocessor(cond)
    rel_constraint = TransformerSortByDictRelationConstraint(preprocessor)
    rel_constraints = rel_constraint.prepare(seq_constraints["seq"][0])
    token_ids = torch.full((1, 1), preprocessor.name_to_id("bos"))
    mask, _ = rel_constraint(token_ids, rel_constraints)
    assert isinstance(mask, torch.Tensor)

    # exercise label size constraint
    size_constraint = TransformerSortByDictLabelSizeConstraint(preprocessor)
    size_constraint.index2label = preprocessor.tokenizer._label_feature.names
    size_constraint.prepare([[1, 2, 0]], [[[0, 0, 1, 1], [0, 0, 2, 2], [0, 0, 0, 0]]])
    tokens, _ = size_constraint(0, 0, torch.tensor([0]))
    assert isinstance(tokens, list)


def test_relation_constraint_progression(layout_tokenizer, small_batch) -> None:
    preprocessor = RelationshipPreprocessor(
        tokenizer=layout_tokenizer, global_task_embedding=False
    )
    cond, _ = get_condition(small_batch, "relation", layout_tokenizer)
    seq_constraints = preprocessor(cond)
    rel_constraint = TransformerSortByDictRelationConstraint(preprocessor)
    rel_constraints = rel_constraint.prepare(seq_constraints["seq"][0])
    empty_constraints = [[] for _ in rel_constraints]

    # step through a few token lengths to exercise branches
    token_ids = torch.tensor(
        [[rel_constraint.label_tokens[0], rel_constraint.label_tokens[0]]]
    )
    mask, _ = rel_constraint(token_ids, empty_constraints)
    assert mask.numel() == rel_constraint.logits_size

    token_ids = torch.tensor(
        [
            [
                rel_constraint.label_tokens[0],
                rel_constraint.label_tokens[0],
                rel_constraint.width_start_idx,
            ]
        ]
    )
    mask, _ = rel_constraint(token_ids, empty_constraints)
    assert mask.numel() == rel_constraint.logits_size


def test_detect_label_idx_and_get_target_bbox(layout_tokenizer, small_batch) -> None:
    preprocessor = RelationshipPreprocessor(
        tokenizer=layout_tokenizer, global_task_embedding=False
    )
    rel_constraint = TransformerSortByDictRelationConstraint(preprocessor)
    types = torch.tensor([preprocessor.name_to_id("text")])
    label_idx = detect_label_idx(types, types[0] + 1, torch.tensor(0), {}, preprocessor)
    assert label_idx == 0

    state = rel_constraint.decode_state = [
        type("State", (), {"pred_bbox": [[1, 1, 1, 1]], "num_bbox": 0})()
    ]
    rel_type, tgt_bbox, back_idx = rel_constraint.get_target_bbox(
        "canvas", RelLoc.TOP, state[0]
    )
    assert tgt_bbox is None
    rel_type, tgt_bbox, back_idx = rel_constraint.get_target_bbox(
        RelLoc.LEFT, None, state[0]
    )
    assert tgt_bbox is not None

    missing = detect_label_idx(
        torch.tensor([1, 2, 3]),
        torch.tensor(999),
        torch.tensor(0),
        {},
        preprocessor,
    )
    assert missing == 0


def test_label_constraints_with_sep(layout_tokenizer) -> None:
    preprocessor = RelationshipPreprocessor(
        tokenizer=layout_tokenizer, global_task_embedding=False
    )
    label_constraint = TransformerSortByDictLabelConstraint(preprocessor)
    label_constraint.add_sep_token = True
    label_constraint.prepare([[1, 2, 0]])
    state = TransformerSortByDictConstraintDecodeState(num_elements=1)
    state.next_token_type = state.SEP
    label_constraint.decode_state = [[state]]
    tokens, _ = label_constraint(0, 0, torch.tensor([0]))
    assert isinstance(tokens, list)

    size_constraint = TransformerSortByDictLabelSizeConstraint(preprocessor)
    size_constraint.add_sep_token = True
    size_constraint.index2label = preprocessor.tokenizer._label_feature.names
    size_constraint.prepare([[1, 2, 0]], [[[0, 0, 1, 1], [0, 0, 2, 2], [0, 0, 0, 0]]])
    state = TransformerSortByDictConstraintDecodeState(num_elements=1)
    state.next_token_type = state.SEP
    size_constraint.decode_state = [[state]]
    tokens, _ = size_constraint(0, 0, torch.tensor([0]))
    assert isinstance(tokens, list)


def test_relation_constraint_branching(layout_tokenizer, small_batch) -> None:
    preprocessor = RelationshipPreprocessor(
        tokenizer=layout_tokenizer, global_task_embedding=False
    )
    rel_constraint = TransformerSortByDictRelationConstraint(preprocessor)
    cond, _ = get_condition(small_batch, "relation", layout_tokenizer)
    seq_constraints = preprocessor(cond)
    rel_constraint.prepare(seq_constraints["seq"][0])

    def _state(curr_element: int, pred_bbox: list[list[int]]):
        state = TransformerSortByDictConstraintDecodeState(curr_element)
        state.curr_element = curr_element
        state.pred_bbox = pred_bbox
        return state

    rel_constraint.decode_state = [
        _state(1, [[10, 12, 20, 22]]),
    ]
    token_ids = torch.tensor(
        [
            [
                rel_constraint.label_tokens[0],
                rel_constraint.width_start_idx,
                rel_constraint.height_start_idx,
                rel_constraint.center_x_start_idx,
                rel_constraint.center_y_start_idx,
            ]
        ]
    )
    mask, _ = rel_constraint(token_ids, [[("canvas", RelLoc.TOP)]])
    assert mask.numel() == rel_constraint.logits_size

    rel_constraints = [[], [(RelLoc.LEFT, 0), (RelSize.SMALLER, 0)]]
    rel_constraint.decode_state = [
        _state(2, [[10, 12, 20, 22], [6]]),
    ]
    token_ids = torch.tensor(
        [[rel_constraint.label_tokens[0], rel_constraint.label_tokens[0]]]
    )
    mask, _ = rel_constraint(token_ids, rel_constraints)
    assert mask.numel() == rel_constraint.logits_size


def test_relation_constraint_additional_branches(layout_tokenizer, small_batch) -> None:
    preprocessor = RelationshipPreprocessor(
        tokenizer=layout_tokenizer, global_task_embedding=False
    )
    rel_constraint = TransformerSortByDictRelationConstraint(preprocessor)
    cond, _ = get_condition(small_batch, "relation", layout_tokenizer)
    seq_constraints = preprocessor(cond)
    rel_constraint.prepare(seq_constraints["seq"][0])

    def _state(curr_element: int, pred_bbox: list[list[int]]):
        state = TransformerSortByDictConstraintDecodeState(curr_element)
        state.curr_element = curr_element
        state.pred_bbox = pred_bbox
        return state

    def _run(tokens: list[int], curr_element: int, pred_bbox: list[list[int]], rels):
        rel_constraint.decode_state = [_state(curr_element, pred_bbox)]
        mask, _ = rel_constraint(torch.tensor([tokens]), rels)
        assert mask.numel() == rel_constraint.logits_size

    # Canvas CENTER branch for first element (curr_element==1, current_elem="Cy")
    _run(
        [
            rel_constraint.label_tokens[0],
            rel_constraint.width_start_idx,
            rel_constraint.height_start_idx,
            rel_constraint.center_x_start_idx,
            rel_constraint.center_y_start_idx,
        ],
        1,
        [[10, 12, 20, 22]],
        [[("canvas", RelLoc.CENTER)]],
    )

    # Width branch with various relation types
    width_tokens = [rel_constraint.label_tokens[0], rel_constraint.label_tokens[0]]
    for rel_type in [
        RelLoc.RIGHT,
        RelLoc.CENTER,
        RelSize.LARGER,
        RelSize.EQUAL,
        RelLoc.UNKNOWN,
    ]:
        _run(
            width_tokens,
            2,
            [[10, 12, 20, 22], [6]],
            [[], [(rel_type, 0)]],
        )

    # Height branch with location relations
    height_tokens = [
        rel_constraint.label_tokens[0],
        rel_constraint.width_start_idx,
        rel_constraint.height_start_idx,
    ]
    for rel_type in [RelLoc.BOTTOM, RelLoc.CENTER]:
        _run(
            height_tokens,
            2,
            [[10, 12, 20, 22], [6]],
            [[], [(rel_type, 0)]],
        )

    # Height branch with size relations (curr_width=0 to hit edge cases)
    for rel_type in [RelSize.SMALLER, RelSize.LARGER, RelSize.EQUAL]:
        _run(
            height_tokens,
            2,
            [[10, 12, 20, 22], [0]],
            [[], [(rel_type, 0)]],
        )

    # Cx branch with position relations
    cx_tokens = [
        rel_constraint.label_tokens[0],
        rel_constraint.width_start_idx,
        rel_constraint.height_start_idx,
        rel_constraint.center_x_start_idx,
    ]
    for rel_type in [RelLoc.RIGHT, RelLoc.CENTER, RelLoc.UNKNOWN]:
        _run(
            cx_tokens,
            2,
            [[10, 12, 20, 22], [6]],
            [[], [(rel_type, 0)]],
        )

    # Cy branch with position relations (non-canvas)
    cy_tokens = [
        rel_constraint.label_tokens[0],
        rel_constraint.width_start_idx,
        rel_constraint.height_start_idx,
        rel_constraint.center_x_start_idx,
        rel_constraint.center_y_start_idx,
    ]
    for rel_type in [RelLoc.BOTTOM, RelLoc.CENTER, RelLoc.UNKNOWN]:
        _run(
            cy_tokens,
            2,
            [[10, 12, 20, 22], [6]],
            [[], [(rel_type, 0)]],
        )

    # Canvas branch error path
    with pytest.raises(ValueError):
        _run(
            cy_tokens,
            1,
            [[10, 12, 20, 22]],
            [[("canvas", RelLoc.UNKNOWN)]],
        )
