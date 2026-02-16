import torch

from ralf.train.helpers.relationships import RelElement, RelLoc
from ralf.train.models.layoutformerpp.relation_restriction import (
    DiscretizeBoundingBox,
    TransformerSortByDictConstraintDecodeState,
    TransformerSortByDictLabelSizeConstraint,
    TransformerSortByDictRelationConstraint,
    decapulate,
    define_canvas_restriction,
    detect_label_idx,
)
from ralf.train.models.layoutformerpp.task_preprocessor import RelationshipPreprocessor


def test_relation_restriction_extras(layout_tokenizer) -> None:
    bbox = torch.tensor([[0.1, 0.2, 0.3, 0.4]])
    x1, y1, x2, y2 = decapulate(bbox)
    assert x1.numel() == 1

    bbox_3d = bbox.unsqueeze(0)
    x1b, _, _, _ = decapulate(bbox_3d)
    assert x1b.shape[0] == 1

    discretizer = DiscretizeBoundingBox(num_x_grid=4, num_y_grid=4)
    discrete = discretizer.discretize(bbox)
    cont = discretizer.continuize(discrete)
    assert cont.shape == bbox.shape
    assert discretizer.continuize_num(2) > 0
    assert discretizer.discretize_num(0.5) >= 0

    preprocessor = RelationshipPreprocessor(
        tokenizer=layout_tokenizer, global_task_embedding=False
    )
    types = torch.tensor([preprocessor.name_to_id(RelElement.A)])
    label = types[0]
    index = torch.tensor(preprocessor.name_to_id(RelElement.A))
    label_pos = detect_label_idx(types, label, index, {RelElement.A: 0}, preprocessor)
    assert label_pos == 0

    label_size_constraint = TransformerSortByDictLabelSizeConstraint(preprocessor)
    label_size_constraint.prepare([[1]], [[[0, 0, 1, 1]]])
    _ = label_size_constraint(0, 0, torch.tensor([0]))

    rel_constraint = TransformerSortByDictRelationConstraint(preprocessor)
    state = TransformerSortByDictConstraintDecodeState(num_elements=1)
    state.add_label(1)
    state.add_bbox_num(1)
    state.pred_bbox = [[1, 1, 2, 2]]
    rel_type, bbox_out, back_idx = rel_constraint.get_target_bbox(RelLoc.LEFT, 0, state)
    assert bbox_out is not None
    rel_type2, bbox_out2, back_idx2 = rel_constraint.get_target_bbox(
        "canvas", RelLoc.TOP, state
    )
    assert bbox_out2 is None

    predictable = torch.zeros(200, dtype=torch.bool)
    predictable[131] = True
    mask_top = define_canvas_restriction(predictable, RelLoc.TOP, total_bin=9)
    mask_center = define_canvas_restriction(predictable, RelLoc.CENTER, total_bin=9)
    mask_bottom = define_canvas_restriction(predictable, RelLoc.BOTTOM, total_bin=9)
    assert mask_top.any() and mask_center.any() and mask_bottom.any()
