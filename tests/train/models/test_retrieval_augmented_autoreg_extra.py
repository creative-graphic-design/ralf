import pytest
import torch
from omegaconf import OmegaConf

from ralf.train.helpers.task import get_condition
from ralf.train.models.retrieval_augmented_autoreg import (
    ConcateAuxilaryTaskAfterConcateTransEncRetrievalAugmentedAutoreg,
    ConcateAuxilaryTaskConcateCrossAttnRetrievalAugmentedAutoreg,
    ConcateAuxilaryTaskConcateRetrievalAugmentedAutoreg,
    ConcateAuxilaryTaskConcateTransEncRetrievalAugmentedAutoreg,
    ConcateAuxilaryTaskCrossAttnRetrievalAugmentedAutoreg,
    ConcateAuxilaryTaskFlagConcateCrossAttnRetrievalAugmentedAutoreg,
    RetrievalAugmentedAutoregAdapter,
    extract_retrieved_features,
    get_ref_layout_input,
)


def _build_retrieved(small_batch):
    image = torch.cat([small_batch["image"], small_batch["saliency"]], dim=1)
    return {
        "image": image.unsqueeze(1),
        "saliency": small_batch["saliency"].unsqueeze(1),
        "center_x": small_batch["center_x"].unsqueeze(1),
        "center_y": small_batch["center_y"].unsqueeze(1),
        "width": small_batch["width"].unsqueeze(1),
        "height": small_batch["height"].unsqueeze(1),
        "label": small_batch["label"].unsqueeze(1),
        "mask": small_batch["mask"].unsqueeze(1),
    }


def test_retrieval_augmented_aux_paths(
    small_batch, features, layout_tokenizer, dataset
) -> None:
    ds_dict, _ = dataset
    retrieved = _build_retrieved(small_batch)
    inputs = {**small_batch, "retrieved": retrieved}

    model = ConcateAuxilaryTaskConcateCrossAttnRetrievalAugmentedAutoreg(
        features=features,
        tokenizer=layout_tokenizer,
        dataset_name="cgl",
        max_seq_length=10,
        db_dataset=ds_dict["train"],
        top_k=1,
        retrieval_backbone="saliency",
        random_retrieval=False,
        saliency_k=1,
        auxilary_task="c",
        use_multitask=False,
        use_flag_embedding=True,
        global_task_embedding=True,
    )
    model.eval()

    model_inputs, targets = model.preprocess(inputs)
    encoded = model._encode_into_memory(model_inputs)
    assert "memory" in encoded
    assert targets["seq"].shape[0] == small_batch["image"].shape[0]

    cond, _ = get_condition(inputs, "c", layout_tokenizer)
    sampling_cfg = OmegaConf.create({"name": "random", "temperature": 1.0})
    output = model.sample(
        cond=cond,
        sampling_cfg=sampling_cfg,
        cond_type="c",
        use_backtrack=False,
    )
    assert "label" in output


def test_extract_retrieved_features_branches(
    small_batch, features, layout_tokenizer, dataset
) -> None:
    ds_dict, _ = dataset
    retrieved = _build_retrieved(small_batch)
    model = RetrievalAugmentedAutoregAdapter(
        features=features,
        tokenizer=layout_tokenizer,
        dataset_name="cgl",
        max_seq_length=10,
        db_dataset=ds_dict["train"],
        top_k=1,
        retrieval_backbone="saliency",
        random_retrieval=False,
        saliency_k=1,
        use_reference_image=False,
    )
    ref_layouts = extract_retrieved_features(
        retrieved_samples=retrieved,
        top_k=1,
        image_encoder=[model.encoder, model.pos_emb_2d, model.transformer_encoder],
        layout_encoder=model.layout_encoer,
        layout_adapter=model.layout_adapter,
        pos_emb_1d=model.pos_emb_1d,
        use_reference_image=False,
    )
    retrieved["hybrid_dynamic_indexes"] = torch.zeros_like(ref_layouts)
    ref_layouts2 = extract_retrieved_features(
        retrieved_samples=retrieved,
        top_k=1,
        image_encoder=[model.encoder, model.pos_emb_2d, model.transformer_encoder],
        layout_encoder=model.layout_encoer,
        layout_adapter=model.layout_adapter,
        pos_emb_1d=model.pos_emb_1d,
        use_reference_image=False,
    )
    assert ref_layouts2.shape == ref_layouts.shape

    ref_layouts3 = extract_retrieved_features(
        retrieved_samples=retrieved,
        top_k=1,
        image_encoder=[model.encoder, model.pos_emb_2d, model.transformer_encoder],
        layout_encoder=model.layout_encoer,
        layout_adapter=model.layout_adapter,
        pos_emb_1d=model.pos_emb_1d,
        use_reference_image=True,
    )
    assert ref_layouts3.shape[0] == small_batch["image"].shape[0]


def test_retrieval_augmented_sample_relation(
    small_batch, features, layout_tokenizer, dataset
) -> None:
    ds_dict, _ = dataset
    retrieved = _build_retrieved(small_batch)
    inputs = {**small_batch, "retrieved": retrieved}

    model = ConcateAuxilaryTaskConcateCrossAttnRetrievalAugmentedAutoreg(
        features=features,
        tokenizer=layout_tokenizer,
        dataset_name="cgl",
        max_seq_length=10,
        db_dataset=ds_dict["train"],
        top_k=1,
        retrieval_backbone="saliency",
        random_retrieval=False,
        saliency_k=1,
        auxilary_task="relation",
        use_multitask=False,
        use_flag_embedding=True,
        global_task_embedding=False,
    )
    model.eval()

    cond, _ = get_condition(inputs, "relation", layout_tokenizer)
    sampling_cfg = OmegaConf.create({"name": "deterministic", "temperature": 1.0})
    output, violation = model.sample_relation(
        cond=cond,
        sampling_cfg=sampling_cfg,
        return_violation=True,
        prob_gate=-1.0,
        RELATION_SIZE=1,
        return_decoded_cond=True,
    )
    assert "label" in output
    assert "decoded_tokens" in output
    assert violation["total"] >= 0


def test_get_ref_layout_input(small_batch) -> None:
    retrieved = _build_retrieved(small_batch)
    out = get_ref_layout_input(retrieved, kdx=0)
    assert "label" in out
    assert out["label"].ndim == 2


@pytest.mark.parametrize(
    "cls",
    [
        ConcateAuxilaryTaskCrossAttnRetrievalAugmentedAutoreg,
        ConcateAuxilaryTaskConcateRetrievalAugmentedAutoreg,
        ConcateAuxilaryTaskConcateCrossAttnRetrievalAugmentedAutoreg,
        ConcateAuxilaryTaskFlagConcateCrossAttnRetrievalAugmentedAutoreg,
        ConcateAuxilaryTaskConcateTransEncRetrievalAugmentedAutoreg,
        ConcateAuxilaryTaskAfterConcateTransEncRetrievalAugmentedAutoreg,
    ],
)
def test_retrieval_augmented_variants_encode(
    cls, small_batch, features, layout_tokenizer, dataset
) -> None:
    ds_dict, _ = dataset
    retrieved = _build_retrieved(small_batch)
    inputs = {**small_batch, "retrieved": retrieved}

    model = cls(
        features=features,
        tokenizer=layout_tokenizer,
        dataset_name="cgl",
        max_seq_length=10,
        db_dataset=ds_dict["train"],
        top_k=1,
        retrieval_backbone="saliency",
        random_retrieval=False,
        saliency_k=1,
        auxilary_task="c",
        use_multitask=False,
        use_flag_embedding=True,
        global_task_embedding=True,
    )
    model.eval()

    model_inputs, _ = model.preprocess(inputs)
    encoded = model._encode_into_memory(model_inputs)
    assert "memory" in encoded
