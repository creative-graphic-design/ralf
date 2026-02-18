# AGENTS Memo (Pre-Compaction)

## Current Goal
- Reach **coverage >= 90%** with **no exclusions** across `src/ralf`.
- Use real dataset sampling where needed; avoid mocks/monkeypatching.
- Ensure imports are standard (`from ralf...`) with code under `src/ralf`.
- Keep project managed via `uv`, with pre-commit enabled.

## Current Status
- Full test run passes with coverage gate: `HF_DATASETS_DISABLE_PROGRESS_BARS=1 uv run pytest --cov=ralf --cov-fail-under=90` -> **249 passed**, **Total coverage: 90.03%**.
- Removed runtime `_target_` rewriting from `inference.py` and `inference_single_data.py`; added tests ensuring no rewriting.
- Added `ralf.compare.job_dir.prepare_migrated_job_dir` plus `scripts/compare/prepare_migrated_job_dir.py` and wired `scripts/compare/run_compare.sh` to use a migrated job dir for the ralf run.
- Added `ralf.compare.training_logs.prepare_migrated_training_logs` and `scripts/compare/prepare_migrated_training_logs.py` to mirror training logs into a writable path with ralf targets.
- Generated a writable mirror at `cache/training_logs_migrated` (symlinks to original artifacts; config.yaml rewritten).
  - `scripts/compare/run_compare.sh` now defaults to this migrated training_logs and auto-creates it if missing.

## Recent Code Changes (Relevant)
- Fixed boolean tensor check in `src/ralf/train/helpers/util.py` (`if mask is not None`).
- Added `convert_tokens_to_ids` to `src/ralf/train/helpers/layout_tokenizer.py`.
- `src/ralf/train/helpers/metric.py`: `compute_rshm` now uses model device instead of `.cuda()`.
- `src/ralf/train/models/autoreg.py`: pass `task_token=None` to `UserConstraintTransformerEncoder`.
- `src/ralf/train/inference_single_data.py`: guard missing attributes and decoded tokens.
- `src/ralf/train/models/layoutformerpp/relation_restriction.py`: initialize missing fields in constraint class.

## Recent/Added Tests (Highlights)
- New or expanded tests in:
  - `tests/train/fid/test_fid_train_main.py`
  - `tests/train/helpers/test_visualizer.py`
  - `tests/train/models/test_layoutformerpp_violate.py`
  - `tests/train/models/test_relation_restriction_extra.py`
  - `tests/train/models/test_cgl_generator.py`
  - `tests/train/models/test_base_model_optim.py`
  - `tests/train/models/test_retrieval_augmented_autoreg_extra.py`
  - `tests/compare/test_compare_outputs.py` (additional branches + main())
  - `tests/preprocess/test_build_retrieval_indexes.py` (main() arg parsing path)
  - Plus edits across sampling/misc/util/io/retriever/cross_retriever/maskgit/layoutdm tests.

## Known Constraints / Pitfalls
- There is a corrupted FAISS index at:
  - `/cvl-gen-vsfs2/public_dataset/layout/RALF_cache/cgl_saliency_wo_head_index.faiss`
  - It is 0 bytes; avoid writing/reading this path.
- Use temp cache dirs or `save_index=False` in tests that touch FAISS.
- Tests should use real data samples from:
  - `/cvl-gen-vsfs2/public_dataset/layout/RALF_cache/dataset/cgl`

## Coverage Hotspots (Still Low)
These modules are still well below 90% and should be targeted with lightweight tests:
- `src/ralf/hfds_builder/dump_dataset.py` (~42%)
- `src/ralf/hfds_builder/inpainting.py` (~49%)
- `src/ralf/preprocess/build_retrieval_indexes.py` (~57%)
- `src/ralf/train/models/cgl.py` (~55%)
- `src/ralf/train/models/detr/util/misc.py` (~65%)
- `src/ralf/train/models/diffusion/common.py` (~67%)
- `src/ralf/train/models/retrieval/image.py` (~62%)
- `src/ralf/train/models/retrieval/reranker.py` (~41%)
- `src/ralf/train/models/retrieval/retriever.py` (~59%)
- `src/ralf/train/models/retrieval_augmented_autoreg.py` (~68%)
- `src/ralf/train/train.py` (~71%)
- `src/ralf/train/inference*.py` (~67-77%)

## Recommended Next Tests (Low-Cost Coverage Gains)
- `reranker.py`: test error branches (invalid score type, top_k bounds, lambda bounds) and both similarity/distance paths.
- `build_retrieval_indexes.py`: test `preprocess_retriever` with `save_index=False`, `max_items_per_split` small, temp `cache_dir`.
- `cgl.py`: cover `CGLDiscriminator` and `RetrievalAugmentedCGLGenerator` preprocess + `_encode_into_memory` (small batch, `top_k=1`).
- `detr/util/misc.py`: cover `NestedTensor.decompose` and `save_on_master`.
- `retrieval/image.py`: exercise `FeatureExtracterBackbone` with saliency path; avoid heavy backbone loads.
- `retrieval/retriever.py`: cover `preprocess_to_merge_retrieval_cache` with minimal FAISS indices in temp dir.
- `hfds_builder/inpainting.py` + `dump_dataset.py`: unit tests for `_get_mask` and `_make_record` branches.
- `train/inference.py` + `inference_unanno.py`: test branches like `preload_data=True`, `use_db_dataset=True`, `save_vis=True`, `repeat_retrieved_layouts`, `dynamic_topk` using debug mode and minimal dataset.

## Notes
- `pytest` config enforces `--cov=ralf --cov-fail-under=90`; until coverage is raised, the full run will fail.
- Pre-commit should remain installed and enabled.
- Writes to `/cvl-gen-vsfs2/public_dataset/layout/RALF_cache/training_logs/*/config.yaml` fail with `OSError: [Errno 122] Disk quota exceeded`, so configs cannot be edited in place from this environment.
- Use `cache/training_logs_migrated` as the writable training_logs root for future runs (or re-run the prepare script if source changes).

## Comparison Plan (2026-02-16)
- Compare original (main) vs migrated outputs under identical conditions using:
  - `scripts/compare/run_compare.sh` to run inference, inference_single_data, and eval
  - `scripts/compare/compare_original_vs_migrated.py` to compare outputs
- Default data source: `/cvl-gen-vsfs2/public_dataset/layout/RALF_cache`
- Default job dir: `$RALF_CACHE_DIR/training_logs/ralf_uncond_cgl`
- Default dataset: `cgl`, cond_type inferred from `config.yaml` (fallback `uncond`)
- Output roots: `tmp/compare/original` and `tmp/compare/migrated`
- Numeric tolerance for comparisons: `1e-4`
- Comparison report: `tmp/compare/comparison_report.json`
