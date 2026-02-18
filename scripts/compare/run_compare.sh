#!/usr/bin/env bash
if [ -z "${BASH_VERSION:-}" ]; then
  exec /usr/bin/env bash "$0" "$@"
fi
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
ORIGINAL_REF="main"
DATASET="cgl"
RALF_CACHE_DIR="${RALF_CACHE_DIR:-/cvl-gen-vsfs2/public_dataset/layout/RALF_cache}"
RALF_MIGRATED_TRAINING_LOGS="${RALF_MIGRATED_TRAINING_LOGS:-$ROOT_DIR/cache/training_logs_migrated}"
JOB_DIR=""
MIGRATED_JOB_DIR=""
OUTPUT_ROOT="${OUTPUT_ROOT:-$ROOT_DIR/tmp/compare}"
CUDA_DEVICE="${CUDA_DEVICE:-0}"
TOLERANCE="${TOLERANCE:-1e-4}"
COND_TYPE="${COND_TYPE:-}"
SAMPLE_ID="${SAMPLE_ID:-0}"
DEBUG_MODE="${DEBUG_MODE:-0}"
DEBUG_NUM_SAMPLES="${DEBUG_NUM_SAMPLES:-64}"
PRELOAD_DATA="${PRELOAD_DATA:-False}"
SKIP_SYNC=0

usage() {
  cat <<EOF
Usage: $0 [options]
  --original-ref <ref>   Git ref for original implementation (default: main)
  --dataset <name>       Dataset name (default: cgl)
  --cache-dir <path>     Cache root (default: /cvl-gen-vsfs2/public_dataset/layout/RALF_cache)
  --job-dir <path>       Training log directory (default: <migrated-training-logs>/ralf_uncond_cgl)
  --output-root <path>   Output root for compare artifacts (default: ./tmp/compare)
  --cuda <id>            CUDA device id (default: 0)
  --tolerance <float>    Numeric tolerance (default: 1e-4)
  --cond-type <name>     Override cond_type (default: inferred from config.yaml)
  --sample-id <id>       Sample id for inference_single_data (default: 0)
  --debug                Enable debug mode (fewer samples)
  --debug-num-samples <n>  Limit number of samples when debug is enabled (default: 64)
  --preload-data <bool>  Override preload_data (default: False)
  --skip-sync            Skip uv sync in worktrees
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --original-ref)
      ORIGINAL_REF="$2"
      shift 2
      ;;
    --dataset)
      DATASET="$2"
      shift 2
      ;;
    --cache-dir)
      RALF_CACHE_DIR="$2"
      shift 2
      ;;
    --job-dir)
      JOB_DIR="$2"
      shift 2
      ;;
    --output-root)
      OUTPUT_ROOT="$2"
      shift 2
      ;;
    --cuda)
      CUDA_DEVICE="$2"
      shift 2
      ;;
    --tolerance)
      TOLERANCE="$2"
      shift 2
      ;;
    --cond-type)
      COND_TYPE="$2"
      shift 2
      ;;
    --sample-id)
      SAMPLE_ID="$2"
      shift 2
      ;;
    --debug)
      DEBUG_MODE=1
      shift
      ;;
    --debug-num-samples)
      DEBUG_NUM_SAMPLES="$2"
      shift 2
      ;;
    --preload-data)
      PRELOAD_DATA="$2"
      shift 2
      ;;
    --skip-sync)
      SKIP_SYNC=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown option: $1"
      usage
      exit 1
      ;;
  esac
done

if [[ -z "${JOB_DIR}" ]]; then
  JOB_DIR="${RALF_MIGRATED_TRAINING_LOGS}/ralf_uncond_cgl"
fi

if [[ "${JOB_DIR}" == "${RALF_MIGRATED_TRAINING_LOGS}"* ]]; then
  if [[ ! -d "${RALF_MIGRATED_TRAINING_LOGS}" ]]; then
    uv run --project "${ROOT_DIR}" --directory "${ROOT_DIR}" \
      python "${ROOT_DIR}/scripts/compare/prepare_migrated_training_logs.py" \
      --source "${RALF_CACHE_DIR}/training_logs" \
      --dest-root "${RALF_MIGRATED_TRAINING_LOGS}"
  fi
fi

JOB_DIR="$(realpath "${JOB_DIR}")"
RALF_CACHE_DIR="$(realpath "${RALF_CACHE_DIR}")"
if [[ -d "${RALF_MIGRATED_TRAINING_LOGS}" ]]; then
  RALF_MIGRATED_TRAINING_LOGS="$(realpath "${RALF_MIGRATED_TRAINING_LOGS}")"
fi
RALF_DATASET_DIR="${RALF_CACHE_DIR}/dataset"
RALF_PRECOMPUTED_WEIGHT_DIR="${RALF_CACHE_DIR}/PRECOMPUTED_WEIGHT_DIR"

if [[ ! -d "${JOB_DIR}" ]]; then
  echo "JOB_DIR not found: ${JOB_DIR}" >&2
  exit 1
fi

if [[ -z "${COND_TYPE}" ]]; then
  COND_TYPE="$(uv run python - <<'PY'
import sys
import yaml
from pathlib import Path

config_path = Path("${JOB_DIR}") / "config.yaml"
if not config_path.exists():
    print("uncond")
    sys.exit(0)
data = yaml.safe_load(config_path.read_text())
generator = data.get("generator", {}) if isinstance(data, dict) else {}
aux_task = generator.get("auxilary_task") or generator.get("auxiliary_task")
print(aux_task or "uncond")
PY
)"
fi

export RALF_CACHE_DIR
export RALF_DATASET_DIR
export RALF_PRECOMPUTED_WEIGHT_DIR
export CUDA_VISIBLE_DEVICES="${CUDA_DEVICE}"
export RALF_NUM_WORKERS="${RALF_NUM_WORKERS:-0}"
export UV_PYTHON="${UV_PYTHON:-python3.10}"
export PYTHONHASHSEED=0
export CUBLAS_WORKSPACE_CONFIG=":4096:8"
export CUDNN_DETERMINISTIC=1
export CUDNN_BENCHMARK=0
export HF_DATASETS_DISABLE_PROGRESS_BARS=1

WORKTREE_BASE="${WORKTREE_BASE:-/tmp/ralf_worktrees}"
ORIGINAL_WT="${WORKTREE_BASE}/ralf_original"

mkdir -p "${WORKTREE_BASE}"

if [[ -e "${ORIGINAL_WT}" ]]; then
  git -C "${ROOT_DIR}" worktree remove --force "${ORIGINAL_WT}"
fi

git -C "${ROOT_DIR}" worktree add "${ORIGINAL_WT}" "${ORIGINAL_REF}"

patch_original_num_workers() {
  local target="${ORIGINAL_WT}/image2layout/train/inference.py"
  if [[ -f "${target}" ]] && ! grep -q "RALF_NUM_WORKERS" "${target}"; then
    python3 - <<'PY'
from pathlib import Path

target = Path("/tmp/ralf_worktrees/ralf_original/image2layout/train/inference.py")
text = target.read_text()
old = """        loaders[split] = torch.utils.data.DataLoader(
            dataset[split],
            num_workers=2 if not test_cfg.debug else 0,
            batch_size=test_cfg.batch_size,
            pin_memory=False,
            collate_fn=collate_fn_partial,
            persistent_workers=False,
            drop_last=False,
        )
"""
new = """        num_workers = 2 if not test_cfg.debug else 0
        env_workers = os.environ.get("RALF_NUM_WORKERS")
        if env_workers is not None:
            try:
                num_workers = int(env_workers)
            except ValueError:
                logger.warning(
                    "Invalid RALF_NUM_WORKERS=%s; using %d",
                    env_workers,
                    num_workers,
                )

        loaders[split] = torch.utils.data.DataLoader(
            dataset[split],
            num_workers=num_workers,
            batch_size=test_cfg.batch_size,
            pin_memory=False,
            collate_fn=collate_fn_partial,
            persistent_workers=False,
            drop_last=False,
        )
"""
if old not in text:
    raise SystemExit("Could not patch num_workers block in original inference.py")
target.write_text(text.replace(old, new))
PY
  fi
}

patch_original_num_workers

rm -rf "${OUTPUT_ROOT}"
mkdir -p "${OUTPUT_ROOT}/original" "${OUTPUT_ROOT}/migrated"
MIGRATED_JOB_DIR="$(uv run --project "${ROOT_DIR}" --directory "${ROOT_DIR}" \
  python "${ROOT_DIR}/scripts/compare/prepare_migrated_job_dir.py" \
  --source "${JOB_DIR}" \
  --dest-root "${OUTPUT_ROOT}/migrated_job_dirs")"

run_uv() {
  local tree_root="$1"
  shift
  local pythonpath="${tree_root}"
  if [[ -n "${PYTHONPATH:-}" ]]; then
    pythonpath="${pythonpath}:${PYTHONPATH}"
  fi
  PYTHONPATH="${pythonpath}" uv run --project "${ROOT_DIR}" --directory "${tree_root}" "$@"
}

run_eval() {
  local tree_root="$1"
  local result_root="$2"
  local dataset_path="${RALF_DATASET_DIR}/${DATASET}"
  local fid_dir="${RALF_PRECOMPUTED_WEIGHT_DIR}/fidnet/${DATASET}"

  for input_dir in "${result_root}/inference"/generated_samples_*; do
    if [[ -d "${input_dir}" ]]; then
      run_uv "${tree_root}" python "${tree_root}/eval.py" \
        --input-dir "${input_dir}" \
        --fid-weight-dir "${fid_dir}" \
        --save-score-dir "${result_root}/eval_scores/${DATASET}" \
        --dataset-path "${dataset_path}"
    fi
  done
}

run_inference() {
  local tree_root="$1"
  local result_root="$2"
  local dataset_path="${RALF_DATASET_DIR}/${DATASET}"
  local module_prefix="$3"
  local job_dir="$4"

  run_uv "${tree_root}" python -m "${module_prefix}.train.inference" \
    job_dir="${job_dir}" \
    result_dir="${result_root}/inference" \
    dataset_path="${dataset_path}" \
    cond_type="${COND_TYPE}" \
    +sampling="top_k" \
    preload_data="${PRELOAD_DATA}" \
    debug=$([[ "${DEBUG_MODE}" -eq 1 ]] && echo True || echo False) \
    debug_num_samples=$([[ "${DEBUG_MODE}" -eq 1 ]] && echo "${DEBUG_NUM_SAMPLES}" || echo -1) \
    hydra/hydra_logging=none \
    hydra/job_logging=none
}

run_inference_single() {
  local tree_root="$1"
  local result_root="$2"
  local dataset_path="${RALF_DATASET_DIR}/${DATASET}"
  local module_prefix="$3"
  local job_dir="$4"

  run_uv "${tree_root}" python -m "${module_prefix}.train.inference_single_data" \
    job_dir="${job_dir}" \
    result_dir="${result_root}/inference_single" \
    dataset_path="${dataset_path}" \
    cond_type="${COND_TYPE}" \
    sample_id="${SAMPLE_ID}" \
    +sampling="top_k" \
    debug=$([[ "${DEBUG_MODE}" -eq 1 ]] && echo True || echo False) \
    hydra/hydra_logging=none \
    hydra/job_logging=none
}

run_pipeline() {
  local tree_root="$1"
  local label="$2"
  local module_prefix=""
  local job_dir="${JOB_DIR}"

  if [[ -d "${tree_root}/src/ralf" ]]; then
    module_prefix="ralf"
  elif [[ -d "${tree_root}/image2layout" ]]; then
    module_prefix="image2layout"
  elif [[ -d "${tree_root}/src/image2layout" ]]; then
    module_prefix="image2layout"
  else
    echo "Could not determine module prefix under ${tree_root}" >&2
    exit 1
  fi

  if [[ "${module_prefix}" == "image2layout" ]]; then
    if [[ ! -e "${tree_root}/cache" ]]; then
      ln -s "${RALF_CACHE_DIR}" "${tree_root}/cache"
    fi
  fi

  if [[ "${SKIP_SYNC}" -eq 0 ]]; then
    if [[ "${tree_root}" == "${ROOT_DIR}" ]]; then
      (
        cd "${tree_root}"
        if ! uv sync --extra original-impl; then
          uv sync
        fi
      )
    fi
  fi

  if [[ "${module_prefix}" == "ralf" ]]; then
    job_dir="${MIGRATED_JOB_DIR}"
  fi

  run_inference "${tree_root}" "${OUTPUT_ROOT}/${label}" "${module_prefix}" "${job_dir}"
  run_inference_single "${tree_root}" "${OUTPUT_ROOT}/${label}" "${module_prefix}" "${job_dir}"
  run_eval "${tree_root}" "${OUTPUT_ROOT}/${label}"
}

run_pipeline "${ORIGINAL_WT}" "original"
run_pipeline "${ROOT_DIR}" "migrated"

uv run python "${ROOT_DIR}/scripts/compare/compare_original_vs_migrated.py" \
  --original "${OUTPUT_ROOT}/original" \
  --migrated "${OUTPUT_ROOT}/migrated" \
  --tolerance "${TOLERANCE}" \
  --ignore-ext .txt \
  --report-path "${OUTPUT_ROOT}/comparison_report.json"
