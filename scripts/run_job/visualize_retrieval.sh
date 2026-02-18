GPU_ID="${1:-"0"}"
export CUDA_VISIBLE_DEVICES=$GPU_ID

OMP_NUM_THREADS=2 uv run python3 -m ralf.train.visualize_retrieval
