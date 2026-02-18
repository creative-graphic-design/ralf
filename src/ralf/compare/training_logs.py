from __future__ import annotations

from pathlib import Path

from .job_dir import prepare_migrated_job_dir


def prepare_migrated_training_logs(
    source_root: str | Path,
    dest_root: str | Path,
) -> Path:
    """Mirror training logs into a writable location with ralf targets."""
    src_root = Path(source_root).expanduser().resolve()
    if not src_root.is_dir():
        raise FileNotFoundError(f"training_logs not found: {src_root}")

    dest_root = Path(dest_root).expanduser().resolve()
    dest_root.mkdir(parents=True, exist_ok=True)

    for job_dir in sorted(p for p in src_root.iterdir() if p.is_dir()):
        prepare_migrated_job_dir(job_dir, dest_root, force_mirror=True)

    return dest_root
