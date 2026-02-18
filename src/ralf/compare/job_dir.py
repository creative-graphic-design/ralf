from __future__ import annotations

import os
import shutil
from pathlib import Path


def prepare_migrated_job_dir(
    source: str | Path,
    dest_root: str | Path,
    *,
    force_mirror: bool = False,
) -> Path:
    """Create a migrated job dir with ralf targets and symlinked artifacts."""
    src = Path(source).expanduser().resolve()
    if not src.is_dir():
        raise FileNotFoundError(f"job_dir not found: {src}")

    config_path = src / "config.yaml"
    if not config_path.exists():
        raise FileNotFoundError(f"config.yaml not found in {src}")

    config_text = config_path.read_text()
    if "image2layout" not in config_text and not force_mirror:
        return src

    dest_root = Path(dest_root).expanduser().resolve()
    dest = dest_root / src.name
    if dest.exists():
        shutil.rmtree(dest)
    dest.mkdir(parents=True)

    for entry in src.iterdir():
        dest_entry = dest / entry.name
        if entry.name == "config.yaml":
            dest_entry.write_text(config_text.replace("image2layout", "ralf"))
        else:
            os.symlink(entry, dest_entry)

    return dest
