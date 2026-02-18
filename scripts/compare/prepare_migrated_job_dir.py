#!/usr/bin/env python3
from __future__ import annotations

import argparse

from ralf.compare.job_dir import prepare_migrated_job_dir


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Create a migrated job dir that rewrites config.yaml to ralf targets and "
            "symlinks the remaining artifacts."
        )
    )
    parser.add_argument("--source", required=True, help="Original job_dir path")
    parser.add_argument(
        "--dest-root",
        required=True,
        help="Destination root for migrated job dirs",
    )
    args = parser.parse_args()

    dest = prepare_migrated_job_dir(args.source, args.dest_root)
    print(dest)


if __name__ == "__main__":
    main()
