#!/usr/bin/env python3
from __future__ import annotations

import argparse

from ralf.compare.training_logs import prepare_migrated_training_logs


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Mirror training logs into a writable location and rewrite "
            "config.yaml targets to ralf."
        )
    )
    parser.add_argument("--source", required=True, help="Source training_logs path")
    parser.add_argument(
        "--dest-root",
        required=True,
        help="Destination root for migrated training_logs",
    )
    args = parser.parse_args()

    dest = prepare_migrated_training_logs(args.source, args.dest_root)
    print(dest)


if __name__ == "__main__":
    main()
