#!/usr/bin/env python3
"""Inventory native-timeline AMASS SMPL-X files and create subject-level splits."""

from __future__ import annotations

import argparse
from pathlib import Path

from mimic_lite_conversion.amass_gmr import (
    inventory_amass,
    inventory_summary,
    select_diverse_subset,
    split_by_subject,
    write_json,
    write_jsonl,
    write_manifest,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--amass-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--pattern", default="**/*_stageii.npz")
    parser.add_argument("--validation-fraction", type=float, default=0.1)
    parser.add_argument("--seed", default="bumi-amass-v1")
    parser.add_argument("--fixture-count", type=int, default=3)
    parser.add_argument("--pilot-count", type=int, default=30)
    parser.add_argument("--limit", type=int)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    output = args.output.expanduser().resolve()
    source_rejects: list[dict[str, str]] = []

    def record_reject(path: Path, exc: Exception) -> None:
        source_rejects.append(
            {
                "source_path": str(path.resolve()),
                "error_type": type(exc).__name__,
                "error": str(exc),
            }
        )

    entries = inventory_amass(
        args.amass_root,
        pattern=args.pattern,
        limit=args.limit,
        on_reject=record_reject,
    )
    train, validation = split_by_subject(
        entries,
        validation_fraction=args.validation_fraction,
        seed=args.seed,
    )
    fixture = select_diverse_subset(
        train,
        args.fixture_count,
        seed=f"{args.seed}:fixture",
    )
    pilot = select_diverse_subset(
        train,
        min(args.pilot_count, len(train)),
        seed=f"{args.seed}:pilot",
    )

    write_manifest(output / "all.jsonl", entries)
    write_manifest(output / "train.jsonl", train)
    write_manifest(output / "val.jsonl", validation)
    write_manifest(output / "fixture_3.jsonl", fixture)
    write_manifest(output / "pilot_20_50.jsonl", pilot)
    write_jsonl(output / "source_rejected.jsonl", source_rejects)
    train_groups = {f"{entry.dataset}/{entry.subject}" for entry in train}
    val_groups = {f"{entry.dataset}/{entry.subject}" for entry in validation}
    if train_groups & val_groups:
        raise RuntimeError("Subject-level train/validation leakage detected")
    summary = {
        "all": inventory_summary(entries),
        "train": inventory_summary(train),
        "validation": inventory_summary(validation),
        "fixture": inventory_summary(fixture),
        "pilot": inventory_summary(pilot),
        "source_rejected": len(source_rejects),
        "validation_fraction": args.validation_fraction,
        "split_seed": args.seed,
    }
    write_json(output / "source_inventory.json", summary)
    print(
        "AMASS manifest ready:",
        f"all={len(entries)}",
        f"train={len(train)}",
        f"val={len(validation)}",
        f"fixture={len(fixture)}",
        f"pilot={len(pilot)}",
        f"rejected={len(source_rejects)}",
        f"output={output}",
    )


if __name__ == "__main__":
    main()
