#!/usr/bin/env python3
"""Convert an AMASS manifest through HumanPose24/GMR into Bumi tracker NPZs."""

from __future__ import annotations

import argparse
from pathlib import Path

from mimic_lite_conversion.amass_gmr import (
    AmassToBumiConverter,
    load_manifest,
    run_batch,
    run_parallel_converter_batch,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--smplx-model-dir", type=Path, required=True)
    parser.add_argument("--gmr-root", type=Path, required=True)
    parser.add_argument("--bumi-mjcf", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--target-fps", type=float, default=50.0)
    parser.add_argument(
        "--actual-human-height",
        type=float,
        default=1.6,
        help="Must match the sim2real Pico publisher setting (default: 1.6 m).",
    )
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument(
        "--torch-threads-per-worker",
        type=int,
        default=4,
        help="CPU threads used by SMPL-X inside each conversion worker.",
    )
    parser.add_argument("--limit", type=int)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--fail-on-reject", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if args.workers <= 0:
        raise ValueError(f"--workers must be positive, got {args.workers}")
    if args.torch_threads_per_worker <= 0:
        raise ValueError(
            "--torch-threads-per-worker must be positive, got "
            f"{args.torch_threads_per_worker}"
        )
    entries = load_manifest(args.manifest)
    if args.limit is not None:
        if args.limit <= 0:
            raise ValueError(f"--limit must be positive, got {args.limit}")
        entries = entries[: args.limit]
    converter_kwargs = {
        "gmr_root": args.gmr_root,
        "smplx_model_dir": args.smplx_model_dir,
        "bumi_mjcf": args.bumi_mjcf,
        "output_root": args.output,
        "target_fps": args.target_fps,
        "actual_human_height": args.actual_human_height,
        "resume": args.resume,
    }
    if args.workers == 1:
        converter = AmassToBumiConverter(**converter_kwargs)
        summary = run_batch(
            entries,
            output_root=args.output,
            converter=converter.convert,
        )
    else:
        summary = run_parallel_converter_batch(
            entries,
            output_root=args.output,
            converter_kwargs=converter_kwargs,
            workers=args.workers,
            torch_threads_per_worker=args.torch_threads_per_worker,
        )
    print(
        "AMASS -> Bumi conversion finished:",
        " ".join(f"{key}={value}" for key, value in summary.items()),
    )
    if args.fail_on_reject and summary["rejected"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
