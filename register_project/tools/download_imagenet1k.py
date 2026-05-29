#!/usr/bin/env python3
"""Download ImageNet-1K from the Hugging Face Hub.

Default target:
    ILSVRC/imagenet-1k

The dataset is gated on Hugging Face. Before running the full download, accept
the dataset terms at https://huggingface.co/datasets/ILSVRC/imagenet-1k and
provide a token with access through HF_TOKEN, --token, hf auth login, or
hf_token.txt.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Iterable


DEFAULT_REPO_ID = "ILSVRC/imagenet-1k"
DEFAULT_TOKEN_FILE = Path("hf_token.txt")


def parse_patterns(values: list[str] | None) -> list[str] | None:
    if not values:
        return None

    patterns: list[str] = []
    for value in values:
        patterns.extend(part.strip() for part in value.split(",") if part.strip())
    return patterns or None


def human_size(num_bytes: int | None) -> str:
    if num_bytes is None:
        return "unknown size"

    size = float(num_bytes)
    for unit in ("B", "KB", "MB", "GB", "TB", "PB"):
        if size < 1024 or unit == "PB":
            return f"{size:.1f} {unit}"
        size /= 1024
    return f"{size:.1f} PB"


def summarize_dry_run(items: Iterable[object]) -> tuple[int, int]:
    count = 0
    total_size = 0
    for item in items:
        count += 1
        size = getattr(item, "file_size", getattr(item, "size", None))
        if isinstance(size, int):
            total_size += size
    return count, total_size


def import_snapshot_download():
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        python = sys.executable or "python3"
        print(
            "Missing dependency: huggingface_hub\n"
            "Install it with:\n"
            f"  {python} -m pip install -U huggingface_hub",
            file=sys.stderr,
        )
        raise SystemExit(1)
    return snapshot_download


def read_token_file(path: Path) -> str | None:
    if not path.exists():
        return None

    token = path.read_text(encoding="utf-8").strip()
    return token or None


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Download ImageNet-1K from Hugging Face.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("imagenet-1k"),
        help="Directory to place the downloaded dataset files.",
    )
    parser.add_argument(
        "--repo-id",
        default=DEFAULT_REPO_ID,
        help=f"Hugging Face dataset repo to download. Defaults to {DEFAULT_REPO_ID}.",
    )
    parser.add_argument(
        "--revision",
        default="main",
        help="Hub revision, branch, tag, or commit hash to download.",
    )
    parser.add_argument(
        "--allow",
        action="append",
        metavar="PATTERN[,PATTERN...]",
        help=(
            "Only download files matching these glob patterns. May be passed "
            "multiple times. Example: --allow 'data/train-*'"
        ),
    )
    parser.add_argument(
        "--ignore",
        action="append",
        metavar="PATTERN[,PATTERN...]",
        help="Skip files matching these glob patterns. May be passed multiple times.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=8,
        help="Number of parallel download workers.",
    )
    parser.add_argument(
        "--token",
        default=None,
        help=(
            "Hugging Face token. Usually omit this and run `hf auth login` "
            "or set HF_TOKEN instead."
        ),
    )
    parser.add_argument(
        "--token-file",
        type=Path,
        default=DEFAULT_TOKEN_FILE,
        help=(
            "File containing a Hugging Face token. Used only when --token and "
            "HF_TOKEN are not set. Defaults to hf_token.txt."
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="List what would be downloaded without downloading files.",
    )
    parser.add_argument(
        "--use-xet",
        action="store_true",
        help=(
            "Allow Hugging Face Xet/CAS transfers. By default this script disables "
            "Xet and uses regular HTTP downloads, which is often more reliable in "
            "restricted environments."
        ),
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if not args.use_xet:
        os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

    snapshot_download = import_snapshot_download()

    output_dir = args.output_dir.expanduser().resolve()
    token_file = args.token_file.expanduser()
    if not token_file.is_absolute():
        token_file = Path.cwd() / token_file
    token = args.token or os.environ.get("HF_TOKEN") or read_token_file(token_file)
    allow_patterns = parse_patterns(args.allow)
    ignore_patterns = parse_patterns(args.ignore)

    print(f"Repo:      {args.repo_id}")
    print(f"Revision:  {args.revision}")
    print(f"Output:    {output_dir}")
    print(f"Cache:     {output_dir / '.cache' / 'huggingface' / 'download'}")
    print(f"Workers:   {args.workers}")
    print(f"Xet:       {'enabled' if args.use_xet else 'disabled'}")
    if token:
        print("Token:     set")
    if allow_patterns:
        print(f"Allow:     {', '.join(allow_patterns)}")
    if ignore_patterns:
        print(f"Ignore:    {', '.join(ignore_patterns)}")

    kwargs = {
        "repo_id": args.repo_id,
        "repo_type": "dataset",
        "revision": args.revision,
        "local_dir": str(output_dir),
        "allow_patterns": allow_patterns,
        "ignore_patterns": ignore_patterns,
        "max_workers": args.workers,
        "token": token,
    }

    if args.dry_run:
        try:
            dry_run_items = snapshot_download(dry_run=True, **kwargs)
        except TypeError:
            print(
                "Your installed huggingface_hub does not support --dry-run. "
                f"Upgrade it with: {sys.executable or 'python3'} -m pip install -U huggingface_hub",
                file=sys.stderr,
            )
            return 2

        count, total_size = summarize_dry_run(dry_run_items)
        print(f"Dry run:   {count} files, {human_size(total_size)}")
        return 0

    path = snapshot_download(**kwargs)
    print(f"Done:      {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
