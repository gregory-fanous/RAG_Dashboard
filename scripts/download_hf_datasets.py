#!/usr/bin/env python3
"""Download third-party benchmark datasets from Hugging Face into ./data."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Any

DATASET_SPECS: dict[str, dict[str, str]] = {
    "open_ragbench": {
        "repo_id": "vectara/open_ragbench",
        "target_subdir": "open_ragbench",
        "size": "716 MB",
        "entries": "1,000 docs, 3,045 queries",
        "domain": "Scientific",
    },
    "retrievalqa": {
        "repo_id": "zihanz/RetrievalQA",
        "target_subdir": "RetrievalQA",
        "size": "16 MB",
        "entries": "1,271 questions",
        "domain": "General",
    },
    "natural_questions": {
        "repo_id": "sentence-transformers/natural-questions",
        "target_subdir": "natural-questions",
        "size": "43 MB",
        "entries": "100,231 pairs",
        "domain": "General",
    },
    "paperzilla": {
        "repo_id": "paperzilla/paperzilla-rag-retrieval-250",
        "target_subdir": "paperzilla-rag-retrieval-250",
        "size": "1.3 MB",
        "entries": "250 papers",
        "domain": "CS Research",
    },
    "finder": {
        "repo_id": "Linq-AI-Research/FinDER",
        "target_subdir": "FinDER",
        "size": "149 MB",
        "entries": "5,703 triplets",
        "domain": "Financial",
    },
    "ragcare_qa": {
        "repo_id": "ChatMED-Project/RAGCare-QA",
        "target_subdir": "RAGCare-QA",
        "size": "1.7 MB",
        "entries": "420 questions",
        "domain": "Medical",
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Download supported third-party datasets from Hugging Face into the local data directory."
        )
    )
    parser.add_argument(
        "--datasets",
        type=str,
        default="all",
        help=(
            "Comma-separated dataset keys to download (default: all). "
            "Use --list to see available keys."
        ),
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("data"),
        help="Destination root for dataset folders (default: ./data).",
    )
    parser.add_argument(
        "--revision",
        type=str,
        default=None,
        help="Optional dataset revision/tag/commit.",
    )
    parser.add_argument(
        "--hf-token",
        type=str,
        default=None,
        help="Optional Hugging Face token. Defaults to HF_TOKEN or HUGGINGFACE_HUB_TOKEN.",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List available dataset keys and exit.",
    )
    return parser.parse_args()


def selected_dataset_keys(raw: str) -> list[str]:
    if raw.strip().lower() == "all":
        return list(DATASET_SPECS.keys())

    keys = [part.strip().lower() for part in raw.split(",") if part.strip()]
    unknown = sorted(set(keys) - set(DATASET_SPECS.keys()))
    if unknown:
        valid = ", ".join(sorted(DATASET_SPECS.keys()))
        raise ValueError(f"Unknown dataset key(s): {', '.join(unknown)}. Valid: {valid}")
    return keys


def print_available() -> None:
    print("Available dataset keys:")
    for key, spec in DATASET_SPECS.items():
        print(
            f"- {key}: {spec['repo_id']} ({spec['size']}, {spec['entries']}, {spec['domain']})"
        )


def _load_hf_snapshot_download() -> Any:
    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        raise RuntimeError(
            "Missing dependency 'huggingface_hub'. Install with: python -m pip install huggingface_hub"
        ) from exc
    return snapshot_download


def download_one(
    dataset_key: str,
    data_root: Path,
    snapshot_download: Any,
    token: str | None,
    revision: str | None,
) -> None:
    spec = DATASET_SPECS[dataset_key]
    destination = data_root / spec["target_subdir"]
    destination.mkdir(parents=True, exist_ok=True)

    print(f"\n[{dataset_key}] Downloading {spec['repo_id']} -> {destination}")

    snapshot_download(
        repo_id=spec["repo_id"],
        repo_type="dataset",
        local_dir=str(destination),
        token=token,
        revision=revision,
    )

    print(f"[{dataset_key}] Complete")


def main() -> int:
    args = parse_args()

    if args.list:
        print_available()
        return 0

    try:
        keys = selected_dataset_keys(args.datasets)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    snapshot_download = _load_hf_snapshot_download()

    token = args.hf_token or os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_HUB_TOKEN")
    data_root = args.data_root.resolve()
    data_root.mkdir(parents=True, exist_ok=True)

    print(
        "This repository does not redistribute third-party datasets. "
        "Each dataset is pulled directly from its Hugging Face source."
    )

    for key in keys:
        download_one(
            dataset_key=key,
            data_root=data_root,
            snapshot_download=snapshot_download,
            token=token,
            revision=args.revision,
        )

    print("\nAll requested datasets completed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
