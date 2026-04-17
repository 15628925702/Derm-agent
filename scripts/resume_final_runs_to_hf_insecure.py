#!/usr/bin/env python

from __future__ import annotations

import os
import sys

import httpx
from huggingface_hub import HfApi, set_client_factory


def main() -> int:
    repo_id = os.environ.get("HF_REPO_ID")
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")
    local_path = os.environ.get("LOCAL_PATH", "/root/DermAgent/final-score/final_runs")
    repo_type = os.environ.get("REPO_TYPE", "dataset")
    revision = os.environ.get("REVISION", "main")
    num_workers = int(os.environ.get("NUM_WORKERS", "2"))

    if not repo_id:
        print("HF_REPO_ID is required", file=sys.stderr)
        return 1
    if not token:
        print("HF_TOKEN or HUGGINGFACE_HUB_TOKEN is required", file=sys.stderr)
        return 1

    set_client_factory(
        lambda: httpx.Client(
            verify=False,
            follow_redirects=True,
            timeout=None,
        )
    )

    api = HfApi(token=token)
    api.upload_large_folder(
        repo_id=repo_id,
        folder_path=local_path,
        repo_type=repo_type,
        revision=revision,
        private=True,
        num_workers=num_workers,
        print_report=True,
        print_report_every=60,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
