from __future__ import annotations

import argparse
from pathlib import Path

from huggingface_hub import snapshot_download


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download SkinVL-MM model files from Hugging Face.")
    parser.add_argument("--repo-id", default="zwq803/SkinVL-MM", help="Hugging Face repo id.")
    parser.add_argument("--local-dir", type=Path, default=Path("/root/models/SkinVL-MM"), help="Local target directory.")
    parser.add_argument("--revision", default=None, help="Optional revision, branch, or commit.")
    parser.add_argument("--token", default=None, help="Optional Hugging Face token for gated/private access.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.local_dir.mkdir(parents=True, exist_ok=True)

    resolved_path = snapshot_download(
        repo_id=args.repo_id,
        local_dir=str(args.local_dir),
        local_dir_use_symlinks=False,
        revision=args.revision,
        token=args.token,
        resume_download=True,
    )
    print(resolved_path)


if __name__ == "__main__":
    main()
