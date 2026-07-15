from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from agent.auditability_scoring import extract_compare_runs_from_tar


def main() -> int:
    parser = argparse.ArgumentParser(description="Extract comparable compare-agent-vs-baseline run shards from a large DermAgent tar.gz export.")
    parser.add_argument("--tar-path", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--root-segment", default="final-data")
    parser.add_argument("--dataset", default="pad")
    parser.add_argument("--experiment", default="hulumed_agent")
    parser.add_argument("--model", default="hulumed")
    parser.add_argument("--dataset-variant", default="pad20")
    args = parser.parse_args()

    manifest = extract_compare_runs_from_tar(
        tar_path=args.tar_path.resolve(),
        output_root=args.output_root.resolve(),
        root_segment=str(args.root_segment).strip(),
        dataset=str(args.dataset).strip(),
        experiment=str(args.experiment).strip(),
        model=str(args.model).strip(),
        dataset_variant=str(args.dataset_variant).strip(),
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
