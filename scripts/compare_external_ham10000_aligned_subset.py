from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from agent.policy_config import load_policy, load_stable_policy
from dataio.ham10000_aligned_loader import DEFAULT_HAM10000_ROOT, load_ham10000_aligned_case_inputs_by_case_ids, load_ham10000_aligned_records_by_case_ids
from integrations.openai_client import DermOpenAIClient
from utils.aligned_subset_compare import run_aligned_subset_compare


DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "external_eval" / "ham10000_aligned_subset" / "comparison"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="External HAM10000 aligned-subset 3-class compare between direct Qwen and current agent.")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_HAM10000_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--policy-config", type=Path, default=None)
    parser.add_argument("--policy-label", type=str, default="")
    parser.add_argument(
        "--label-constraint",
        action="store_true",
        help="Inject a constrained label-space prompt for aligned-subset evaluation.",
    )
    parser.add_argument(
        "--conservative-fusion",
        action="store_true",
        help="Preserve direct baseline when external agent evidence is not strong enough to justify an override.",
    )
    parser.add_argument("--client-timeout", type=float, default=None)
    parser.add_argument("--client-max-retries", type=int, default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    case_ids = list(manifest.get("selected_case_ids", []))
    client = DermOpenAIClient(timeout=args.client_timeout, max_retries=args.client_max_retries)
    policy = load_policy(args.policy_config).to_dict() if args.policy_config else load_stable_policy().to_dict()

    records = load_ham10000_aligned_records_by_case_ids(case_ids, data_root=args.data_root)
    case_inputs = load_ham10000_aligned_case_inputs_by_case_ids(case_ids, data_root=args.data_root)
    run_root = args.output_dir / f"compare_external_ham10000_aligned_subset_{compact_timestamp()}"
    payload = run_aligned_subset_compare(
        dataset_name="HAM10000",
        data_root=args.data_root,
        manifest_path=args.manifest,
        output_dir=run_root,
        output_filename="compare_external_ham10000_aligned_subset.json",
        records=records,
        case_inputs=case_inputs,
        policy=policy,
        policy_label=args.policy_label,
        client=client,
        label_constraint=bool(args.label_constraint),
        dataset_summary={"selected_class_counts": manifest.get("class_counts", {})},
        use_conservative_fusion=bool(args.conservative_fusion),
    )
    print(
        json.dumps(
            {
                "summary": payload["summary"],
                "report_path": str(run_root / "compare_external_ham10000_aligned_subset.json"),
                "run_root": str(run_root),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


def compact_timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


if __name__ == "__main__":
    raise SystemExit(main())
