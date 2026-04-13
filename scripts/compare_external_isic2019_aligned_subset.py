from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from agent.policy_config import load_policy, load_stable_policy
from dataio.isic2019_aligned_loader import (
    DEFAULT_ISIC2019_ROOT,
    discover_isic2019_aligned_assets,
    load_isic2019_aligned_case_inputs_by_case_ids,
    load_isic2019_aligned_records_by_case_ids,
)
from integrations.openai_client import DermOpenAIClient
from utils.aligned_subset_compare import run_aligned_subset_compare


DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "external_eval" / "isic2019_aligned_subset" / "comparison_smoke"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="External ISIC2019 aligned-subset 3-class compare between direct Qwen and current agent.")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_ISIC2019_ROOT)
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

    assets = discover_isic2019_aligned_assets(args.data_root)
    records = load_isic2019_aligned_records_by_case_ids(case_ids, data_root=args.data_root)
    case_inputs = load_isic2019_aligned_case_inputs_by_case_ids(case_ids, data_root=args.data_root)
    dataset_summary = assets.to_dict()
    dataset_summary["selected_class_counts"] = manifest.get("class_counts", {})

    payload = run_aligned_subset_compare(
        dataset_name="ISIC2019",
        data_root=args.data_root,
        manifest_path=args.manifest,
        output_dir=args.output_dir,
        output_filename="compare_external_isic2019_aligned_subset.json",
        records=records,
        case_inputs=case_inputs,
        policy=policy,
        policy_label=args.policy_label,
        client=client,
        label_constraint=bool(args.label_constraint),
        dataset_summary=dataset_summary,
        use_conservative_fusion=bool(args.conservative_fusion),
    )
    print(
        json.dumps(
            {
                "summary": payload["summary"],
                "report_path": str(args.output_dir / "compare_external_isic2019_aligned_subset.json"),
                "run_root": str(args.output_dir),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
