from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from workflow_evolution.runtime import DEFAULT_APPROVED_WORKFLOW_DIR


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Approve a workflow evolution proposal for optional runtime use.")
    parser.add_argument("--proposal", type=Path, required=True)
    parser.add_argument("--approve", action="store_true", help="Mark proposal approved and copy it into the approved directory.")
    parser.add_argument("--reviewer", default="", help="Human reviewer name or identifier.")
    parser.add_argument("--approved-dir", type=Path, default=DEFAULT_APPROVED_WORKFLOW_DIR)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not args.approve:
        raise SystemExit("Refusing to apply without --approve.")
    payload = _read_json(args.proposal)
    if payload.get("proposal_type") != "workflow_evolution_candidate":
        raise SystemExit("Proposal is not a workflow_evolution_candidate.")
    if not payload.get("proposed_workflow_cell"):
        raise SystemExit("Proposal has no proposed_workflow_cell.")

    payload["review_status"] = "approved"
    payload["reviewed_by"] = args.reviewer or "manual_reviewer"
    payload["reviewed_at"] = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    activation = dict(payload.get("activation", {}) or {})
    activation["enabled"] = True
    activation["runtime_env_required"] = "DERMAGENT_ENABLE_WORKFLOW_EVOLUTION=1"
    payload["activation"] = activation

    args.approved_dir.mkdir(parents=True, exist_ok=True)
    output_path = args.approved_dir / f"{payload['proposal_id']}.json"
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    manifest_path = args.approved_dir / "README.md"
    if not manifest_path.exists():
        manifest_path.write_text(
            "# Approved Workflow Evolution Proposals\n\n"
            "Files in this directory are ignored unless `DERMAGENT_ENABLE_WORKFLOW_EVOLUTION=1` is set.\n",
            encoding="utf-8",
        )
    print(
        json.dumps(
            {
                "approved_proposal_path": str(output_path),
                "runtime_enabled_only_when": "DERMAGENT_ENABLE_WORKFLOW_EVOLUTION=1",
                "note": "The proposal is approved but remains inactive unless the runtime env switch is enabled.",
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


def _read_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise SystemExit("Proposal JSON must be an object.")
    return payload


if __name__ == "__main__":
    raise SystemExit(main())
