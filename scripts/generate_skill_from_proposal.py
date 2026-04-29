from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from agent.composite_skill_codegen import generate_skill_code, _slugify

DEFAULT_COMPOSITE_PROPOSALS_DIR = PROJECT_ROOT / "proposals" / "composite_skills"
DEFAULT_CONFUSION_PROPOSALS_DIR = PROJECT_ROOT / "proposals" / "confusion_triggered_skills"
DEFAULT_PENDING_DIR = PROJECT_ROOT / "skills" / "pending"


def _load_json(path: Path) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _iter_proposals(dirs: list[Path]) -> list[tuple[Path, dict]]:
    results = []
    for d in dirs:
        if not d.exists():
            continue
        for p in sorted(d.glob("*.json")):
            proposal = _load_json(p)
            if isinstance(proposal, dict) and proposal.get("review_status") == "pending_review":
                results.append((p, proposal))
    return results


def process_proposal(proposal_path: Path, proposal: dict, pending_dir: Path) -> dict:
    artifacts = proposal.get("proposal_artifacts", {})
    raw_id = artifacts.get("suggested_skill_id") or proposal.get("proposal_id", "composite_skill")
    skill_id = _slugify(raw_id)
    if not skill_id.endswith("_skill"):
        skill_id = skill_id + "_skill"

    code = generate_skill_code(proposal)

    pending_dir.mkdir(parents=True, exist_ok=True)
    skill_file = pending_dir / f"{skill_id}.py"
    meta_file = pending_dir / f"{skill_id}.meta.json"

    skill_file.write_text(code, encoding="utf-8")

    meta = {
        "skill_id": skill_id,
        "proposal_id": proposal.get("proposal_id"),
        "proposal_path": str(proposal_path),
        "source_type": proposal.get("_source_type", "composite"),
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "review_status": "pending_review",
        "skill_file": str(skill_file),
    }
    meta_file.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    return {"skill_id": skill_id, "skill_file": str(skill_file), "meta_file": str(meta_file)}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate pending skill Python files from composite/confusion-triggered proposals."
    )
    parser.add_argument("--composite-proposals-dir", type=Path, default=DEFAULT_COMPOSITE_PROPOSALS_DIR)
    parser.add_argument("--confusion-proposals-dir", type=Path, default=DEFAULT_CONFUSION_PROPOSALS_DIR)
    parser.add_argument("--pending-dir", type=Path, default=DEFAULT_PENDING_DIR)
    parser.add_argument("--proposal-id", type=str, default=None, help="Process only this proposal_id.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    proposals = _iter_proposals([args.composite_proposals_dir, args.confusion_proposals_dir])

    if args.proposal_id:
        proposals = [(p, d) for p, d in proposals if d.get("proposal_id") == args.proposal_id]

    if not proposals:
        print(json.dumps({"generated": [], "message": "No pending_review proposals found."}, ensure_ascii=False, indent=2))
        return 0

    generated = []
    for proposal_path, proposal in proposals:
        result = process_proposal(proposal_path, proposal, args.pending_dir)
        generated.append(result)
        print(f"Generated: {result['skill_file']}", file=sys.stderr)

    print(json.dumps({"generated": generated, "count": len(generated)}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
