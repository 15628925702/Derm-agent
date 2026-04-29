from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

DEFAULT_PENDING_DIR = PROJECT_ROOT / "skills" / "pending"
DEFAULT_SKILLS_DIR = PROJECT_ROOT / "skills"
REGISTRY_PATH = PROJECT_ROOT / "skills" / "registry.py"

# Marker lines in registry.py that approve_skill.py appends after
_IMPORT_ANCHOR = "from skills.uncertainty import UncertaintyAssessmentSkill"
_REGISTRY_ANCHOR = "        EscalationRecommendationSkill(),"


def _load_json(path: Path) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _find_meta_by_skill_id(pending_dir: Path, skill_id: str) -> Path | None:
    candidate = pending_dir / f"{skill_id}.meta.json"
    return candidate if candidate.exists() else None


def _find_meta_by_proposal_id(pending_dir: Path, proposal_id: str) -> tuple[str, Path] | None:
    for meta_file in pending_dir.glob("*.meta.json"):
        meta = _load_json(meta_file)
        if meta.get("proposal_id") == proposal_id:
            return meta.get("skill_id", ""), meta_file
    return None


def _class_name_from_skill_id(skill_id: str) -> str:
    parts = re.split(r"[_\-\s]+", skill_id.lower().replace("_skill", ""))
    return "".join(p.capitalize() for p in parts if p) + "Skill"


def _module_name(skill_id: str) -> str:
    return skill_id  # file is skills/{skill_id}.py → module skills.{skill_id}


def _patch_registry(skill_id: str, registry_path: Path) -> None:
    content = registry_path.read_text(encoding="utf-8")
    class_name = _class_name_from_skill_id(skill_id)
    module = _module_name(skill_id)

    import_line = f"from skills.{module} import {class_name}"
    instance_line = f"        {class_name}(),"

    if import_line in content:
        print(f"  Registry already contains import for {class_name}, skipping import patch.")
    else:
        content = content.replace(
            _IMPORT_ANCHOR,
            f"{_IMPORT_ANCHOR}\n{import_line}",
        )

    if instance_line in content:
        print(f"  Registry already contains instance for {class_name}, skipping instance patch.")
    else:
        content = content.replace(
            _REGISTRY_ANCHOR,
            f"{_REGISTRY_ANCHOR}\n{instance_line}",
        )

    registry_path.write_text(content, encoding="utf-8")


def _update_proposal_status(proposal_path: str, skill_id: str) -> None:
    p = Path(proposal_path)
    if not p.exists():
        return
    try:
        proposal = json.loads(p.read_text(encoding="utf-8"))
        proposal["review_status"] = "approved"
        p.write_text(json.dumps(proposal, ensure_ascii=False, indent=2), encoding="utf-8")
    except Exception as e:
        print(f"  Warning: could not update proposal status: {e}", file=sys.stderr)


def approve(skill_id: str, pending_dir: Path, skills_dir: Path, registry_path: Path) -> dict:
    meta_file = _find_meta_by_skill_id(pending_dir, skill_id)
    if meta_file is None:
        return {"error": f"No pending skill found for skill_id={skill_id!r}. Run generate_skill_from_proposal.py first."}

    meta = _load_json(meta_file)
    skill_py_src = pending_dir / f"{skill_id}.py"
    if not skill_py_src.exists():
        return {"error": f"Skill file not found: {skill_py_src}"}

    skill_py_dst = skills_dir / f"{skill_id}.py"
    shutil.copy2(skill_py_src, skill_py_dst)
    print(f"  Copied: {skill_py_src} → {skill_py_dst}")

    _patch_registry(skill_id, registry_path)
    print(f"  Patched registry: {registry_path}")

    proposal_path = meta.get("proposal_path", "")
    if proposal_path:
        _update_proposal_status(proposal_path, skill_id)
        print(f"  Updated proposal status → approved: {proposal_path}")

    skill_py_src.unlink(missing_ok=True)
    meta_file.unlink(missing_ok=True)
    print(f"  Removed pending files.")

    return {
        "approved_skill_id": skill_id,
        "skill_file": str(skill_py_dst),
        "registry_patched": str(registry_path),
        "proposal_path": proposal_path,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Approve a pending skill: copy to skills/, patch registry.py, mark proposal as approved."
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--skill-id", type=str, help="skill_id of the pending skill (e.g. mel_nev_v2_skill).")
    group.add_argument("--proposal-id", type=str, help="proposal_id to look up the skill_id automatically.")
    parser.add_argument("--pending-dir", type=Path, default=DEFAULT_PENDING_DIR)
    parser.add_argument("--skills-dir", type=Path, default=DEFAULT_SKILLS_DIR)
    parser.add_argument("--registry-path", type=Path, default=REGISTRY_PATH)
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    skill_id = args.skill_id
    if skill_id is None:
        result = _find_meta_by_proposal_id(args.pending_dir, args.proposal_id)
        if result is None:
            print(json.dumps({"error": f"No pending skill found for proposal_id={args.proposal_id!r}."}, indent=2))
            return 1
        skill_id = result[0]

    result = approve(skill_id, args.pending_dir, args.skills_dir, args.registry_path)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if "error" not in result else 1


if __name__ == "__main__":
    sys.exit(main())
