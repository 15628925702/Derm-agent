from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from agent.auditability_scoring import auditability_root, load_rules, slugify, write_audit_run


def default_output_root(inputs: list[Path]) -> Path:
    if not inputs:
        return auditability_root() / "auditability_run"
    stem = slugify("_".join(path.name for path in inputs[:3]))
    return auditability_root() / stem


def main() -> int:
    parser = argparse.ArgumentParser(description="Score evidence auditability from DermAgent result folders or extracted case bundles.")
    parser.add_argument("--input", dest="inputs", type=Path, action="append", required=True)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--rules-path", type=Path)
    args = parser.parse_args()

    inputs = [path.resolve() for path in list(args.inputs or [])]
    output_root = args.output_root.resolve() if args.output_root else default_output_root(inputs)
    rules = load_rules(args.rules_path.resolve()) if args.rules_path else load_rules()
    manifest = write_audit_run(input_paths=inputs, output_root=output_root, rules=rules)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
