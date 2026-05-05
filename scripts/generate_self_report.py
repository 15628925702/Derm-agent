from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from cognition.cognition_state import CognitionState

DEFAULT_COGNITION_PATH = PROJECT_ROOT / "state" / "cognition_state.json"
DEFAULT_BATCH_CRITIQUE_PATH = PROJECT_ROOT / "outputs" / "batch_reflection" / "batch_critique.json"
DEFAULT_SELF_REPORT_DIR = PROJECT_ROOT / "state" / "self_reports"


def _load_json(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _build_prompt(cognition: dict, batch_critique: dict) -> str:
    generation = cognition.get("evolution_generation", 0)
    failure_stats = cognition.get("failure_statistics", {})
    total = failure_stats.get("total_cases", 0)
    failed = failure_stats.get("failed_cases", 0)
    confusion_cases = failure_stats.get("confusion_cases", 0)
    accuracy = round((total - failed) / total * 100, 1) if total > 0 else None

    top_confusions = sorted(
        cognition.get("known_confusion_patterns", {}).items(),
        key=lambda x: -x[1],
    )[:5]

    skill_stats = cognition.get("skill_statistics", {})
    top_skills = sorted(
        [(k, v.get("helpful_rate", 0) or 0) for k, v in skill_stats.items()],
        key=lambda x: -x[1],
    )[:5]

    batch_summary = batch_critique.get("source_summary", {})
    rule_count = len(batch_critique.get("rule_candidates", []))
    composite_count = len(batch_critique.get("composite_skill_seed_candidates", []))

    lines = [
        f"You are DermAgent, an AI dermatology diagnostic assistant. Write a concise self-assessment report (200-300 words) in the first person, describing what you have learned and how you have evolved.",
        f"",
        f"Current evolution generation: {generation}",
        f"Cases processed: {total}",
    ]
    if accuracy is not None:
        lines.append(f"Overall accuracy: {accuracy}%")
    lines.append(f"Confusion cases: {confusion_cases}")

    if top_confusions:
        lines.append(f"\nTop confusion pairs (diagnosis→reference: count):")
        for pair, count in top_confusions:
            lines.append(f"  - {pair}: {count} times")

    if top_skills:
        lines.append(f"\nMost effective skills (by helpful rate):")
        for skill, rate in top_skills:
            lines.append(f"  - {skill}: {rate:.2f}")

    if batch_summary:
        lines.append(f"\nLatest batch: {batch_summary.get('total_records', 0)} records, "
                     f"{batch_summary.get('success_count', 0)} successes, "
                     f"{batch_summary.get('failure_count', 0)} failures")
    if rule_count or composite_count:
        lines.append(f"Discovered {rule_count} rule candidates and {composite_count} composite skill seeds.")

    lines += [
        "",
        "Write the self-assessment report now. Be specific about what patterns you have learned, "
        "which skills proved most valuable, and what challenges remain. Do not make a final diagnosis.",
    ]
    return "\n".join(lines)


def generate_self_report(
    cognition_path: Path = DEFAULT_COGNITION_PATH,
    batch_critique_path: Path = DEFAULT_BATCH_CRITIQUE_PATH,
    output_dir: Path = DEFAULT_SELF_REPORT_DIR,
) -> Path:
    cognition_dict = _load_json(cognition_path)
    batch_critique = _load_json(batch_critique_path)
    generation = int(cognition_dict.get("evolution_generation", 0))

    prompt = _build_prompt(cognition_dict, batch_critique)

    base_url = os.getenv("OPENAI_BASE_URL", "http://127.0.0.1:8000/v1")
    api_key = os.getenv("OPENAI_API_KEY", "EMPTY")
    model = os.getenv("DERMAGENT_MODEL_NAME", "Qwen/Qwen2.5-VL-7B-Instruct")

    try:
        from openai import OpenAI
        client = OpenAI(base_url=base_url, api_key=api_key, timeout=120)
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=512,
            temperature=0.7,
        )
        report_text = response.choices[0].message.content or ""
    except Exception as e:
        report_text = f"[Self-report generation failed: {e}]\n\nPrompt used:\n{prompt}"

    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_path = output_dir / f"self_report_gen{generation:04d}_{timestamp}.md"

    header = (
        f"# DermAgent Self-Report — Generation {generation}\n\n"
        f"**Generated**: {datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')}  \n"
        f"**Cases processed**: {cognition_dict.get('failure_statistics', {}).get('total_cases', 0)}  \n\n"
        f"---\n\n"
    )
    out_path.write_text(header + report_text, encoding="utf-8")
    return out_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate a natural-language self-assessment report from DermAgent cognition state.")
    parser.add_argument("--cognition-path", type=Path, default=DEFAULT_COGNITION_PATH)
    parser.add_argument("--batch-critique-path", type=Path, default=DEFAULT_BATCH_CRITIQUE_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_SELF_REPORT_DIR)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    out_path = generate_self_report(
        cognition_path=args.cognition_path,
        batch_critique_path=args.batch_critique_path,
        output_dir=args.output_dir,
    )
    print(json.dumps({"self_report_path": str(out_path)}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
