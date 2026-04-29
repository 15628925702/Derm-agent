from __future__ import annotations

import json
import os
from collections import defaultdict
from pathlib import Path
from typing import Any

DEFAULT_ADAPTATION_DIR = Path("/root/DermAgent/state/meta_learning/adaptations")
DEFAULT_MIN_CASES = 5


class FewShotAdapter:
    def __init__(self, adaptation_dir: Path = DEFAULT_ADAPTATION_DIR) -> None:
        self.adaptation_dir = adaptation_dir

    def adapt(self, cases: list[dict[str, Any]], dataset_name: str, min_cases: int = DEFAULT_MIN_CASES) -> dict[str, Any]:
        if len(cases) < min_cases:
            return {"warning": f"Only {len(cases)} cases provided, minimum is {min_cases}. Adaptation may be unreliable."}

        skill_success: dict[str, int] = defaultdict(int)
        skill_total: dict[str, int] = defaultdict(int)
        label_skill_map: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))

        for case in cases:
            skill_assessments = case.get("skill_assessments") or case.get("reflection", {}).get("skill_assessments", [])
            label = str(case.get("reference_label") or case.get("case_input", {}).get("reference_label", "unknown"))
            for assessment in skill_assessments:
                skill_name = str(assessment.get("skill_name", "")).strip()
                if not skill_name:
                    continue
                skill_total[skill_name] += 1
                if assessment.get("helpfulness") in {"success", "partially_helpful"}:
                    skill_success[skill_name] += 1
                    label_skill_map[label][skill_name] += 1

        retrieval_bias: dict[str, float] = {}
        for skill, total in skill_total.items():
            if total == 0:
                continue
            rate = skill_success[skill] / total
            if rate > 0.6:
                retrieval_bias[skill] = round(rate, 3)

        workflow_preferences_patch: dict[str, Any] = {}
        for label, skill_counts in label_skill_map.items():
            top_skills = sorted(skill_counts.items(), key=lambda x: -x[1])[:5]
            if top_skills:
                workflow_preferences_patch[f"meta_learning__{dataset_name}__{label}"] = {
                    "preferred_skills": [s for s, _ in top_skills],
                    "avg_accuracy": 0.0,
                    "case_count": len(cases),
                }

        adaptation = {
            "dataset_name": dataset_name,
            "case_count": len(cases),
            "retrieval_bias": retrieval_bias,
            "workflow_preferences_patch": workflow_preferences_patch,
            "top_skills_by_label": {
                label: sorted(counts.items(), key=lambda x: -x[1])[:5]
                for label, counts in label_skill_map.items()
            },
        }
        return adaptation

    def save_adaptation(self, dataset_name: str, adaptation: dict[str, Any]) -> Path:
        self.adaptation_dir.mkdir(parents=True, exist_ok=True)
        out_path = self.adaptation_dir / f"{dataset_name}.json"
        out_path.write_text(json.dumps(adaptation, ensure_ascii=False, indent=2), encoding="utf-8")
        return out_path

    @staticmethod
    def load_adaptation(dataset_name: str, adaptation_dir: Path = DEFAULT_ADAPTATION_DIR) -> dict[str, Any] | None:
        path = adaptation_dir / f"{dataset_name}.json"
        if not path.exists():
            return None
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return None

    @staticmethod
    def is_enabled() -> bool:
        return os.getenv("DERMAGENT_META_LEARNING_ENABLED", "").strip() in {"1", "true", "yes"}
