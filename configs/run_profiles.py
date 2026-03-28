from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class SmokeRunProfile:
    profile_id: str
    description: str
    train_seed_cases: int
    stage_training_limit: int
    training_epochs: int
    val_compare_cases: int
    test_eval_cases: int
    ablation_cases: int
    include_ablations: bool
    include_batch_reflection: bool
    include_paper_exports: bool
    estimated_runtime: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


RUN_PROFILES: dict[str, SmokeRunProfile] = {
    "quick4h_v1": SmokeRunProfile(
        profile_id="quick4h_v1",
        description=(
            "4-hour-ish end-to-end smoke profile: enough train cases to accumulate experience, "
            "enough val/test cases to see whether the system is directionally working, "
            "but still much cheaper than a full long-run study."
        ),
        train_seed_cases=24,
        stage_training_limit=24,
        training_epochs=4,
        val_compare_cases=8,
        test_eval_cases=8,
        ablation_cases=4,
        include_ablations=True,
        include_batch_reflection=True,
        include_paper_exports=True,
        estimated_runtime="about_1_to_4_hours",
    ),
    "medium_signal_no_ablation_v1": SmokeRunProfile(
        profile_id="medium_signal_no_ablation_v1",
        description=(
            "Medium-scale signal-check profile: large enough to start seeing directional gains or regressions "
            "without paying the full ablation cost. Intended for an overnight or workday run."
        ),
        train_seed_cases=96,
        stage_training_limit=96,
        training_epochs=6,
        val_compare_cases=32,
        test_eval_cases=32,
        ablation_cases=0,
        include_ablations=False,
        include_batch_reflection=True,
        include_paper_exports=True,
        estimated_runtime="about_6_to_10_hours",
    ),
    "medium_signal_no_ablation_10h_v1": SmokeRunProfile(
        profile_id="medium_signal_no_ablation_10h_v1",
        description=(
            "Slightly heavier no-ablation medium profile tuned for a fuller overnight signal check. "
            "It increases seeded training cases, controller/retrieval training exposure, and frozen "
            "validation/test coverage without paying the ablation cost."
        ),
        train_seed_cases=120,
        stage_training_limit=120,
        training_epochs=8,
        val_compare_cases=40,
        test_eval_cases=40,
        ablation_cases=0,
        include_ablations=False,
        include_batch_reflection=True,
        include_paper_exports=True,
        estimated_runtime="about_9_to_11_hours",
    ),
    "micro_validation_v1": SmokeRunProfile(
        profile_id="micro_validation_v1",
        description=(
            "Very small validation profile for code-path checking. Not meaningful for performance, "
            "only for end-to-end integration verification."
        ),
        train_seed_cases=2,
        stage_training_limit=2,
        training_epochs=2,
        val_compare_cases=1,
        test_eval_cases=1,
        ablation_cases=1,
        include_ablations=True,
        include_batch_reflection=True,
        include_paper_exports=True,
        estimated_runtime="minutes_to_under_1_hour",
    ),
}


def available_profile_ids() -> list[str]:
    return sorted(RUN_PROFILES.keys())


def get_run_profile(profile_id: str) -> SmokeRunProfile:
    normalized = str(profile_id).strip()
    profile = RUN_PROFILES.get(normalized)
    if profile is None:
        raise KeyError(f"Unknown run profile `{profile_id}`. Available: {available_profile_ids()}")
    return profile
