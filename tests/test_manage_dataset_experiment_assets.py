from __future__ import annotations

import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.manage_dataset_experiment_assets import init_experiment_assets, promote_split_state


class _Args:
    def __init__(self, **kwargs) -> None:
        for key, value in kwargs.items():
            setattr(self, key, value)


def test_init_experiment_assets_creates_isolated_roots(tmp_path: Path) -> None:
    base_policy_path = tmp_path / "base_policy.json"
    base_policy_path.write_text(
        json.dumps(
            {
                "policy_id": "seed_policy",
                "version": 3,
                "status": "stable",
                "planner_policy": {},
                "retrieval_policy": {},
                "evidence_policy": {},
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    args = _Args(
        experiment_id="toy_new_dataset",
        base_policy_config=base_policy_path,
        assets_root=tmp_path / "assets",
        outputs_root=tmp_path / "outputs",
        checkpoints_root=tmp_path / "checkpoints",
    )

    payload = init_experiment_assets(args)

    assert Path(payload["policy_root"]).exists()
    assert Path(payload["split_state_root"]).exists()
    assert Path(payload["output_root"]).exists()
    assert Path(payload["checkpoint_root"]).exists()
    assert Path(payload["env_path"]).exists()
    assert Path(payload["policy_root"]) / "current_stable_policy.json"
    assert (Path(payload["split_state_root"]) / "train" / "experience" / "manifest.json").exists()


def test_promote_split_state_copies_train_assets_into_eval_splits(tmp_path: Path) -> None:
    split_state_root = tmp_path / "split_states"
    train_experience_root = split_state_root / "train" / "experience"
    train_experience_root.mkdir(parents=True, exist_ok=True)
    (train_experience_root / "raw_case_memory.jsonl").write_text("", encoding="utf-8")
    (train_experience_root / "tactical_experience.jsonl").write_text("", encoding="utf-8")
    (train_experience_root / "abstract_experience.jsonl").write_text("", encoding="utf-8")
    (train_experience_root / "indexes").mkdir(parents=True, exist_ok=True)
    (train_experience_root / "indexes" / "case_id_to_raw.json").write_text("{}", encoding="utf-8")
    (train_experience_root / "indexes" / "tactical_by_case.json").write_text("{}", encoding="utf-8")
    (train_experience_root / "indexes" / "abstract_by_type.json").write_text("{}", encoding="utf-8")
    (train_experience_root / "indexes" / "confusion_memory_index.json").write_text("{}", encoding="utf-8")
    (train_experience_root / "manifest.json").write_text(
        json.dumps(
            {
                "version": "experience_v2",
                "state_split": "train",
                "split_aware_version": "experience_bank:train:abc123",
                "state_partition": {
                    "component_id": "experience_bank",
                    "source_path": str(train_experience_root),
                    "split_name": "train",
                    "split_aware_version": "experience_bank:train:abc123",
                },
                "raw_case_count": 0,
                "tactical_count": 0,
                "abstract_count": 0,
                "files": {},
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    train_cognition_path = split_state_root / "train" / "cognition_state.json"
    train_cognition_path.parent.mkdir(parents=True, exist_ok=True)
    train_cognition_path.write_text(
        json.dumps(
            {
                "state_split": "train",
                "state_version": "cognition_state:train:def456",
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    args = _Args(
        split_state_root=split_state_root,
        source_split="train",
        target_splits="val,test",
    )
    payload = promote_split_state(args)

    assert payload["target_splits"] == ["val", "test"]
    for split_name in ("val", "test"):
        manifest = json.loads((split_state_root / split_name / "experience" / "manifest.json").read_text(encoding="utf-8"))
        cognition = json.loads((split_state_root / split_name / "cognition_state.json").read_text(encoding="utf-8"))
        assert manifest["state_split"] == split_name
        assert manifest["state_partition"]["split_name"] == split_name
        assert cognition["state_split"] == split_name
